import sys
import os
import fcntl
import mmap
import ctypes
import struct
import errno
import argparse
import xml.etree.ElementTree as ET
from gen_parser import Parser, Reg, Enum

# ioctl direction constants
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

# ioctl macro
def _IOC(dir_, type_, nr, size):
    return ((dir_ << 30) | (size << 16) | (ord(type_) << 8) | nr)

def _IOWR(type_, nr, struct_):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, ctypes.sizeof(struct_))

# DRM constants
DRM_IOC_BASE = 'd'
DRM_COMMAND_BASE = 0x40

class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'

    @staticmethod
    def highlight_hex(text):
        """Highlight hexadecimal numbers and addresses"""
        import re
        # Highlight addresses in brackets [0x...]
        text = re.sub(r'(\[0x[0-9a-fA-F]+\])', f'{Colors.CYAN}\\1{Colors.RESET}', text)
        # Highlight hex values 0x...
        text = re.sub(r'(0x[0-9a-fA-F]+)', f'{Colors.CYAN}\\1{Colors.RESET}', text)
        return text

    @staticmethod
    def highlight_register(text):
        """Highlight register names like REG_CORE_SOMETHING"""
        import re
        text = re.sub(r'(REG_[A-Z_]+)', f'{Colors.GREEN}\\1{Colors.RESET}', text)
        return text

    @staticmethod
    def highlight_keyword(text):
        """Highlight keywords like EMIT, ENABLED, DISABLED"""
        import re
        keywords = ['EMIT', 'ENABLED', 'DISABLED', 'Unknown']
        for keyword in keywords:
            text = re.sub(rf'\b({keyword})\b', f'{Colors.YELLOW}\\1{Colors.RESET}', text)
        return text

    @staticmethod
    def highlight_destination(text):
        """Highlight destinations like PC, CNA, CORE, DPU"""
        import re
        destinations = ['PC', 'CNA', 'CORE', 'DPU', 'DPU_RDMA', 'PPU', 'PPU_RDMA']
        for dest in destinations:
            text = re.sub(rf'\b({dest})\b', f'{Colors.YELLOW}\\1{Colors.RESET}', text)
        return text

    @staticmethod
    def highlight_instruction(text):
        """Highlight instruction data like lsb 1234567890abcdef"""
        import re
        text = re.sub(r'(lsb\s+[0-9a-fA-F]+)', f'{Colors.WHITE}\\1{Colors.RESET}', text)
        return text

    @staticmethod
    def apply_all_highlighting(text):
        """Apply all highlighting to text"""
        text = Colors.highlight_hex(text)
        text = Colors.highlight_register(text)
        text = Colors.highlight_keyword(text)
        text = Colors.highlight_destination(text)
        text = Colors.highlight_instruction(text)
        return text


# DRM structs
class drm_version(ctypes.Structure):
    _fields_ = [
        ("version_major", ctypes.c_int),
        ("version_minor", ctypes.c_int),
        ("version_patchlevel", ctypes.c_int),
        ("name_len", ctypes.c_size_t),
        ("name", ctypes.POINTER(ctypes.c_char)),  # Changed to POINTER(c_char)
        ("date_len", ctypes.c_size_t),
        ("date", ctypes.POINTER(ctypes.c_char)),  # Changed to POINTER(c_char)
        ("desc_len", ctypes.c_size_t),
        ("desc", ctypes.POINTER(ctypes.c_char)),  # Changed to POINTER(c_char)
    ]

class drm_unique(ctypes.Structure):
    _fields_ = [
        ("unique_len", ctypes.c_size_t),
        ("unique", ctypes.POINTER(ctypes.c_char)),  # Changed to POINTER(c_char)
    ]

class drm_gem_open(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

# DRM ioctl numbers
DRM_IOCTL_VERSION = _IOWR('d', 0x00, drm_version)
DRM_IOCTL_GET_UNIQUE = _IOWR('d', 0x01, drm_unique)
DRM_IOCTL_GEM_OPEN = _IOWR('d', 0x0b, drm_gem_open)
RKNPU_MEM_MAP = 0x03
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', DRM_COMMAND_BASE + RKNPU_MEM_MAP, rknpu_mem_map)

class Register:
    def __init__(self, offset, name, domain):
        self.offset = offset
        self.name = name
        self.domain = domain
        self.full_name = f"{domain}_{name}"
        self.bitset = BitSet()

class BitField:
    def __init__(self, name, low, high, type_name):
        self.name = name
        self.low = low
        self.high = high
        self.type = type_name
        self.pos = low if low == high else None

class BitSet:
    def __init__(self):
        self.fields = []

class RegisterDecoder:
    def __init__(self, xml_file):
        self.registers = {}
        self.domains = {}
        self.parse_xml(xml_file)

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Parse target domains
        for enum in root.findall(".//enum[@name='target']"):
            for value in enum.findall("value"):
                name = value.get("name")
                val = int(value.get("value"), 16)
                self.domains[name] = val

        # Parse registers
        for domain in root.findall("domain"):
            domain_name = domain.get("name")

            for reg in domain.findall("reg32"):
                offset = int(reg.get("offset"), 16)
                name = reg.get("name")

                register = Register(offset, name, domain_name)

                # Parse bitfields
                for bitfield in reg.findall("bitfield"):
                    bf_name = bitfield.get("name")
                    low_str = bitfield.get("low")
                    high_str = bitfield.get("high")
                    type_name = bitfield.get("type")

                    if low_str is not None and high_str is not None:
                        low = int(low_str)
                        high = int(high_str)
                    else:
                        # Handle single bit fields (pos attribute)
                        pos_str = bitfield.get("pos")
                        if pos_str is not None:
                            pos = int(pos_str)
                            low = pos
                            high = pos
                        else:
                            continue

                    bf = BitField(bf_name, low, high, type_name)
                    register.bitset.fields.append(bf)

                self.registers[offset] = register

    def decode_register(self, offset, value, target):
        if offset not in self.registers:
            return None

        reg = self.registers[offset]

        # Skip target validation for now - let decode.py handle domain matching
        # expected_domain = self.domains.get(reg.domain, 0)
        # if target & 0xfffffffe != expected_domain:
        #     return None

        decoded = {
            'register': reg,
            'value': value,
            'fields': []
        }

        # Decode bitfields
        if value == 0 or len(reg.bitset.fields) == 1:
            decoded['fields'].append({
                'name': 'RAW_VALUE',
                'value': value,
                'formatted': f'0x{value:08x}'
            })
        else:
            for field in reg.bitset.fields:
                if field.type == "boolean":
                    field_value = 1 if (1 << field.low) & value else 0
                    if field_value:
                        decoded['fields'].append({
                            'name': f"{reg.full_name.upper()}_{field.name.upper()}",
                            'value': field_value,
                            'formatted': 'ENABLED' if field_value else 'DISABLED'
                        })
                elif field.type == "uint":
                    mask = (1 << (field.high - field.low + 1)) - 1
                    field_value = (value >> field.low) & mask
                    if field_value != 0:
                        decoded['fields'].append({
                            'name': f"{reg.full_name.upper()}_{field.name.upper()}",
                            'value': field_value,
                            'formatted': f'{field_value}'
                        })

        return decoded

# Register decoder will be initialized only when decode mode is used

def decode_dump_file(xml_file, dump_file):
    """Decode a register dump file using XML register definitions"""
    # Parse XML file
    p = Parser()
    try:
        p.parse("", xml_file)
    except Exception as e:
        print(f"Error parsing XML file: {e}", file=sys.stderr)
        sys.exit(1)

    regs = {}
    for e in p.file:
        if isinstance(e, Reg):
            regs[e.offset] = e

    domains = {}
    for e in p.file:
        if isinstance(e, Enum):
            if e.name == "target":
                for name, val in e.values:
                    domains[name] = val

    # Read and decode dump file
    with open(dump_file, 'rb') as f:
        for i in range(0, os.path.getsize(dump_file) // 8):
            cmd = f.read(8)
            (offset, value, target) = struct.unpack("<hIh", cmd)

            # Keep offset as read from file (no conversion needed)

            if offset in regs.keys():
                reg = regs[offset]

                        # Skip domain validation warnings as requested
                        # if (target & 0xfffffffe) != domains[reg.domain]:
                        #     print("WARNING: target 0x%x doesn't match register's domain 0x%x" % (target, domains[reg.domain]))

                emit_str = "EMIT(REG_%s, " % regs[offset].full_name.upper()
                first = True
                if value == 0 or len(reg.bitset.fields) == 1:
                    emit_str += "0x%x" % value
                else:
                    for field in reg.bitset.fields:
                        if field.type == "boolean":
                            if 1 << field.high & value:
                                if not first:
                                    emit_str += " | "
                                emit_str += "%s_%s" % (reg.full_name.upper(), field.name.upper())
                                first = False
                        elif field.type == "uint":
                            field_value = (value & mask(field.low, field.high)) >> field.low
                            if field_value != 0:
                                if not first:
                                    emit_str += " | "
                                emit_str += "%s_%s(%d)" % (reg.full_name.upper(), field.name.upper(), field_value)
                                first = False
                emit_str += ");"
                print(Colors.apply_all_highlighting(emit_str))
            else:
                output = "%x %x %x" % (target, offset, value)
                print(Colors.apply_all_highlighting(output))

def mask(low, high):
    return ((0xffffffffffffffff >> (64 - (high + 1 - low))) << low)

def register3_stuff(fd):
    try:
        gopen = drm_gem_open()
        gopen.name = 3
        fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, gopen)
        output = f"gem open got 0 {gopen.handle} {gopen.size}"
        print(Colors.apply_all_highlighting(output))

        mem_map = rknpu_mem_map()
        mem_map.handle = gopen.handle
        fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
        output = f"memmap returned 0 {hex(mem_map.offset)}"
        print(Colors.apply_all_highlighting(output))

        instr_map = mmap.mmap(fd, gopen.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
        print(f"mmap returned {instr_map}")

        # Create dump directory if it doesn't exist
        os.makedirs("dump", exist_ok=True)

        with open("dump/gem3-dump", "wb") as fdo:
            fdo.write(instr_map[:])

        a = 0
        for i in range(gopen.size // 40):
            block = instr_map[i * 40 : (i + 1) * 40]
            instrs = struct.unpack("<10I", block)
            output = f"[{i}] {instrs[7] - a}"
            print(Colors.apply_all_highlighting(output))
            output = f"\t{hex(instrs[8])}"
            print(Colors.apply_all_highlighting(output))
            a = instrs[7]
            if instrs[8] == 0:
                break

        instr_map.close()
    except OSError as e:
        print(f"Error in register3_stuff: {os.strerror(e.errno)}")
        sys.exit(2)

def dump_gem_flink(fd, flink_name):
    try:
        print(f"\n{'='*50}")
        print(f"Processing GEM Flink {flink_name}")
        print(f"{'='*50}")

        gopen = drm_gem_open()
        gopen.name = flink_name
        fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, gopen)
        output = f"gem flink {flink_name}: ret=0 handle={gopen.handle} size={gopen.size}"
        print(Colors.apply_all_highlighting(output))

        mem_map = rknpu_mem_map()
        mem_map.handle = gopen.handle
        fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
        output = f"memmap returned 0 {hex(mem_map.offset)}"
        print(Colors.apply_all_highlighting(output))

        instr_map = mmap.mmap(fd, gopen.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
        print(f"mmap returned {instr_map}")

        print(f"GEM via flink {flink_name}")
        total_blocks = gopen.size // 16
        for i in range(0, gopen.size, 16):
            block = instr_map[i : i + 16]
            here = struct.unpack("<4I", block)
            # Check for trailing zeros - when we encounter all-zero blocks
            if all(x == 0 for x in here):
                remaining_blocks = total_blocks - i // 16
                remaining_bytes = remaining_blocks * 16
                output = f"[{i:08x}] = {here[0]:08x} {here[1]:08x} {here[2]:08x} {here[3]:08x}"
                print(Colors.apply_all_highlighting(output))
                output = f"... {remaining_blocks} blocks ({remaining_bytes} bytes) from 0x{i:08x} to 0x{gopen.size-1:08x} are all zeros"
                print(Colors.apply_all_highlighting(output))
                break
            output = f"[{i:08x}] = {here[0]:08x} {here[1]:08x} {here[2]:08x} {here[3]:08x}"
            print(Colors.apply_all_highlighting(output))

        instr_map.close()
    except OSError as e:
        print(f"Failed in dump_gem_flink for {flink_name}: {os.strerror(e.errno)}")
        return

def dump_gem_for_decode(fd, flink_name):
    print(f"\n{'='*50}")
    print(f"Processing GEM Flink {flink_name} for Register Decode")
    print(f"{'='*50}")

    try:
        gopen = drm_gem_open()
        gopen.name = flink_name
        fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, gopen)
        print(f"gem flink {flink_name}: ret=0 handle={gopen.handle} size={gopen.size}")
        print(f"Successfully opened GEM via flink {flink_name}")

        mem_map = rknpu_mem_map()
        mem_map.handle = gopen.handle
        fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
        print(f"memmap returned 0 {hex(mem_map.offset)}")

        instr_map = mmap.mmap(fd, gopen.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
        print(f"mmap returned {instr_map}")

        # Create dump directory if it doesn't exist
        os.makedirs("dump", exist_ok=True)

        dump_filename = f"dump/gem{flink_name}-dump"
        regdump_filename = f"dump/gem{flink_name}_regdump.bin"

        with open(dump_filename, "wb") as fdo:
            fdo.write(instr_map[:])

        with open(regdump_filename, "wb") as dump_fd:
            print(f"Successfully created dump/gem{flink_name}_regdump.bin")
            for i in range(gopen.size // 8):
                instr = struct.unpack("<Q", instr_map[i * 8 : (i + 1) * 8])[0]

                # Skip all-zero entries
                if instr == 0:
                    continue

                val = (instr >> 16) & 0xffffffff
                high = (instr >> 48) & 0xffff
                low = instr & 0xffff

                target = 0
                dst = "noone"
                if (instr >> 56) & 1:
                    target = 0x100
                    dst = "PC"
                elif (instr >> 57) & 1:
                    target = 0x200
                    dst = "CNA"
                elif (instr >> 59) & 1:
                    target = 0x800
                    dst = "CORE"
                elif (instr >> 60) & 1:
                    target = 0x1000
                    dst = "DPU"
                elif (instr >> 61) & 1:
                    target = 0x2000
                    dst = "DPU_RDMA"
                elif (instr >> 62) & 1:
                    target = 0x4000
                    dst = "PPU"
                elif (instr >> 63) & 1:
                    target = 0x8000
                    dst = "PPU_RDMA"

                # Map target to domain value for decode.py compatibility
                domain_target = target
                if target == 0x100:
                    domain_target = 0x100  # PC
                elif target == 0x200:
                    domain_target = 0x200  # CNA
                elif target == 0x800:
                    domain_target = 0x800  # CORE
                elif target == 0x1000:
                    domain_target = 0x1000  # DPU
                elif target == 0x2000:
                    domain_target = 0x2000  # DPU_RDMA
                elif target == 0x4000:
                    domain_target = 0x4000  # PPU
                elif target == 0x8000:
                    domain_target = 0x8000  # PPU_RDMA

                op_en = "Enable op" if (instr >> 55) & 1 else ""

                # Try to decode register if XML file is available and we find a valid register
                if os.path.exists("registers.xml"):
                    try:
                        if not hasattr(dump_gem_for_decode, 'parser'):
                            dump_gem_for_decode.parser = Parser()
                            dump_gem_for_decode.parser.parse("", "registers.xml")

                        regs = {}
                        for e in dump_gem_for_decode.parser.file:
                            if isinstance(e, Reg):
                                regs[e.offset] = e

                        domains = {}
                        for e in dump_gem_for_decode.parser.file:
                            if isinstance(e, Enum):
                                if e.name == "target":
                                    for name, val in e.values:
                                        domains[name] = val

                        if low in regs:
                            reg = regs[low]

                            # Skip domain validation warnings as requested
                            # if (target & 0xfffffffe) != domains[reg.domain]:
                            #     print("WARNING: target 0x%x doesn't match register's domain 0x%x" % (target, domains[reg.domain]))

                            # Show decoded register info and EMIT in ultra-compact format with alignment
                            reg_info = f"[{8 * i + 0xffef0000:x}] lsb {instr:016x} - {dst}"

                            emit_str = f"EMIT(REG_{regs[low].full_name.upper()}, "
                            first = True
                            if val == 0 or len(reg.bitset.fields) == 1:
                                emit_str += f"0x{val:08x}"
                            else:
                                for field in reg.bitset.fields:
                                    if field.type == "boolean":
                                        if 1 << field.high & val:
                                            if not first:
                                                emit_str += " | "
                                            emit_str += f"{reg.full_name.upper()}_{field.name.upper()}"
                                            first = False
                                    elif field.type == "uint":
                                        field_value = (val & mask(field.low, field.high)) >> field.low
                                        if field_value != 0:
                                            if not first:
                                                emit_str += " | "
                                            emit_str += f"{reg.full_name.upper()}_{field.name.upper()}({field_value})"
                                            first = False
                            emit_str += ");"

                            # Align EMIT statements for better readability
                            if len(reg_info) < 50:
                                spacing = " " * (50 - len(reg_info))
                            else:
                                spacing = " "
                            output = f"{reg_info}{spacing}{emit_str}"
                            print(Colors.apply_all_highlighting(output))
                        else:
                            # Show raw register info for unknown registers in ultra-compact format
                            output = f"[{8 * i + 0xffef0000:x}] lsb {instr:016x} - {dst} Unknown"
                            print(Colors.apply_all_highlighting(output))
                    except Exception as e:
                        # If XML parsing fails, show raw register info in ultra-compact format
                        output = f"[{8 * i + 0xffef0000:x}] lsb {instr:016x} - {dst} Unknown"
                        print(Colors.apply_all_highlighting(output))
                        pass
                else:
                    # Show raw register info when no XML file in ultra-compact format
                    output = f"[{8 * i + 0xffef0000:x}] lsb {instr:016x} - {dst} Unknown"
                    print(Colors.apply_all_highlighting(output))

                # Convert to signed short for compatibility with decode.py
                signed_low = low if low <= 32767 else low - 65536
                signed_target = target if target <= 32767 else target - 65536

                dump_fd.write(struct.pack("<h", signed_low))
                dump_fd.write(struct.pack("<I", val))
                dump_fd.write(struct.pack("<h", signed_target))

        print(f"Dumped {gopen.size // 8} register commands to {regdump_filename}")

        instr_map.close()
    except OSError as e:
        print(f"Failed in dump_gem_for_decode for {flink_name}: {os.strerror(e.errno)}")
        if e.errno == errno.ENOENT or e.errno == errno.EINVAL:
            print("The GEM objects may not be available. Try running a matmul program first.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPU Register Dumper and Decoder')
    parser.add_argument('--xml', type=str, help='XML register definition file')
    parser.add_argument('--dump', type=str, help='Binary dump file to decode')
    parser.add_argument('gems', nargs='*', type=int, help='GEM object numbers to dump (default: 1, 2)')

    args = parser.parse_args()

    # If --xml and --dump are provided, run in decode mode
    if args.xml and args.dump:
        if not os.path.exists(args.xml):
            print(f"Error: XML file '{args.xml}' not found", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.dump):
            print(f"Error: Dump file '{args.dump}' not found", file=sys.stderr)
            sys.exit(1)
        decode_dump_file(args.xml, args.dump)
        sys.exit(0)

    # Original dump functionality
    try:
        fd = os.open("/dev/dri/card1", os.O_RDWR)
    except OSError as e:
        print(f"Failed to open /dev/dri/card1: {os.strerror(e.errno)}")
        sys.exit(1)

    try:
        # Initialize buffers for drm_version
        name_buf = (ctypes.c_char * 256)()
        date_buf = (ctypes.c_char * 256)()
        desc_buf = (ctypes.c_char * 256)()

        # Initialize drm_version structure
        dv = drm_version()
        dv.version_major = 0
        dv.version_minor = 0
        dv.version_patchlevel = 0
        dv.name_len = 256
        dv.name = ctypes.cast(name_buf, ctypes.POINTER(ctypes.c_char))
        dv.date_len = 256
        dv.date = ctypes.cast(date_buf, ctypes.POINTER(ctypes.c_char))
        dv.desc_len = 256
        dv.desc = ctypes.cast(desc_buf, ctypes.POINTER(ctypes.c_char))

        # Perform DRM_IOCTL_VERSION
        fcntl.ioctl(fd, DRM_IOCTL_VERSION, dv)
        name_str = ctypes.string_at(dv.name, dv.name_len).decode('utf-8', errors='ignore').rstrip('\x00')
        date_str = ctypes.string_at(dv.date, dv.date_len).decode('utf-8', errors='ignore').rstrip('\x00')
        desc_str = ctypes.string_at(dv.desc, dv.desc_len).decode('utf-8', errors='ignore').rstrip('\x00')
        print(f"drm name is {name_str} - {date_str} - {desc_str}")

        # Get unique identifier
        unique_buf = (ctypes.c_char * 256)()
        du = drm_unique()
        du.unique_len = 256
        du.unique = ctypes.cast(unique_buf, ctypes.POINTER(ctypes.c_char))
        fcntl.ioctl(fd, DRM_IOCTL_GET_UNIQUE, du)
        unique_str = ctypes.string_at(du.unique, du.unique_len).decode('utf-8', errors='ignore').rstrip('\x00')
        print(f"du is {unique_str}")
    except OSError as e:
        print(f"Error in DRM ioctl: {os.strerror(e.errno)}")
        os.close(fd)
        sys.exit(2)

    # Determine which GEMs to process
    if args.gems:
        gems_to_process = args.gems
    else:
        gems_to_process = [1, 2]  # Default behavior

    print("Dumping specified GEM objects...")
    for gem_num in gems_to_process:
        if gem_num > 0:
            print(f"\n=== Processing GEM {gem_num} ===")
            dump_gem_flink(fd, gem_num)
            dump_gem_for_decode(fd, gem_num)
        else:
            print(f"Invalid GEM number: {gem_num}", file=sys.stderr)

    os.close(fd)
