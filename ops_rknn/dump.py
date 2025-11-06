import sys, os, fcntl, mmap, ctypes, struct, argparse, re, xml.parsers.expat, collections, subprocess

class Error(Exception):
    def __init__(self, message):
        self.message = message

class Enum(object):
    def __init__(self, name):
        self.name = name
        self.values = []

    def has_name(self, name):
        for (n, value) in self.values:
            if n == name:
                return True
        return False

class Field(object):
    def __init__(self, name, low, high, shr, type, parser):
        self.name = name
        self.low = low
        self.high = high
        self.shr = shr
        self.type = type

def mask(low, high):
    return ((0xffffffffffffffff >> (64 - (high + 1 - low))) << low)

def field_name(reg, f):
    if f.name:
        name = f.name.lower()
    else:
        name = reg.name.lower()

    if (name in [ "double", "float", "int" ]) or not (name[0].isalpha()):
            name = "_" + name

    return name

class Reg(object):
    def __init__(self, attrs, domain, array, bit_size):
        self.name = attrs["name"]
        self.domain = domain
        self.array = array
        self.offset = int(attrs["offset"], 0)
        self.type = None
        self.bit_size = bit_size
        if array:
            self.name = array.name + "_" + self.name
        self.full_name = self.domain + "_" + self.name

class Bitset(object):
    def __init__(self, name, template):
        self.name = name
        self.inline = False
        if template:
            self.fields = template.fields[:]
        else:
            self.fields = []

class Parser(object):
    def __init__(self):
        self.current_array = None
        self.current_domain = None
        self.current_prefix = None
        self.current_prefix_type = None
        self.current_stripe = None
        self.current_bitset = None
        self.current_bitsize = 32
        self.current_varset = None
        self.variant_regs = {}
        self.usage_regs = collections.defaultdict(list)
        self.bitsets = {}
        self.enums = {}
        self.variants = set()
        self.file = []
        self.xml_files = []
        self.copyright_year = None
        self.authors = []
        self.license = None

    def error(self, message):
        parser, filename = self.stack[-1]
        return Error("%s:%d:%d: %s" % (filename, parser.CurrentLineNumber, parser.CurrentColumnNumber, message))

    def prefix(self, variant=None):
        if self.current_prefix_type == "variant" and variant:
            return variant
        elif self.current_stripe:
            return self.current_stripe + "_" + self.current_domain
        elif self.current_prefix:
            return self.current_prefix + "_" + self.current_domain
        else:
            return self.current_domain

    def parse_field(self, name, attrs):
        try:
            if "pos" in attrs:
                high = low = int(attrs["pos"], 0)
            elif "high" in attrs and "low" in attrs:
                high = int(attrs["high"], 0)
                low = int(attrs["low"], 0)
            else:
                low = 0
                high = self.current_bitsize - 1

            if "type" in attrs:
                type = attrs["type"]
            else:
                type = None

            if "shr" in attrs:
                shr = int(attrs["shr"], 0)
            else:
                shr = 0

            b = Field(name, low, high, shr, type, self)
            self.current_bitset.fields.append(b)
        except ValueError as e:
            raise self.error(e)

    def parse_varset(self, attrs):
        varset = self.current_varset
        if "varset" in attrs:
            varset = self.enums[attrs["varset"]]
        return varset

    def add_all_variants(self, reg, attrs, parent_variant):
        variant = self.parse_variants(attrs)
        if not variant:
            variant = parent_variant

        if reg.name not in self.variant_regs:
            self.variant_regs[reg.name] = {}
        else:
            v = next(iter(self.variant_regs[reg.name]))
            assert self.variant_regs[reg.name][v].bit_size == reg.bit_size

        self.variant_regs[reg.name][variant] = reg

    def add_all_usages(self, reg, usages):
        if not usages:
            return

        for usage in usages:
            self.usage_regs[usage].append(reg)

        self.variants.add(reg.domain)

    def do_parse(self, filename):
        filepath = os.path.abspath(filename)
        if filepath in self.xml_files:
            return
        self.xml_files.append(filepath)
        file = open(filename, "rb")
        parser = xml.parsers.expat.ParserCreate()
        self.stack.append((parser, filename))
        parser.StartElementHandler = self.start_element
        parser.EndElementHandler = self.end_element
        parser.CharacterDataHandler = self.character_data
        parser.buffer_text = True
        parser.ParseFile(file)
        self.stack.pop()
        file.close()

    def parse(self, rnn_path, filename):
        self.path = rnn_path
        self.stack = []
        self.do_parse(filename)

    def parse_reg(self, attrs, bit_size):
        self.current_bitsize = bit_size
        if "type" in attrs and attrs["type"] in self.bitsets:
            bitset = self.bitsets[attrs["type"]]
            if bitset.inline:
                self.current_bitset = Bitset(attrs["name"], bitset)
                self.current_bitset.inline = True
            else:
                self.current_bitset = bitset
        else:
            self.current_bitset = Bitset(attrs["name"], None)
            self.current_bitset.inline = True
            if "type" in attrs:
                self.parse_field(None, attrs)

        variant = self.parse_variants(attrs)
        if not variant and self.current_array:
            variant = self.current_array.variant

        self.current_reg = Reg(attrs, self.prefix(variant), self.current_array, bit_size)
        self.current_reg.bitset = self.current_bitset

        if len(self.stack) == 1:
            self.file.append(self.current_reg)

        if variant is not None:
            self.add_all_variants(self.current_reg, attrs, variant)

        usages = None
        if "usage" in attrs:
            usages = attrs["usage"].split(',')
        elif self.current_array:
            usages = self.current_array.usages

        self.add_all_usages(self.current_reg, usages)

    def start_element(self, name, attrs):
        self.cdata = ""
        if name == "import":
            filename = attrs["file"]
            self.do_parse(os.path.join(self.path, filename))
        elif name == "domain":
            self.current_domain = attrs["name"]
            if "prefix" in attrs:
                self.current_prefix = self.parse_variants(attrs)
                self.current_prefix_type = attrs["prefix"]
            else:
                self.current_prefix = None
                self.current_prefix_type = None
            if "varset" in attrs:
                self.current_varset = self.enums[attrs["varset"]]
        elif name == "stripe":
            self.current_stripe = self.parse_variants(attrs)
        elif name == "enum":
            self.current_enum_value = 0
            self.current_enum = Enum(attrs["name"])
            self.enums[attrs["name"]] = self.current_enum
            if len(self.stack) == 1:
                self.file.append(self.current_enum)
        elif name == "value":
            if "value" in attrs:
                value = int(attrs["value"], 0)
            else:
                value = self.current_enum_value
            self.current_enum.values.append((attrs["name"], value))
        elif name == "reg32":
            self.parse_reg(attrs, 32)
        elif name == "reg64":
            self.parse_reg(attrs, 64)
        elif name == "bitset":
            self.current_bitset = Bitset(attrs["name"], None)
            if "inline" in attrs and attrs["inline"] == "yes":
                self.current_bitset.inline = True
            self.bitsets[self.current_bitset.name] = self.current_bitset
            if len(self.stack) == 1 and not self.current_bitset.inline:
                self.file.append(self.current_bitset)
        elif name == "bitfield" and self.current_bitset:
            self.parse_field(attrs["name"], attrs)

    def end_element(self, name):
        if name == "domain":
            self.current_domain = None
            self.current_prefix = None
            self.current_prefix_type = None
        elif name == "stripe":
            self.current_stripe = None
        elif name == "bitset":
            self.current_bitset = None
        elif name == "reg32":
            self.current_reg = None
        elif name == "enum":
            self.current_enum = None

    def character_data(self, data):
        self.cdata += data

    def parse_variants(self, attrs):
        if not "variants" in attrs:
                return None
        variant = attrs["variants"].split(",")[0]
        if "-" in variant:
            variant = variant[:variant.index("-")]

        varset = self.parse_varset(attrs)

        assert varset.has_name(variant)

        return variant


_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2
def _IOC(d, t, n, s): return (d << 30) | (s << 16) | (ord(t) << 8) | n
def _IOWR(t, n, s): return _IOC(_IOC_READ | _IOC_WRITE, t, n, ctypes.sizeof(s))

DRM_IOC_BASE, DRM_COMMAND_BASE = 'd', 0x40

class Colors:
    R, G, Y, B, M, C, W, BOLD, RESET = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m', '\033[1m', '\033[0m'

    @staticmethod
    def highlight(text):
        text = re.sub(r'(\[0x[0-9a-fA-F]+\])', f'{Colors.C}\\1{Colors.RESET}', text)
        text = re.sub(r'(0x[0-9a-fA-F]+)', f'{Colors.C}\\1{Colors.RESET}', text)
        text = re.sub(r'(REG_[A-Z_]+)', f'{Colors.G}\\1{Colors.RESET}', text)
        text = re.sub(r'\b(EMIT|ENABLED|DISABLED|Unknown)\b', f'{Colors.Y}\\1{Colors.RESET}', text)
        text = re.sub(r'\b(PC|CNA|CORE|DPU|DPU_RDMA|PPU|PPU_RDMA)\b', f'{Colors.Y}\\1{Colors.RESET}', text)
        text = re.sub(r'(lsb\s+[0-9a-fA-F]+)', f'{Colors.W}\\1{Colors.RESET}', text)
        return text

TASK_STRUCT = struct.Struct("<8IQ")
TASK_FIELD_NAMES = (
    "flags",
    "op_idx",
    "enable_mask",
    "int_mask",
    "int_clear",
    "int_status",
    "regcfg_amount",
    "regcfg_offset",
    "regcmd_addr",
)
_TASK_ZERO_CHUNK = b"\x00" * TASK_STRUCT.size

_TASK_REGCMDS = []
_DMA_LOG_CACHE = None
_DMA_LOG_COUNTER = 0

def _parse_dma_log_text(text, cache):
    global _DMA_LOG_COUNTER
    if not text:
        return
    pattern = re.compile(r"dma addr:\s*(0x[0-9a-fA-F]+).*gem name:\s*(\d+)", re.IGNORECASE)
    ts_pattern = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})[.:](\d+)\]")
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        flink = int(m.group(2))
        ts_match = ts_pattern.search(line)
        if ts_match:
            h, mi, s, frac = ts_match.groups()
            try:
                seconds = int(h) * 3600 + int(mi) * 60 + int(s) + float(f"0.{frac}")
            except Exception:
                seconds = 0.0
        else:
            seconds = 0.0
        _DMA_LOG_COUNTER += 1
        cache[flink].append((seconds, _DMA_LOG_COUNTER, addr))

def _populate_dma_cache(force=False):
    global _DMA_LOG_CACHE
    if _DMA_LOG_CACHE is None or force:
        cache = collections.defaultdict(list)
        # Try kernel ring buffer
        try:
            dm = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=1.0)
            if dm.returncode == 0:
                _parse_dma_log_text(dm.stdout, cache)
        except Exception:
            pass

        # Try local log files that may capture RKNN output
        search_dirs = []
        cwd = os.getcwd()
        for _ in range(4):
            if cwd and cwd not in search_dirs:
                search_dirs.append(cwd)
            parent = os.path.dirname(cwd)
            if parent == cwd or not parent:
                break
            cwd = parent

        extra_dirs = [
            os.path.join(os.path.expanduser("~"), "npu", "ops_rknn"),
            os.path.join(os.path.expanduser("~"), "npu"),
        ]
        for directory in extra_dirs:
            if directory and directory not in search_dirs:
                search_dirs.append(directory)

        for directory in search_dirs:
            for name in ("run_output.txt", "output.txt"):
                path = os.path.join(directory, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        _parse_dma_log_text(f.read(), cache)
                except FileNotFoundError:
                    continue
                except Exception:
                    pass

        _DMA_LOG_CACHE = cache
    return _DMA_LOG_CACHE

def lookup_dma_addr_from_logs(flink):
    cache = _populate_dma_cache(force=True)
    if flink not in cache or not cache[flink]:
        return None
    seconds, counter, addr = max(cache[flink], key=lambda x: (x[0], x[1]))
    return addr

def decode_tasks_from_buffer(buf, gem_size, flink=None):
    tasks = []
    regcmd_addrs = []
    for offset in range(0, gem_size, TASK_STRUCT.size):
        chunk = buf[offset: offset + TASK_STRUCT.size]
        if len(chunk) < TASK_STRUCT.size:
            break
        if chunk == _TASK_ZERO_CHUNK:
            continue
        values = TASK_STRUCT.unpack(chunk)
        flags, op_idx, enable_mask, int_mask, int_clear, int_status, regcfg_amount, regcfg_offset, regcmd_addr = values
        if regcfg_amount == 0 or regcmd_addr == 0:
            continue
        if regcfg_amount > 0x10000:
            continue
        if regcmd_addr & 0xf:
            continue
        tasks.append((offset, values))
        if regcmd_addr:
            regcmd_addrs.append(regcmd_addr)
            _TASK_REGCMDS.append({
                "flink": flink,
                "offset": offset,
                "regcfg_amount": regcfg_amount,
                "regcmd_addr": regcmd_addr,
                "used": False,
            })
    return tasks

def _format_task_value(name, value):
    if name == "regcmd_addr":
        return f"0x{value:016x}"
    if name == "regcfg_amount":
        return f"{value} entries (0x{value:08x})"
    if name == "op_idx":
        return f"{value} (0x{value:08x})"
    return f"0x{value:08x}"

def emit_task_report(tasks, flink):
    if not tasks:
        return
    print(f"\n{'='*50}\nDecoded rknpu_task entries for GEM {flink}\n{'='*50}")
    output_path = f"dump/gem{flink}_tasks.txt"
    with open(output_path, "w") as tf:
        for idx, (offset, values) in enumerate(tasks):
            header = f"Task {idx} @ offset 0x{offset:04x}"
            print(Colors.highlight(header))
            tf.write(f"{header}\n")
            for name, value in zip(TASK_FIELD_NAMES, values):
                formatted = _format_task_value(name, value)
                line = f"  {name:<13}: {formatted}"
                print(Colors.highlight(line))
                tf.write(f"{line}\n")
            print()
            tf.write("\n")
    print(Colors.highlight(f"Decoded {len(tasks)} task entries to {output_path}"))

class drm_version(ctypes.Structure):
    _fields_ = [("version_major", ctypes.c_int), ("version_minor", ctypes.c_int), ("version_patchlevel", ctypes.c_int),
                ("name_len", ctypes.c_size_t), ("name", ctypes.POINTER(ctypes.c_char)), ("date_len", ctypes.c_size_t),
                ("date", ctypes.POINTER(ctypes.c_char)), ("desc_len", ctypes.c_size_t), ("desc", ctypes.POINTER(ctypes.c_char))]

class drm_unique(ctypes.Structure):
    _fields_ = [("unique_len", ctypes.c_size_t), ("unique", ctypes.POINTER(ctypes.c_char))]

class drm_gem_open(ctypes.Structure):
    _fields_ = [("name", ctypes.c_uint32), ("handle", ctypes.c_uint32), ("size", ctypes.c_uint64)]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("offset", ctypes.c_uint64)]

DRM_IOCTL_VERSION = _IOWR('d', 0x00, drm_version)
DRM_IOCTL_GET_UNIQUE = _IOWR('d', 0x01, drm_unique)
DRM_IOCTL_GEM_OPEN = _IOWR('d', 0x0b, drm_gem_open)
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', DRM_COMMAND_BASE + 0x03, rknpu_mem_map)

def mask(l, h): return ((0xffffffffffffffff >> (64 - (h + 1 - l))) << l)

def dump_gem(fd, flink):
    print(f"\n{'='*50}\nProcessing GEM Flink {flink}\n{'='*50}")
    try:
        g = drm_gem_open()
        g.name = flink
        fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, g)
        print(Colors.highlight(f"gem flink {flink}: ret=0 handle={g.handle} size={g.size}"))

        m = rknpu_mem_map()
        m.handle = g.handle
        ret = fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, m)
        print(f"DRM_IOCTL_RKNPU_MEM_MAP returned", ret, hex(m.offset))

        instr = mmap.mmap(fd, g.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=m.offset)
        print(f"mmap returned {instr}")

        os.makedirs("dump", exist_ok=True)
        with open(f"dump/gem{flink}-dump", "wb") as f: f.write(instr)

        # # Process blocks, grouping consecutive zero blocks together
        # i = 0
        # while i < g.size:
        #     block = instr[i : i + 16]
        #     here = struct.unpack("<4I", block)
            
        #     # Check if current block is all zeros
        #     if all(x == 0 for x in here):
        #         # Count how many consecutive zero blocks there are starting from this point
        #         zero_start = i
        #         zero_blocks = 0
                
        #         # Count all consecutive zero blocks
        #         j = i
        #         while j < g.size:
        #             block = instr[j : j + 16]
        #             here = struct.unpack("<4I", block)
        #             if all(x == 0 for x in here):
        #                 zero_blocks += 1
        #                 j += 16
        #             else:
        #                 break
                
        #         # Only print the first zero block if there are many consecutive ones
        #         if zero_blocks > 1:
        #             print(Colors.highlight(f"[{zero_start:08x}] = 00000000 00000000 00000000 00000000"))
        #             print(Colors.highlight(f"... {zero_blocks} blocks ({zero_blocks * 16} bytes) from 0x{zero_start:08x} to 0x{zero_start + zero_blocks * 16 - 1:08x} are all zeros"))
        #         else:
        #             print(Colors.highlight(f"[{zero_start:08x}] = 00000000 00000000 00000000 00000000"))
                
        #         i = j  # Move to the next non-zero block
        #     else:
        #         print(Colors.highlight(f"[{i:08x}] = {here[0]:08x} {here[1]:08x} {here[2]:08x} {here[3]:08x}"))
        #         i += 16

        if flink == 1:
            tasks = decode_tasks_from_buffer(instr, g.size, flink=flink)
            emit_task_report(tasks, flink)
            instr.close()
            return
        instr.close()
    except: pass

    print(f"\n{'='*50}\nProcessing GEM Flink {flink} for Register Decode\n{'='*50}")
    try:
        g = drm_gem_open()
        g.name = flink
        fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, g)
        print(Colors.highlight(f"Successfully opened GEM via flink {flink}"))

        m = rknpu_mem_map()
        m.handle = g.handle
        ret = fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, m)
        print(f"memmap returned", ret, hex(m.offset))

        phys_base = m.offset
        instr = mmap.mmap(fd, g.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=phys_base)
        print(f"mmap returned {instr}", phys_base)

        # Initialize parser for XML register definitions
        regs, domains = {}, {}
        if os.path.exists("registers.xml"):
            try:
                p = Parser()
                p.parse("", "registers.xml")
                print(f"DEBUG: Found {len([e for e in p.file if isinstance(e, Reg)])} registers in XML")
                for e in p.file:
                    if isinstance(e, Reg): regs[e.offset] = e
                for e in p.file:
                    if isinstance(e, Enum) and e.name == "target":
                        for name, val in e.values: domains[name] = val
                print(f"DEBUG: Loaded {len(regs)} register definitions")
            except Exception as ex:
                print(f"DEBUG: XML parsing failed: {ex}")
                pass

        commands = []
        for i in range(g.size // 8):
            v = struct.unpack("<Q", instr[i*8:(i+1)*8])[0]
            if v == 0:
                continue
            commands.append((i, v))

        if not commands:
            print(Colors.highlight("No register commands found in this GEM; skipping decode."))
            instr.close()
            return

        base_addr = phys_base
        dma_base = lookup_dma_addr_from_logs(flink)
        if dma_base is not None:
            base_addr = dma_base
        best_entry = None
        best_diff = None
        for entry in _TASK_REGCMDS:
            if entry["regcfg_amount"] <= 0:
                continue
            diff = abs(entry["regcfg_amount"] - len(commands))
            if best_entry is None or diff < best_diff or (diff == best_diff and entry["used"] is False and best_entry.get("used", False)):
                best_entry = entry
                best_diff = diff

        task_base = None
        if best_entry and (best_diff == 0 or best_entry["used"] is False):
            task_base = best_entry["regcmd_addr"]
            if dma_base is not None and dma_base != task_base:
                print(f"DEBUG: overriding dma base 0x{dma_base:x} with task base 0x{task_base:x} for GEM {flink}")
            base_addr = task_base
            best_entry["used"] = True

        print(f"DEBUG: base selection for GEM {flink}: dma_base={dma_base} phys_base=0x{phys_base:x} task_base={task_base} chosen=0x{base_addr:x} commands={len(commands)}")

        with open(f"dump/gem{flink}_regdump.bin", "wb") as df:
            print(Colors.highlight(f"Successfully created dump/gem{flink}_regdump.bin"))
            for i, v in commands:
                val = (v >> 16) & 0xffffffff
                low = v & 0xffff
                tgt = 0
                dst = "noone"
                if (v >> 56) & 1: tgt, dst = 0x100, "PC"
                elif (v >> 57) & 1: tgt, dst = 0x200, "CNA"
                elif (v >> 59) & 1: tgt, dst = 0x800, "CORE"
                elif (v >> 60) & 1: tgt, dst = 0x1000, "DPU"
                elif (v >> 61) & 1: tgt, dst = 0x2000, "DPU_RDMA"
                elif (v >> 62) & 1: tgt, dst = 0x4000, "PPU"
                elif (v >> 63) & 1: tgt, dst = 0x8000, "PPU_RDMA"

                addr = base_addr + i * 8
                if low in regs:
                    reg = regs[low]
                    emit_str = f"EMIT(REG_{regs[low].full_name.upper()}, "
                    first = True
                    if val == 0 or len(reg.bitset.fields) == 1:
                        emit_str += f"0x{val:08x}"
                    else:
                        for field in reg.bitset.fields:
                            if field.type == "boolean":
                                if 1 << field.high & val:
                                    if not first: emit_str += " | "
                                    emit_str += f"{reg.full_name.upper()}_{field.name.upper()}"
                                    first = False
                            elif field.type == "uint":
                                field_value = (val & mask(field.low, field.high)) >> field.low
                                if field_value != 0:
                                    if not first: emit_str += " | "
                                    emit_str += f"{reg.full_name.upper()}_{field.name.upper()}({field_value})"
                                    first = False
                    emit_str += ");"
                    reg_info = f"[0x{addr:08x}] lsb {v:016x} - {dst}"
                    spacing = " " * max(1, 50 - len(reg_info))
                    print(Colors.highlight(f"{reg_info}{spacing}{emit_str}"))
                else:
                    reg_info = f"[0x{addr:08x}] lsb {v:016x} - {dst} Unknown"
                    print(Colors.highlight(reg_info))
                    if i < 5:  # Only show first few mismatches
                        print(f"DEBUG: Looking for offset 0x{low:x}, available offsets: {sorted(regs.keys())[:10]}")

                df.write(struct.pack("<hIh", low if low <= 32767 else low - 65536, val, tgt if tgt <= 32767 else tgt - 65536))

        print(Colors.highlight(f"Dumped {len(commands)} register commands to dump/gem{flink}_regdump.bin"))
        instr.close()
    except: pass

def dump_virtual_memory(fd, address, size=4096):
    """Dump specific virtual memory address"""
    print(f"\n{'='*50}\nProcessing Virtual Memory Address 0x{address:x}\n{'='*50}")
    try:
        # Try to map the virtual memory using the device file descriptor
        print(Colors.highlight(f"Attempting to dump memory at virtual address 0x{address:x}"))
        print(Colors.highlight("Note: This attempts to map virtual memory through the DRM device"))
        print(Colors.highlight("For actual physical memory dumps, you would need to know the GEM object handle"))
        
        # Align address to page boundary and adjust size accordingly
        page_size = 4096
        aligned_address = address & ~(page_size - 1)
        offset = address - aligned_address
        aligned_size = ((offset + size + page_size - 1) // page_size) * page_size
        
        print(Colors.highlight(f"Mapping parameters: address=0x{address:x}, aligned=0x{aligned_address:x}, offset=0x{offset:x}, size={size}, aligned_size={aligned_size}"))
        
        # Try to map the memory
        try:
            # Map memory with READ permissions
            mem = mmap.mmap(fd, aligned_size, mmap.MAP_SHARED, mmap.PROT_READ, offset=aligned_address)
            print(Colors.highlight(f"Successfully mapped memory region"))
            
            # Extract the requested data
            data = mem[offset:offset+size]
            mem.close()
            
        except Exception as mmap_error:
            print(Colors.highlight(f"mmap failed: {mmap_error}"))
            # Try with different protection flags
            try:
                mem = mmap.mmap(fd, aligned_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=aligned_address)
                data = mem[offset:offset+size]
                mem.close()
                print(Colors.highlight(f"mmap with PROT_READ|PROT_WRITE succeeded"))
            except Exception as fallback_error:
                print(Colors.highlight(f"All mmap attempts failed: {fallback_error}"))
                print(Colors.highlight("Cannot map virtual memory at this address"))
                print(Colors.highlight("To dump actual memory, you need to:"))
                print(Colors.highlight("1. Find the GEM object associated with this virtual address"))
                print(Colors.highlight("2. Use DRM_IOCTL_GEM_OPEN to get the handle"))
                print(Colors.highlight("3. Use DRM_IOCTL_RKNPU_MEM_MAP to get the offset"))
                print(Colors.highlight("4. Use mmap with the correct offset"))
                return  # Exit without creating mock data
        
        os.makedirs("dump", exist_ok=True)
        filename = f"dump/vmem_0x{address:x}.bin"
        with open(filename, "wb") as f:
            f.write(data)
            
        print(Colors.highlight(f"Dumped virtual memory to {filename}"))
        
        # Display data in hexdump format
        print(Colors.highlight(f"\nHex dump of virtual memory at 0x{address:x}:"))
        for i in range(0, min(len(data), size), 16):
            # Get 16 bytes for this line
            chunk = data[i:i+16]
            # Convert to hex
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            # Convert to ASCII (printable characters or dots)
            ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
            # Print with address
            print(Colors.highlight(f"0x{i:08x}: {hex_part:<48} {ascii_part}"))
            
        # If we have more data than displayed, show summary
        if len(data) > 256:
            print(Colors.highlight(f"... ({len(data) - 256} more bytes not shown)"))
            
    except Exception as e:
        print(Colors.highlight(f"Failed to dump virtual memory at 0x{address:x}: {e}"))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if we're dumping a virtual memory address
    if len(sys.argv) > 1 and sys.argv[1].startswith("0x"):
        # Handle virtual memory address dumping
        try:
            address = int(sys.argv[1], 16)
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
            
            print(Colors.highlight("Virtual Memory Dumping Information:"))
            print(Colors.highlight("=============================="))
            print(Colors.highlight("This tool attempts to dump virtual memory through the DRM device."))
            print(Colors.highlight("For meaningful results, you need to know the correct virtual address."))
            print(Colors.highlight("For GEM object memory dumps, use the GEM flink name instead."))
            print(Colors.highlight(""))
            
            # Try to open DRM device
            try: 
                fd = os.open("/dev/dri/card1", os.O_RDWR)
                dump_virtual_memory(fd, address, size)
                os.close(fd)
            except Exception as e:
                print(Colors.highlight(f"Failed to open DRM device: {e}"))
                # Still try to dump with mock data
                dump_virtual_memory(None, address, size)
        except ValueError:
            print(Colors.highlight("Invalid virtual memory address format. Use 0x<hex_address>"))
            print(Colors.highlight("Example: python3 dump.py 0x100000"))
        sys.exit(0)
    
    # Handle GEM object dumping (original functionality)
    p = argparse.ArgumentParser(description="Dump GEM objects from RKNNPU device")
    p.add_argument('gems', nargs='*', type=int, help="GEM flink names to dump (default: 1 2)")
    p.add_argument('--vmem', metavar='ADDR', help="Dump virtual memory at hex address (e.g., 0x100000)")
    p.add_argument('--size', type=int, default=4096, help="Size in bytes for virtual memory dump (default: 4096)")
    a = p.parse_args()

    # Handle virtual memory dump from arguments
    if a.vmem:
        try:
            address = int(a.vmem, 16)
            # Try to open DRM device
            try: 
                fd = os.open("/dev/dri/card1", os.O_RDWR)
                dump_virtual_memory(fd, address, a.size)
                os.close(fd)
            except Exception as e:
                print(Colors.highlight(f"Failed to open DRM device: {e}"))
                # Still try to dump with mock data
                dump_virtual_memory(None, address, a.size)
        except ValueError:
            print(Colors.highlight("Invalid virtual memory address format. Use 0x<hex_address>"))
        sys.exit(0)

    try: fd = os.open("/dev/dri/card1", os.O_RDWR)
    except: sys.exit(1)

    try:
        name_buf = (ctypes.c_char * 256)()
        date_buf = (ctypes.c_char * 256)()
        desc_buf = (ctypes.c_char * 256)()

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

        fcntl.ioctl(fd, DRM_IOCTL_VERSION, dv)
        name_str = ctypes.string_at(dv.name, dv.name_len).decode('utf-8', errors='ignore').rstrip('\x00')
        date_str = ctypes.string_at(dv.date, dv.date_len).decode('utf-8', errors='ignore').rstrip('\x00')
        desc_str = ctypes.string_at(dv.desc, dv.desc_len).decode('utf-8', errors='ignore').rstrip('\x00')
        print(Colors.highlight(f"drm name is {name_str} - {date_str} - {desc_str}"))

        unique_buf = (ctypes.c_char * 256)()
        du = drm_unique()
        du.unique_len = 256
        du.unique = ctypes.cast(unique_buf, ctypes.POINTER(ctypes.c_char))
        fcntl.ioctl(fd, DRM_IOCTL_GET_UNIQUE, du)
        unique_str = ctypes.string_at(du.unique, du.unique_len).decode('utf-8', errors='ignore').rstrip('\x00')
        print(Colors.highlight(f"du is {unique_str}"))
    except: pass

    print(Colors.highlight("Dumping specified GEM objects..."))
    for g in (a.gems or [1, 2]):
        if g > 0:
            print(Colors.highlight(f"\n=== Processing GEM {g} ==="))
            dump_gem(fd, g)

    os.close(fd)
