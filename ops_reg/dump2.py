import sys, os, struct, re, argparse, xml.parsers.expat, collections


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


def decode_value(v, regs):
    if v == 0:
        return None
    val = (v >> 16) & 0xffffffff
    low = v & 0xffff
    tgt = 0
    dst = "noone"
    if (v >> 56) & 1:
        tgt, dst = 0x100, "PC"
    elif (v >> 57) & 1:
        tgt, dst = 0x200, "CNA"
    elif (v >> 59) & 1:
        tgt, dst = 0x800, "CORE"
    elif (v >> 60) & 1:
        tgt, dst = 0x1000, "DPU"
    elif (v >> 61) & 1:
        tgt, dst = 0x2000, "DPU_RDMA"
    elif (v >> 62) & 1:
        tgt, dst = 0x4000, "PPU"
    elif (v >> 63) & 1:
        tgt, dst = 0x8000, "PPU_RDMA"

    line = None
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
        reg_info = f"lsb {v:016x} - {dst}"
        spacing = " " * 1
        line = Colors.highlight(f"{reg_info}{spacing}{emit_str}")
    else:
        reg_info = f"lsb {v:016x} - {dst} Unknown"
        line = Colors.highlight(reg_info)

    return low, val, tgt, line


def load_regs(registers_xml_path):
    regs, domains = {}, {}
    if os.path.exists(registers_xml_path):
        try:
            p = Parser()
            p.parse("", registers_xml_path)
            for e in p.file:
                if isinstance(e, Reg):
                    regs[e.offset] = e
            for e in p.file:
                if isinstance(e, Enum) and e.name == "target":
                    for name, val in e.values:
                        domains[name] = val
        except Exception:
            pass
    return regs


def decode_andreg(input_path, out_dir="dump"):
    regs = load_regs("registers.xml")
    os.makedirs(out_dir, exist_ok=True)
    out_bin = os.path.join(out_dir, "andreg_regdump.bin")
    out_txt = os.path.join(out_dir, "andreg_regdump.txt")

    line_re = re.compile(r"npu_regs_map2\[(\d+)\]:\s*0x([0-9a-fA-F]+)")

    count = 0
    with open(input_path, "r") as f, open(out_bin, "wb") as bf, open(out_txt, "w") as tf:
        for raw in f:
            m = line_re.search(raw)
            if not m:
                continue
            v = int(m.group(2), 16)
            decoded = decode_value(v, regs)
            if decoded is None:
                continue
            low, val, tgt, line = decoded
            bf.write(struct.pack(
                "<hIh",
                low if low <= 32767 else low - 65536,
                val,
                tgt if tgt <= 32767 else tgt - 65536,
            ))
            tf.write(line + "\n")
            print(line)
            count += 1

    print(Colors.highlight(f"Decoded {count} register commands to {out_bin} and {out_txt}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Decode ops_reg/andreg.h register dump")
    p.add_argument("input", nargs="?", default=os.path.join(os.path.dirname(__file__), "andreg.h"), help="Path to andreg.h-like file")
    p.add_argument("--out", dest="out", default="dump", help="Output directory (default: dump)")
    a = p.parse_args()
    decode_andreg(a.input, a.out)


