set pagination off
set breakpoint pending on

python
import gdb

IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_DIRBITS = 2
IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
IOC_DIRSHIFT = IOC_SIZESHIFT + IOC_SIZEBITS

def _mask(bits):
  return (1 << bits) - 1

DIR_NAMES = {
  0: "IOC_NONE",
  1: "IOC_WRITE",
  2: "IOC_READ",
  3: "IOC_READ|IOC_WRITE",
}

DRM_COMMAND_BASE = 0x40
RKNPU_COMMANDS = {
  DRM_COMMAND_BASE + 0: ("DRM_IOCTL_RKNPU_ACTION", "struct rknpu_action"),
  DRM_COMMAND_BASE + 1: ("DRM_IOCTL_RKNPU_SUBMIT", "struct rknpu_submit"),
  DRM_COMMAND_BASE + 2: ("DRM_IOCTL_RKNPU_MEM_CREATE", "struct rknpu_mem_create"),
  DRM_COMMAND_BASE + 3: ("DRM_IOCTL_RKNPU_MEM_MAP", "struct rknpu_mem_map"),
  DRM_COMMAND_BASE + 4: ("DRM_IOCTL_RKNPU_MEM_DESTROY", "struct rknpu_mem_destroy"),
  DRM_COMMAND_BASE + 5: ("DRM_IOCTL_RKNPU_MEM_SYNC", "struct rknpu_mem_sync"),
}

SUBMIT_STRUCT_SIZE = 104
TASK_STRUCT_SIZE = 40
MEM_SYNC_STRUCT_SIZE = 32
MEM_SYNC_DUMP_SIZE = 64

def _format_hex(val, width=0):
  if width:
    return f"0x{val:0{width}x}"
  return f"0x{val:x}"

def _print_submit(addr):
  inf = gdb.selected_inferior()
  try:
    data = inf.read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
  except gdb.MemoryError:
    print("  unable to read struct rknpu_submit from memory")
    return
  offset = 0

  def read(size, signed=False):
    nonlocal offset
    val = int.from_bytes(data[offset:offset + size], "little", signed=signed)
    offset += size
    return val

  flags = read(4)
  timeout = read(4)
  task_start = read(4)
  task_number = read(4)
  task_counter = read(4)
  priority = read(4, signed=True)
  task_obj_addr = read(8)
  regcfg_obj_addr = read(8)
  task_base_addr = read(8)
  user_data = read(8)
  core_mask = read(4)
  fence_fd = read(4, signed=True)
  subcores = []
  for idx in range(5):
    subcores.append((read(4), read(4)))

  print("  struct rknpu_submit {")
  print(f"    flags=0x{flags:08x}")
  print(f"    timeout={timeout}")
  print(f"    task_start={task_start}")
  print(f"    task_number={task_number}")
  print(f"    task_counter={task_counter}")
  print(f"    priority={priority}")
  print(f"    task_obj_addr={_format_hex(task_obj_addr)}")
  print(f"    regcfg_obj_addr={_format_hex(regcfg_obj_addr)}")
  print(f"    task_base_addr={_format_hex(task_base_addr)}")
  print(f"    user_data={_format_hex(user_data)}")
  print(f"    core_mask=0x{core_mask:08x}")
  print(f"    fence_fd={fence_fd}")
  for idx, (start, number) in enumerate(subcores):
    print(f"    subcore_task[{idx}]={{task_start={start}, task_number={number}}}")
  print("  }")
  if task_number:
    _print_tasks(task_obj_addr, task_number)

def _print_mem_sync(addr):
  if addr == 0:
    print("  struct rknpu_mem_sync at NULL")
    return
  inf = gdb.selected_inferior()
  try:
    data = inf.read_memory(addr, MEM_SYNC_STRUCT_SIZE).tobytes()
  except gdb.MemoryError:
    print("  unable to read struct rknpu_mem_sync from memory")
    return
  offset = 0

  def read(size):
    nonlocal offset
    val = int.from_bytes(data[offset:offset + size], "little")
    offset += size
    return val

  flags = read(4)
  reserved = read(4)
  obj_addr = read(8)
  offset_val = read(8)
  size_val = read(8)

  print("  struct rknpu_mem_sync {")
  print(f"    flags=0x{flags:08x}")
  print(f"    reserved=0x{reserved:08x}")
  print(f"    obj_addr={_format_hex(obj_addr)}")
  print(f"    offset={offset_val}")
  print(f"    size={size_val}")
  print("  }")
  _dump_memory(obj_addr, MEM_SYNC_DUMP_SIZE)

def _dump_memory(addr, length):
  if addr == 0:
    print("  obj_addr is NULL, nothing to dump")
    return
  inf = gdb.selected_inferior()
  try:
    data = inf.read_memory(addr, length).tobytes()
  except gdb.MemoryError:
    print(f"  unable to read {length} bytes at {_format_hex(addr)}")
    return
  print(f"  {length} bytes from obj_addr:")
  for i in range(0, len(data), 16):
    chunk = data[i:i+16]
    hex_bytes = " ".join(f"{b:02x}" for b in chunk)
    print(f"    {addr + i:#016x}: {hex_bytes}")

def _print_tasks(addr, count):
  if addr == 0:
    print("  task object address is NULL")
    return
  inf = gdb.selected_inferior()
  for idx in range(count):
    base = addr + idx * TASK_STRUCT_SIZE
    try:
      data = inf.read_memory(base, TASK_STRUCT_SIZE).tobytes()
    except gdb.MemoryError:
      print(f"  unable to read task[{idx}] at 0x{base:x}")
      break
    offset = 0

    def read(size, signed=False):
      nonlocal offset
      val = int.from_bytes(data[offset:offset + size], "little", signed=signed)
      offset += size
      return val

    flags = read(4)
    op_idx = read(4)
    enable_mask = read(4)
    int_mask = read(4)
    int_clear = read(4)
    int_status = read(4)
    regcfg_amount = read(4)
    regcfg_offset = read(4)
    regcmd_addr = read(8)

    print(f"  task[{idx}]: flags=0x{flags:08x}, op_idx={op_idx}, enable_mask=0x{enable_mask:08x}")
    print(f"    int_mask=0x{int_mask:08x}, int_clear=0x{int_clear:08x}, int_status=0x{int_status:08x}")
    print(f"    regcfg_amount={regcfg_amount}, regcfg_offset={regcfg_offset}")
    print(f"    regcmd_addr={_format_hex(regcmd_addr)}")

class IoctlDecoder:
  calls = 0

  @classmethod
  def handle(cls):
    regs = {
      "fd": int(gdb.parse_and_eval("$x0")),
      "cmd": int(gdb.parse_and_eval("$x1")),
      "arg": int(gdb.parse_and_eval("$x2")),
    }
    cls.calls += 1
    cmd = regs["cmd"]
    dir_val = (cmd >> IOC_DIRSHIFT) & _mask(IOC_DIRBITS)
    type_val = (cmd >> IOC_TYPESHIFT) & _mask(IOC_TYPEBITS)
    nr = (cmd >> IOC_NRSHIFT) & _mask(IOC_NRBITS)
    size = (cmd >> IOC_SIZESHIFT) & _mask(IOC_SIZEBITS)
    type_char = repr(chr(type_val)) if 32 <= type_val < 127 else f"{type_val:#x}"
    dir_name = DIR_NAMES.get(dir_val, f"{dir_val:#x}")
    macro = RKNPU_COMMANDS.get(nr)
    print(
      f"ioctl call #{cls.calls}: fd={regs['fd']}, cmd=0x{cmd:08x} "
      f"({dir_name}, type={type_char}, nr=0x{nr:02x}, size={size}), arg=0x{regs['arg']:x}"
    )
    if macro is not None:
      print(
        f"  reconstructed: ioctl(fd, {macro[0]}, "
        f"(const {macro[1]} *)0x{regs['arg']:x});"
      )
      if macro[0] == "DRM_IOCTL_RKNPU_SUBMIT":
        _print_submit(regs["arg"])
      elif macro[0] == "DRM_IOCTL_RKNPU_MEM_SYNC":
        _print_mem_sync(regs["arg"])
    else:
      print(
        f"  reconstructed: ioctl(fd, /* unknown req */ 0x{cmd:08x}, "
        f"(void *)0x{regs['arg']:x});"
      )

IoctlDecoder  # keep class alive for commands block
end

b ioctl
commands
  python IoctlDecoder.handle()
  continue
end

r
q
