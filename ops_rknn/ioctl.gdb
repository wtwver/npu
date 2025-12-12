set pagination off
set breakpoint pending on

break rknn_run
commands 
    shell python3 patch_task_buffer.py --mode=none --reg-flink 2 --write-offset 0x5178 --write-value 0x1001000003804070 --write-length 8

    printf "break rknn_run============\n"
    shell python3 dump.py 2 | grep -v "0x00000000"
    printf "end break rknn_run============\n"
    continue
end

break rknn_destroy
commands 
    printf "break rknn_destroy============\n"
    shell python3 dump.py 3
    shell python3 dump.py 4
    shell python3 dump.py 5
    printf "end break rknn_destroy============\n"
    continue
end

python
import gdb
import os
import re
from pathlib import Path

SUBMIT_COUNT = 0
KEEP_TASKS = 2  # tweak here to adjust how many tasks are kept

IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_DIRBITS = 2
IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
IOC_DIRSHIFT = IOC_SIZESHIFT + IOC_SIZEBITS

REG_DPU_EW_CFG = 0x4070
EW_LUT_BYPASS_MASK = 0x80  # DPU_EW_CFG_EW_LUT_BYPASS(1)
EW_REF_DUMP = Path("/tmp/sigmoid2")
EW_SCAN_BYTES_DEFAULT = 0x1400
EW_REF_FALLBACK_WORD = 0x1001000003004070  # word seen in /tmp/sigmoid2 before LUT patch
REFERENCE_EW_CFG_OFFSET = None

def _derive_last_ew_cfg_offset_from_dump(path: Path):
  if not path.exists():
    return None
  try:
    lines = path.read_text().splitlines()
  except OSError:
    return None
  addr_re = re.compile(r"^\[0x([0-9a-fA-F]+)\]")
  base_addr = None
  last_ew_addr = None
  for line in lines:
    m = addr_re.match(line.strip())
    if not m:
      continue
    addr = int(m.group(1), 16)
    if base_addr is None:
      base_addr = addr
    if "REG_DPU_EW_CFG" in line:
      last_ew_addr = addr
  if base_addr is None or last_ew_addr is None:
    return None
  return last_ew_addr - base_addr

REFERENCE_EW_CFG_OFFSET = _derive_last_ew_cfg_offset_from_dump(EW_REF_DUMP)


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


def _decode_submit(addr):
  inf = gdb.selected_inferior()
  try:
    data = inf.read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
  except gdb.MemoryError:
    return None
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

  return {
    "flags": flags,
    "timeout": timeout,
    "task_start": task_start,
    "task_number": task_number,
    "task_counter": task_counter,
    "priority": priority,
    "task_obj_addr": task_obj_addr,
    "regcfg_obj_addr": regcfg_obj_addr,
    "task_base_addr": task_base_addr,
    "user_data": user_data,
    "core_mask": core_mask,
    "fence_fd": fence_fd,
    "subcores": subcores,
  }


def _print_submit(addr, submit_data=None):
  if submit_data is None:
    submit_data = _decode_submit(addr)
  if submit_data is None:
    print("  unable to read struct rknpu_submit from memory")
    return

  print("  struct rknpu_submit {")
  print(f"    flags=0x{submit_data['flags']:08x}")
  print(f"    timeout={submit_data['timeout']}")
  print(f"    task_start={submit_data['task_start']}")
  print(f"    task_number={submit_data['task_number']}")
  print(f"    task_counter={submit_data['task_counter']}")
  print(f"    priority={submit_data['priority']}")
  print(f"    task_obj_addr={_format_hex(submit_data['task_obj_addr'])}")
  print(f"    regcfg_obj_addr={_format_hex(submit_data['regcfg_obj_addr'])}")
  print(f"    task_base_addr={_format_hex(submit_data['task_base_addr'])}")
  print(f"    user_data={_format_hex(submit_data['user_data'])}")
  print(f"    core_mask=0x{submit_data['core_mask']:08x}")
  print(f"    fence_fd={submit_data['fence_fd']}")
  for idx, (start, number) in enumerate(submit_data["subcores"]):
    print(f"    subcore_task[{idx}]={{task_start={start}, task_number={number}}}")
  print("  }")
  if submit_data["task_number"]:
    _print_tasks(submit_data["task_obj_addr"], submit_data["task_number"])


def _patch_submit(addr):
  """Force the submit to only send a limited number of tasks (KEEP_TASKS)."""
  inf = gdb.selected_inferior()
  try:
    inf.write_memory(addr + 12, (KEEP_TASKS).to_bytes(4, "little"))  # task_number
    for idx in range(5):
      base = addr + 64 + idx * 8
      start_bytes = (1).to_bytes(4, "little")
      number_bytes = (KEEP_TASKS if idx == 0 else 0).to_bytes(4, "little")
      inf.write_memory(base, start_bytes)
      inf.write_memory(base + 4, number_bytes)
    print(f"  patched rknpu_submit: task_number={KEEP_TASKS}; subcore_task[0]={{start=1, number={KEEP_TASKS}}}")
  except gdb.MemoryError:
    print("  unable to patch rknpu_submit (memory error)")


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


def _decode_tasks(addr, count):
  tasks = []
  if addr == 0:
    print("  task object address is NULL")
    return tasks
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
    tasks.append(
      {
        "flags": flags,
        "op_idx": op_idx,
        "enable_mask": enable_mask,
        "int_mask": int_mask,
        "int_clear": int_clear,
        "int_status": int_status,
        "regcfg_amount": regcfg_amount,
        "regcfg_offset": regcfg_offset,
        "regcmd_addr": regcmd_addr,
      }
    )
  return tasks


def _print_tasks(addr, count):
  if addr == 0:
    print("  task object address is NULL")
    return
  tasks = _decode_tasks(addr, count)
  for idx, task in enumerate(tasks):
    print(f"  task[{idx}]: flags=0x{task['flags']:08x}, op_idx={task['op_idx']}, enable_mask=0x{task['enable_mask']:08x}")
    print(f"    int_mask=0x{task['int_mask']:08x}, int_clear=0x{task['int_clear']:08x}, int_status=0x{task['int_status']:08x}")
    print(f"    regcfg_amount={task['regcfg_amount']}, regcfg_offset={task['regcfg_offset']}")
    print(f"    regcmd_addr={_format_hex(task['regcmd_addr'])}")


def _patch_last_ew_cfg_in_buffer(base_addr, label, scan_len=None):
  inf = gdb.selected_inferior()
  if scan_len is None:
    scan_len = max(EW_SCAN_BYTES_DEFAULT, (REFERENCE_EW_CFG_OFFSET or 0) + 0x100)
  try:
    data = inf.read_memory(base_addr, scan_len).tobytes()
  except gdb.MemoryError:
    print(f"  unable to read regcmd buffer at {_format_hex(base_addr)} for {label}")
    return

  candidates = []
  for idx in range(0, len(data) - 7, 8):
    word = int.from_bytes(data[idx:idx + 8], "little")
    if (word & 0xffff) == REG_DPU_EW_CFG:
      candidates.append((idx, word))

  target_offset = None
  word = None
  source = None
  if candidates:
    target_offset, word = candidates[-1]
    source = "scan"
  elif REFERENCE_EW_CFG_OFFSET is not None and REFERENCE_EW_CFG_OFFSET + 8 <= len(data):
    target_offset = REFERENCE_EW_CFG_OFFSET
    word = int.from_bytes(data[target_offset:target_offset + 8], "little")
    source = f"reference {EW_REF_DUMP}"
  else:
    print(f"  no REG_DPU_EW_CFG found for {label}; skipping LUT bypass patch")
    return

  reg = word & 0xffff
  value = (word >> 16) & 0xffffffff
  target = word >> 48
  if reg != REG_DPU_EW_CFG:
    print(f"  reference offset for {label} did not land on REG_DPU_EW_CFG; skipping")
    return
  if value & EW_LUT_BYPASS_MASK:
    print(f"  {label} last REG_DPU_EW_CFG already has LUT bypass (source={source})")
    return
  new_value = value | EW_LUT_BYPASS_MASK
  new_word = (target << 48) | (new_value << 16) | reg
  try:
    inf.write_memory(base_addr + target_offset, new_word.to_bytes(8, "little"))
  except gdb.MemoryError:
    print(f"  failed to patch REG_DPU_EW_CFG for {label} at {_format_hex(base_addr + target_offset)}")
    return
  print(
    f"  patched {label} REG_DPU_EW_CFG at {_format_hex(base_addr + target_offset)}:"
    f" value 0x{value:08x} -> 0x{new_value:08x} (source={source})"
  )


def _patch_last_ew_cfg_for_tasks(submit_data):
  if not submit_data or submit_data.get("task_number", 0) <= 0:
    return
  tasks = _decode_tasks(submit_data["task_obj_addr"], submit_data["task_number"])
  if not tasks:
    # Fallback: patch via reg GEM flink=2 using the reference offset/word
    if REFERENCE_EW_CFG_OFFSET is not None:
      new_word = EW_REF_FALLBACK_WORD | ((EW_LUT_BYPASS_MASK & 0xffffffff) << 16)
      cmd = (
        f"python3 patch_task_buffer.py --mode=none --reg-flink 2 "
        f"--write-offset {REFERENCE_EW_CFG_OFFSET:#x} "
        f"--write-value {new_word:#x} --write-length 8"
      )
      print(f"  task decode failed; fallback patch via reg GEM flink=2 at offset {REFERENCE_EW_CFG_OFFSET:#x}")
      os.system(cmd)
    return
  for idx, task in enumerate(tasks):
    regcmd_addr = task.get("regcmd_addr")
    if not regcmd_addr:
      print(f"  task[{idx}] has null regcmd_addr; skipping LUT bypass patch")
      continue
    _patch_last_ew_cfg_in_buffer(regcmd_addr, f"task[{idx}]")


class IoctlDecoder:
  calls = 0
  submit_count = 0

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
        cls.submit_count += 1
        if cls.submit_count >= 2:
          _patch_submit(regs["arg"])
        else:
          print("  submit #1 detected; skipping patch (will patch next submit if any)")
        submit_data = _decode_submit(regs["arg"])
        _print_submit(regs["arg"], submit_data=submit_data)
        _patch_last_ew_cfg_for_tasks(submit_data)
        os.system("python3 dump.py 3")
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

