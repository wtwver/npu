set pagination off
set breakpoint pending on

python
import gdb

IOCTL_SUBMIT = 0xc0686441
TASK_NUMBER_OFFSET = 12
SUBCORE_OFFSET = 64
SUBCORE_COUNT = 3
SUBMIT_STRUCT_SIZE = 104
TASK_STRUCT_SIZE = 40
SUBCORE_ENTRIES = 5

def patch_submit():
  if getattr(patch_submit, "done", False):
    return
  inferior = gdb.selected_inferior()
  addr = int(gdb.parse_and_eval("$x2"))
  if addr == 0 or not inferior.is_valid():
    return

  def write32(offset, value):
    inferior.write_memory(
      addr + offset, int(value).to_bytes(4, "little", signed=False)
    )

  write32(TASK_NUMBER_OFFSET, SUBCORE_COUNT)
  for idx in range(SUBCORE_COUNT):
    base = SUBCORE_OFFSET + idx * 8
    write32(base, 0)  # task_start
    write32(base + 4, 1)  # task_number
  gdb.write("patched submit: task_number=3 with 3 subcore entries\n")
  patch_submit.done = True

# decoding helpers borrowed from npu/ops_rknn/ioctl.gdb
def _format_hex(val, width=0):
  if width:
    return f"0x{val:0{width}x}"
  return f"0x{val:x}"

def _print_tasks(addr, count):
  if addr == 0:
    gdb.write("  task object address is NULL\n")
    return
  inferior = gdb.selected_inferior()
  for idx in range(count):
    base = addr + idx * TASK_STRUCT_SIZE
    try:
      data = inferior.read_memory(base, TASK_STRUCT_SIZE).tobytes()
    except gdb.MemoryError:
      gdb.write(f"  unable to read task[{idx}] at {_format_hex(base)}\n")
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

    gdb.write(
      f"  task[{idx}]: flags=0x{flags:08x}, op_idx={op_idx}, enable_mask=0x{enable_mask:08x}\n"
    )
    gdb.write(
      f"    int_mask=0x{int_mask:08x}, int_clear=0x{int_clear:08x}, int_status=0x{int_status:08x}\n"
    )
    gdb.write(
      f"    regcfg_amount={regcfg_amount}, regcfg_offset={regcfg_offset}\n"
    )
    gdb.write(f"    regcmd_addr={_format_hex(regcmd_addr)}\n")

def print_submit(addr, label):
  gdb.write(f"{label} submit at {_format_hex(addr)}:\n")
  inferior = gdb.selected_inferior()
  try:
    data = inferior.read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
  except gdb.MemoryError:
    gdb.write("  unable to read struct rknpu_submit from memory\n")
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
  for _ in range(SUBCORE_ENTRIES):
    subcores.append((read(4), read(4)))

  gdb.write("  struct rknpu_submit {\n")
  gdb.write(f"    flags=0x{flags:08x}\n")
  gdb.write(f"    timeout={timeout}\n")
  gdb.write(f"    task_start={task_start}\n")
  gdb.write(f"    task_number={task_number}\n")
  gdb.write(f"    task_counter={task_counter}\n")
  gdb.write(f"    priority={priority}\n")
  gdb.write(f"    task_obj_addr={_format_hex(task_obj_addr)}\n")
  gdb.write(f"    regcfg_obj_addr={_format_hex(regcfg_obj_addr)}\n")
  gdb.write(f"    task_base_addr={_format_hex(task_base_addr)}\n")
  gdb.write(f"    user_data={_format_hex(user_data)}\n")
  gdb.write(f"    core_mask=0x{core_mask:08x}\n")
  gdb.write(f"    fence_fd={fence_fd}\n")
  for idx, (start, number) in enumerate(subcores):
    gdb.write(
      f"    subcore_task[{idx}]={{task_start={start}, task_number={number}}}\n"
    )
  gdb.write("  }\n")
  if task_number:
    _print_tasks(task_obj_addr, task_number)

end

break ioctl
commands
  if $x1 != 0xc0686441
    continue
  end
  # print decoded original submit
  python
addr = int(gdb.parse_and_eval("$x2"))
if addr != 0:
  print_submit(addr, "original")
end
  python patch_submit()
  # print decoded modified submit
  python
addr = int(gdb.parse_and_eval("$x2"))
if addr != 0:
  print_submit(addr, "modified")
end
  continue
end

run