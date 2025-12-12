break rknn_run
break rknn_destroy


run

echo 
echo =================================
echo ===before task modify============

set $taskbuf = 0
set $taskbuf_size = 0

define find_task_buffer
  python
import gdb, pathlib, struct
PATTERN = (0x00000000, 0x00000001, 0x0000000d, 0x00000300)
pid = gdb.selected_inferior().pid
maps_path = pathlib.Path(f"/proc/{pid}/maps")
buf_addr = None
buf_size = None
inferior = gdb.selected_inferior()
lines = maps_path.read_text().splitlines()
for line in lines:
  cols = line.split()
  if len(cols) < 5:
    continue
  addr_range, perms, offset_hex, dev, inode = cols[:5]
  pathname = " ".join(cols[5:]) if len(cols) > 5 else ""
  start_s, end_s = addr_range.split('-')
  start = int(start_s, 16)
  end = int(end_s, 16)
  region_size = end - start
  if region_size < 0x1b8:
    continue
  try:
    raw = inferior.read_memory(start, len(PATTERN) * 4)
  except gdb.MemoryError:
    continue
  words = struct.unpack('<' + 'I' * len(PATTERN), raw)
  if words != PATTERN:
    continue
  buf_addr = start
  buf_size = region_size
  gdb.write(f"Found GEM1 mapping candidate: {line}\n")
  break
if buf_addr is None:
  gdb.write("Unable to find GEM1 task buffer mapping via pattern scan\n")
  gdb.execute("set $taskbuf = 0")
  gdb.execute("set $taskbuf_size = 0")
else:
  gdb.execute(f"set $taskbuf = (char*)0x{buf_addr:x}")
  gdb.execute(f"set $taskbuf_size = 0x{buf_size:x}")
  gdb.write(f"Task buffer located at 0x{buf_addr:x} (size 0x{buf_size:x})\n")
  end
end

document find_task_buffer
Locate the user-mapped address of GEM1's task buffer by parsing /proc/<pid>/maps.
end

define zero_gem1_tasks
  if $taskbuf == 0
    printf "task buffer not located, aborting zero\n"
  else
    set $task_payload = 0x1b8
    set $task_first = 0x28
    set $task_zero_len = $task_payload - $task_first
    printf "Zeroing %ld bytes starting at %p\n", $task_zero_len, $taskbuf + $task_first
    call memset($taskbuf + $task_first, 0, $task_zero_len)
  end
end

document zero_gem1_tasks
Zero every task descriptor after task 0 so only the first task remains populated.
end

echo rknn_run============
find_task_buffer
if $taskbuf != 0
  zero_gem1_tasks
  x/32xw $taskbuf
else
  printf "Skipping memory dump, no task buffer\n"
  shell python3 patch_task_buffer.py 1
end

# Preserve GEM2 register programming only through the first PC_OPERATION_ENABLE (offset 0x10b8).
shell python3 patch_task_buffer.py --mode regs --reg-flink 2 --reg-offset 0x10c0

echo ============================\n
echo after task modify\n
echo ============================\n
shell python3 dump.py 1
shell python3 dump.py 2
shell python3 dump.py 3

# shell python3 dump.py 2 | grep "PC_OPERATION_ENABLE_OP_EN"
# shell python3 dump.py 2 | grep "PC_OPERATION_ENABLE_OP_EN" | wc -l
# shell python3 dump.py 2 | grep "PC_BASE_ADDRESS_PC_SOURCE_ADDR" 
# shell python3 dump.py 2 | grep "PC_BASE_ADDRESS_PC_SOURCE_ADDR" | wc -l
c

echo ============================\n
echo before rknn_destroy\n
echo ============================\n
shell python3 dump.py 3
shell python3 dump.py 4
shell python3 dump.py 5
shell python3 dump.py 6
detach
quit
