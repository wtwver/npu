#!/usr/bin/env python3
import argparse, errno, fcntl, mmap, os
from dump import drm_gem_open, rknpu_mem_map, DRM_IOCTL_GEM_OPEN, DRM_IOCTL_RKNPU_MEM_MAP

TASK_TABLE_BYTES = 0x1b8
TASK0_BYTES = 0x28
DEFAULT_REG_ZERO_OFFSET = 0x0dc0


def safe_flush(mm):
  try:
    mm.flush()
  except OSError as exc:
    if exc.errno != errno.EINVAL:
      raise


def map_gem(fd, flink):
  gem = drm_gem_open()
  gem.name = flink
  fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, gem)

  mapper = rknpu_mem_map()
  mapper.handle = gem.handle
  fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mapper)

  mm = mmap.mmap(fd, gem.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mapper.offset)
  return gem, mm


def zero_after_task0(fd, flink):
  gem, mm = map_gem(fd, flink)
  try:
    end = gem.size
    if end <= TASK0_BYTES:
      return 0
    zero_len = end - TASK0_BYTES
    mm[TASK0_BYTES:end] = b"\x00" * zero_len
    safe_flush(mm)
    return zero_len
  finally:
    mm.close()


def zero_reg_after_offset(fd, flink, zero_from, zero_to=None):
  gem, mm = map_gem(fd, flink)
  try:
    start = max(0, min(zero_from, gem.size))
    if zero_to is None:
      end = gem.size
    else:
      end = max(0, min(zero_to, gem.size))
    if start >= end:
      return 0, b"", False
    zero_len = end - start
    before = mm[start:start + min(32, zero_len)]
    mm[start:end] = b"\x00" * zero_len
    safe_flush(mm)
    after = mm[start:start + min(32, zero_len)]
    cleaned = all(b == 0 for b in after)
    return zero_len, before, cleaned
  finally:
    mm.close()


def parse_args():
  parser = argparse.ArgumentParser(description="Patch GEM buffers to keep only the first task")
  parser.add_argument("flink", type=int, nargs="?", help="Task GEM flink (default: 1)")
  parser.add_argument("--mode", choices=["task", "regs", "both"], default="task",
                      help="Which buffers to patch (default: task)")
  parser.add_argument("--task-flink", type=int, help="Override task GEM flink (default: positional or 1)")
  parser.add_argument("--reg-flink", type=int, default=2, help="GEM flink holding register commands (default: 2)")
  parser.add_argument("--reg-offset", type=lambda v: int(v, 0), default=DEFAULT_REG_ZERO_OFFSET,
                      help="Byte offset after which reg GEM should be zeroed (default: 0xdc0)")
  parser.add_argument("--reg-end", type=lambda v: int(v, 0),
                      help="Byte offset at which reg GEM zeroing should stop (default: end of GEM)")
  return parser.parse_args()


def main():
  args = parse_args()
  do_tasks = args.mode in ("task", "both")
  do_regs = args.mode in ("regs", "both")
  if not do_tasks and not do_regs:
    do_tasks = True

  task_flink = args.task_flink or args.flink or 1

  fd = os.open("/dev/dri/card1", os.O_RDWR)
  try:
    if do_tasks:
      zeroed = zero_after_task0(fd, task_flink)
      print(f"GEM {task_flink}: zeroed 0x{zeroed:x} bytes after task0")
    if do_regs:
      zeroed, sample, verified = zero_reg_after_offset(
        fd, args.reg_flink, args.reg_offset, zero_to=args.reg_end)
      sample_hex = sample.hex() or "(already zero)"
      if args.reg_end is None:
        range_desc = f"from offset 0x{args.reg_offset:x}"
      else:
        range_desc = f"from 0x{args.reg_offset:x} to 0x{args.reg_end:x}"
      print(f"GEM {args.reg_flink}: zeroed 0x{zeroed:x} bytes {range_desc}")
      print(f"  sample before: {sample_hex[:64]}")
      print(f"  tail verified zero: {'yes' if verified else 'no'}")
  finally:
    os.close(fd)


if __name__ == "__main__":
  main()
