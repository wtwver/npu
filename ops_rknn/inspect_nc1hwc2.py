#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def _parse_shape(text):
  try:
    return tuple(int(part) for part in text.split(",") if part)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"invalid shape '{text}'") from exc


def _dtype_from_string(label):
  mapping = {
      "fp16": np.dtype("<f2"),
      "fp32": np.dtype("<f4"),
  }
  try:
    return mapping[label.lower()]
  except KeyError as exc:
    raise argparse.ArgumentTypeError(f"unsupported dtype '{label}'") from exc


def _reshape_expected(array, shape):
  if len(shape) == 3:
    batch, channels, width = shape
    return array.reshape(batch, channels, 1, width)
  if len(shape) == 4:
    batch, channels, height, width = shape
    return array.reshape(batch, channels, height, width)
  raise ValueError("logical shape must have 3 or 4 dimensions")


def main():
  parser = argparse.ArgumentParser(
      description="Inspect RKNN NC1HWC2 dumps by showing which lane/batch slots map to logical slices.")
  parser.add_argument("--dump", required=True, type=Path, help="path to RKNN_DUMP_NATIVE binary (fp16)")
  parser.add_argument(
      "--native-shape",
      required=True,
      type=_parse_shape,
      help="NC1HWC2 dims reported by rknn_query (N,C1,H,W,C2)",
  )
  parser.add_argument(
      "--logical-shape",
      required=True,
      type=_parse_shape,
      help="logical output shape (N,C,W) or (N,C,H,W)",
  )
  parser.add_argument("--stride-w", type=int, default=None, help="width stride reported by the driver (optional)")
  parser.add_argument("--expected", type=Path, help="optional fp32 expected tensor for comparison")
  parser.add_argument("--expected-dtype", default="fp32", type=_dtype_from_string, help="expected tensor dtype")
  parser.add_argument("--abs-tol", type=float, default=1e-6, help="absolute tolerance for comparisons")
  parser.add_argument("--rel-tol", type=float, default=1e-3, help="relative tolerance for comparisons")

  args = parser.parse_args()

  native_shape = args.native_shape
  if len(native_shape) != 5:
    parser.error("native-shape must list 5 integers (N,C1,H,W,C2)")
  native_batch, c1, native_height, native_width, c2 = native_shape
  padded_width = args.stride_w if args.stride_w else native_width
  logical_shape = args.logical_shape
  if len(logical_shape) not in (3, 4):
    parser.error("logical-shape must have 3 or 4 integers")
  logical_batch = logical_shape[0]
  logical_channels = logical_shape[1]
  logical_height = 1 if len(logical_shape) == 3 else logical_shape[2]
  logical_width = logical_shape[-1]

  dump_bytes = args.dump.read_bytes()
  src = np.frombuffer(dump_bytes, dtype=np.dtype("<f2"))
  expected_count = native_batch * c1 * native_height * padded_width * c2
  if src.size != expected_count:
    raise RuntimeError(
        f"dump size {src.size} fp16 elems does not match native dims {native_shape} "
        f"with padded width {padded_width} ({expected_count})")

  total_logical_slices = logical_batch * logical_channels
  dst = np.zeros((logical_batch, logical_channels, logical_height, logical_width), dtype=np.float32)
  lane_mappings = []
  idx = 0
  for g in range(c1):
    for c in range(c2):
      lane_id = g * c2 + c
      lane_entries = []
      for n_idx in range(native_batch):
        logical_index = lane_id * native_batch + n_idx
        if logical_index < total_logical_slices:
          batch_idx = logical_index // logical_channels
          channel_idx = logical_index % logical_channels
        else:
          batch_idx = channel_idx = None
        lane_entries.append((n_idx, logical_index, batch_idx, channel_idx))
        for h in range(native_height):
          for w in range(padded_width):
            val = float(src[idx])
            idx += 1
            if batch_idx is None:
              continue
            if h >= logical_height or w >= logical_width:
              continue
            dst[batch_idx, channel_idx, h, w] = val
      lane_mappings.append((lane_id, g, c, lane_entries))

  print(f"Parsed {len(src)} fp16 values for native shape {native_shape} with stride width {padded_width}")
  print(f"Logical tensor: batch={logical_batch}, channels={logical_channels}, "
        f"height={logical_height}, width={logical_width}")
  print(f"Lane capacity: {c1 * c2} lanes x {native_batch} slices per lane "
        f"= {c1 * c2 * native_batch} total slots")
  print(f"Logical slices used: {total_logical_slices}")

  for lane_id, g, c_slot, entries in lane_mappings:
    lane_used = any(batch_idx is not None for (_, _, batch_idx, _) in entries)
    status = "active" if lane_used else "padding"
    print(f"\nLane {lane_id:2d} (c1={g}, c2={c_slot}) [{status}]")
    for n_idx, logical_index, batch_idx, channel_idx in entries:
      if batch_idx is None:
        print(f"  n={n_idx:2d}: logical_slice={logical_index} -> padding")
      else:
        print(
            f"  n={n_idx:2d}: logical_slice={logical_index:3d} -> batch={batch_idx}, channel={channel_idx}")

  if args.expected:
    expected = np.fromfile(args.expected, dtype=args.expected_dtype)
    expected = _reshape_expected(expected, logical_shape)
    if expected.shape != dst.shape:
      raise RuntimeError(
          f"expected tensor shape {expected.shape} does not match logical tensor {dst.shape}")
    diff = np.abs(expected - dst)
    max_abs = float(diff.max())
    max_index_flat = int(diff.argmax())
    batch_idx = max_index_flat // (logical_channels * logical_height * logical_width)
    rem = max_index_flat % (logical_channels * logical_height * logical_width)
    channel_idx = rem // (logical_height * logical_width)
    rem %= logical_height * logical_width
    height_idx = rem // logical_width
    width_idx = rem % logical_width
    rel = max_abs / max(1e-12, abs(float(expected[batch_idx, channel_idx, height_idx, width_idx])))
    print("\nComparison against expected tensor:")
    print(f"  max_abs_diff={max_abs}")
    print(f"  rel_diff_at_worst={rel}")
    print(f"  worst_index=(batch={batch_idx}, channel={channel_idx}, h={height_idx}, w={width_idx})")
    print(f"  worst_expected={expected[batch_idx, channel_idx, height_idx, width_idx]}")
    print(f"  worst_actual  ={dst[batch_idx, channel_idx, height_idx, width_idx]}")
    print(f"  pass? abs_diff<={args.abs_tol} and rel_diff<={args.rel_tol} -> "
          f"{'yes' if max_abs <= args.abs_tol else 'no'}")


if __name__ == "__main__":
  main()
