#!/usr/bin/env python3
"""Generate deterministic conv1d_simple input/weight tensors and reference data."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Conv1dTestCase:
  name: str
  batch: int
  in_channels: int
  input_width: int
  out_channels: int
  kernel_width: int
  groups: int = 1
  stride: int = 1
  dilation: int = 1
  seed: int = 0

  def __post_init__(self) -> None:
    if self.groups <= 0:
      raise ValueError("groups must be positive")
    if self.in_channels % self.groups != 0:
      raise ValueError("in_channels must be divisible by groups")
    if self.input_width <= 0 or self.kernel_width <= 0:
      raise ValueError("input_width and kernel_width must be positive")
    if self.stride <= 0 or self.dilation <= 0:
      raise ValueError("stride and dilation must be positive")

  @property
  def output_width(self) -> int:
    effective_kernel = self.dilation * (self.kernel_width - 1) + 1
    return 1 + (self.input_width - effective_kernel) // self.stride

  @property
  def description(self) -> str:
    return (
      f"conv1d|lhs={self.batch},{self.in_channels},{self.input_width}"
      f"|rhs={self.out_channels},{self.in_channels//self.groups},{self.kernel_width}"
      f"|out={self.batch},{self.out_channels},{self.output_width}"
      f"|hw={self.kernel_width}|stride={self.stride}|dilation={self.dilation}|groups={self.groups}"
    )


CASES = [
  Conv1dTestCase(name="conv1d_simple_bs1", batch=1, in_channels=1, input_width=11, out_channels=6, kernel_width=1),
  Conv1dTestCase(name="conv1d_simple_bs8", batch=8, in_channels=1, input_width=11, out_channels=6, kernel_width=1),
]


def _validate_output_size(width: int, kernel: int, stride: int, dilation: int) -> int:
  effective_kernel = dilation * (kernel - 1) + 1
  if width < effective_kernel:
    raise ValueError("input width smaller than expanded kernel")
  return 1 + (width - effective_kernel) // stride


def _random_fp16_tensor(rng: np.random.RandomState, shape: Sequence[int], low: float, high: float) -> np.ndarray:
  return rng.uniform(low, high, size=shape).astype(np.float16)


def _write_binary(path: Path, data: np.ndarray, dtype: str) -> None:
  path.write_bytes(data.astype(dtype).tobytes())


def _write_expected(path: Path, expected: np.ndarray) -> None:
  path.write_bytes(expected.astype(np.float32).tobytes())


def _compute_expected(input_arr: np.ndarray, kernel_arr: np.ndarray, case: Conv1dTestCase) -> np.ndarray:
  input_tensor = torch.tensor(input_arr.astype(np.float32))
  weight_tensor = torch.tensor(kernel_arr.astype(np.float32))
  output_tensor = F.conv1d(
    input_tensor,
    weight_tensor,
    stride=case.stride,
    dilation=case.dilation,
    groups=case.groups,
  )
  return output_tensor.detach().cpu().numpy()


def generate_case(case: Conv1dTestCase, base_dir: Path) -> None:
  output_width = _validate_output_size(case.input_width, case.kernel_width, case.stride, case.dilation)
  case_dir = base_dir / case.name
  case_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.RandomState(case.seed)
  input_shape = (case.batch, case.in_channels, case.input_width)
  kernel_shape = (case.out_channels, case.in_channels // case.groups, case.kernel_width)
  input_arr = _random_fp16_tensor(rng, input_shape, -2, 2)
  kernel_arr = _random_fp16_tensor(rng, kernel_shape, -2, 2)
  expected_arr = _compute_expected(input_arr, kernel_arr, case)

  input_bin = case_dir / "input.bin"
  kernel_bin = case_dir / "kernel.bin"
  expected_bin = case_dir / "expected.bin"
  metadata_path = case_dir / "metadata.json"
  np.save(case_dir / "input.npy", input_arr.astype(np.float16))
  np.save(case_dir / "kernel.npy", kernel_arr.astype(np.float16))
  np.save(case_dir / "expected.npy", expected_arr.astype(np.float32))

  _write_binary(input_bin, input_arr, "<f2")
  _write_binary(kernel_bin, kernel_arr, "<f2")
  _write_expected(expected_bin, expected_arr)

  metadata = {
    "name": case.name,
    "description": case.description,
    "seed": case.seed,
    "dtype": "fp16",
    "tensor_layout": {
      "input": {"shape": list(input_shape), "format": "NCHW"},
      "kernel": {"shape": list(kernel_shape), "format": "OIHW"},
      "output": {"shape": [case.batch, case.out_channels, output_width], "format": "NCHW"},
    },
    "groups": case.groups,
    "stride": case.stride,
    "dilation": case.dilation,
    "kernel_height": case.kernel_width,
    "expected": {
      "dtype": "fp32",
      "path": expected_bin.name,
      "rtol": 1e-3,
      "atol": 1e-6,
    },
    "paths": {
      "input": input_bin.name,
      "kernel": kernel_bin.name,
    },
  }
  metadata_path.write_text(json.dumps(metadata, indent=2))

  print(f"Generated {case.name} -> inputs {input_shape} kernel {kernel_shape} output width {output_width}")


def main() -> None:
  base_dir = Path(__file__).resolve().parent
  data_dir = base_dir / "conv1d_simple_data"
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", dest="cases", action="append", choices=[case.name for case in CASES],
                      help="limit generation to named cases")
  args = parser.parse_args()

  selected: Iterable[Conv1dTestCase]
  if args.cases:
    names = set(args.cases)
    selected = [case for case in CASES if case.name in names]
  else:
    selected = CASES

  for case in selected:
    generate_case(case, data_dir)


if __name__ == "__main__":
  main()
