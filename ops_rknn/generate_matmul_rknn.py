#!/usr/bin/env python3
"""Generate ONNX and RKNN MatMul models for a few square-ish shapes.

The exported graph uses a single Conv op (no reshapes) by treating each matrix
row as a separate batch item and using a 1xK kernel whose weights come from B.
"""
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

try:
  from rknn.api import RKNN
except ImportError:
  RKNN = None


class Mt19937:
  def __init__(self) -> None:
    self.mt = [0] * 624
    self.index = 624


def mt_seed(rng: Mt19937, seed: int) -> None:
  rng.mt[0] = seed & 0xFFFFFFFF
  for i in range(1, 624):
    prev = rng.mt[i - 1]
    rng.mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
  rng.index = 624


def mt_extract(rng: Mt19937) -> int:
  mag01 = [0, 0x9908b0df]
  if rng.index >= 624:
    for kk in range(624):
      y = (rng.mt[kk] & 0x80000000) | (rng.mt[(kk + 1) % 624] & 0x7fffffff)
      rng.mt[kk] = rng.mt[(kk + 397) % 624] ^ (y >> 1) ^ mag01[y & 1]
    rng.index = 0
  y = rng.mt[rng.index]
  rng.index += 1
  y ^= (y >> 11)
  y ^= (y << 7) & 0x9d2c5680
  y ^= (y << 15) & 0xefc60000
  y ^= (y >> 18)
  return y & 0xFFFFFFFF


def mt_uniform(rng: Mt19937, low: float, high: float) -> float:
  a = mt_extract(rng) >> 5
  b = mt_extract(rng) >> 6
  random = (a * 67108864.0 + b) / 9007199254740992.0
  return low + (high - low) * random


class ConvOnlyMatMulModel(torch.nn.Module):
  def __init__(self, b_matrix: torch.Tensor) -> None:
    super().__init__()
    b_rows, b_cols = b_matrix.shape
    # Shape weights for a 1xK Conv2d: (out_channels, in_channels, kH, kW)
    weight = b_matrix.t().contiguous().view(b_cols, 1, 1, b_rows).to(torch.float16)
    self.register_buffer("weight", weight)

  def forward(self, a: torch.Tensor) -> torch.Tensor:
    # Input shape: (batch = rows of A, 1, 1, cols of A)
    return F.conv2d(a.to(torch.float16), self.weight)


def export_onnx(model: torch.nn.Module, input_shape: Sequence[int], onnx_path: Path) -> None:
  dummy_input = torch.randn(*input_shape, dtype=torch.float16)
  torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=13,
    do_constant_folding=False,
    input_names=["A"],
    output_names=["C"],
  )
  print(f"Saved {onnx_path.name}")


def build_rknn(onnx_path: Path, input_shape: Sequence[int], target_platform: str = "rk3588") -> bool:
  if RKNN is None:
    print("RKNN toolkit not installed, skipping .rknn compilation")
    return False

  rknn = RKNN()
  rknn.config(target_platform=target_platform)
  rknn.load_onnx(model=str(onnx_path), input_size_list=[list(input_shape)])
  rknn.build(do_quantization=False, dataset=None)
  rknn.export_rknn(str(onnx_path.with_suffix(".rknn")))
  print(f"Compiled {onnx_path.with_suffix('.rknn').name}")
  return True


def make_b_matrix(shape: Tuple[int, int], rng: Mt19937) -> torch.Tensor:
  """Create a deterministic float16 weight matrix using mt_uniform."""
  rows, cols = shape
  values = []
  for _ in range(rows * cols):
    values.append(mt_uniform(rng, -2.0, 2.0))
  return torch.tensor(values, dtype=torch.float16).reshape(rows, cols)


def main() -> None:
  base_dir = Path(__file__).resolve().parent
  models_dir = base_dir / "models"
  models_dir.mkdir(exist_ok=True)

  cases = [
    {"a_shape": (8, 8), "b_shape": (8, 8)},
    {"a_shape": (9, 9), "b_shape": (9, 9)},
    {"a_shape": (64, 64), "b_shape": (64, 64)},
    {"a_shape": (256, 256), "b_shape": (256, 256)},
  ]

  for case in cases:
    a_rows, a_cols = case["a_shape"]
    b_rows, b_cols = case["b_shape"]
    if a_cols != b_rows:
      raise ValueError(f"Incompatible shapes: A {case['a_shape']} vs B {case['b_shape']}")

    name = f"matmul_a{a_rows}x{a_cols}_b{b_rows}x{b_cols}"
    onnx_path = models_dir / f"{name}.onnx"
    print(f"\nGenerating {name}...")

    rng = Mt19937()
    mt_seed(rng, 0)
    # Skip draws for input tensor values so B matches what matmul_multi expects
    for _ in range(a_rows * a_cols):
      mt_uniform(rng, -2.0, 2.0)
    b_matrix = make_b_matrix(case["b_shape"], rng)
    input_shape = (a_rows, 1, 1, a_cols)
    model = ConvOnlyMatMulModel(b_matrix)
    export_onnx(model, input_shape, onnx_path)
    if not build_rknn(onnx_path, input_shape):
      raise SystemExit("RKNN compilation failed; ensure rknn-toolkit2 is installed")


if __name__ == "__main__":
  main()
