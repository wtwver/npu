#!/usr/bin/env python3
"""Generate RKNN models for the conv1d_simple test cases."""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
  from rknn.api import RKNN
except ImportError:
  RKNN = None


def create_conv2d_model(in_channels: int, out_channels: int, kernel_size: int,
                        kernel_values: Optional[np.ndarray], groups: int) -> nn.Module:
  conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False, groups=groups)
  with torch.no_grad():
    if kernel_values is not None:
      weight_tensor = torch.tensor(kernel_values.astype(np.float32)).view_as(conv.weight)
      conv.weight.copy_(weight_tensor)
    else:
      in_per_group = in_channels // groups
      for oc in range(out_channels):
        for ic in range(in_per_group):
          for k in range(kernel_size):
            conv.weight[oc, ic, 0, k] = float(oc + 1)
  conv.eval()
  return conv


def export_onnx(model: nn.Module, input_shape: Sequence[int], onnx_path: Path) -> None:
  dummy_input = torch.randn(*input_shape, dtype=torch.float32)
  torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=13,
    do_constant_folding=False,
    input_names=["input"],
    output_names=["output"],
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


def main() -> None:
  base_dir = Path(__file__).resolve().parent
  models_dir = base_dir / "models"
  models_dir.mkdir(exist_ok=True)
  data_dir = base_dir / "conv1d_simple_data"

  cases = [
    {"name": "conv1d-i-1-1-11-w-6-1-1", "batch": 1, "in_channels": 1, "input_length": 11, "out_channels": 6, "kernel": 1, "groups": 1},
    {"name": "conv1d-i-1-1-11-w-6-1-2", "batch": 1, "in_channels": 1, "input_length": 11, "out_channels": 6, "kernel": 2, "groups": 1},
    {"name": "conv1d-i-1-1-11-w-6-1-5", "batch": 1, "in_channels": 1, "input_length": 11, "out_channels": 6, "kernel": 5, "groups": 1},
    {"name": "conv1d-i-1-3-11-w-6-3-1", "batch": 1, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 1, "groups": 1},
    {"name": "conv1d-i-1-3-11-w-6-3-2", "batch": 1, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 2, "groups": 1},
    {"name": "conv1d-i-1-3-11-w-6-3-5", "batch": 1, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 5, "groups": 1},
    {"name": "conv1d-i-1-3-11-w-6-1-5-g3", "batch": 1, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 5, "groups": 3},
    {"name": "conv1d-i-8-1-11-w-6-1-1", "batch": 8, "in_channels": 1, "input_length": 11, "out_channels": 6, "kernel": 1, "groups": 1},
    {"name": "conv1d-i-8-1-11-w-6-1-2", "batch": 8, "in_channels": 1, "input_length": 11, "out_channels": 6, "kernel": 2, "groups": 1},
    {"name": "conv1d-i-8-1-11-w-6-1-5", "batch": 8, "in_channels": 1, "input_length": 11, "out_channels": 6, "kernel": 5, "groups": 1},
    {"name": "conv1d-i-8-3-11-w-6-3-1", "batch": 8, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 1, "groups": 1},
    {"name": "conv1d-i-8-3-11-w-6-3-2", "batch": 8, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 2, "groups": 1},
    {"name": "conv1d-i-8-3-11-w-6-3-5", "batch": 8, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 5, "groups": 1},
    {"name": "conv1d-i-8-3-11-w-6-1-5-g3", "batch": 8, "in_channels": 3, "input_length": 11, "out_channels": 6, "kernel": 5, "groups": 3},
  ]

  for case in cases:
    onnx_path = models_dir / f"{case['name']}.onnx"
    kernel_values = None
    kernel_path = data_dir / case["name"] / "kernel.bin"
    if kernel_path.exists():
      kernel_values = np.fromfile(kernel_path, dtype=np.float16).astype(np.float32)
      kernel_values = kernel_values.reshape(case["out_channels"], case["in_channels"] // case["groups"], case["kernel"])
    print(f"\nGenerating {case['name']}... ")
    model = create_conv2d_model(case["in_channels"], case["out_channels"], case["kernel"], kernel_values, case["groups"])
    input_shape = (case["batch"], case["in_channels"], 1, case["input_length"])
    export_onnx(model, input_shape, onnx_path)
    if not build_rknn(onnx_path, input_shape):
      raise SystemExit("RKNN compilation failed; ensure rknn-toolkit2 is installed")


if __name__ == "__main__":
  main()
