#!/usr/bin/env python3
import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, checker, helper
import torch
import torch.nn as nn
import torch.nn.functional as F
from rknn.api import RKNN


ACTIVATION_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
  "tanh": torch.tanh,
}


class Mt19937:
  def __init__(self) -> None:
    self.mt = [0] * 624
    self.index = 624


def _mt_seed(rng: Mt19937, seed: int) -> None:
  rng.mt[0] = seed & 0xFFFFFFFF
  for i in range(1, 624):
    prev = rng.mt[i - 1]
    rng.mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
  rng.index = 624


def _mt_extract(rng: Mt19937) -> int:
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


def _mt_uniform(rng: Mt19937, low: float, high: float) -> float:
  a = _mt_extract(rng) >> 5
  b = _mt_extract(rng) >> 6
  random = (a * 67108864.0 + b) / 9007199254740992.0
  return low + (high - low) * random


def _export_rknn(
  onnx_path: Path, input_shapes: list[list[int]], target_platform: str, *, config_kwargs: dict | None = None, init_runtime: bool = False
) -> None:
  rknn_path = onnx_path.with_suffix(".rknn")
  rknn = RKNN()
  rknn.config(target_platform=target_platform, **(config_kwargs or {}))
  ret = rknn.load_onnx(model=str(onnx_path), input_size_list=input_shapes)
  if ret != 0:
    raise RuntimeError(f"Failed to load {onnx_path} (ret={ret})")
  ret = rknn.build(do_quantization=False, dataset=None)
  if ret != 0:
    raise RuntimeError(f"Failed to build {onnx_path} (ret={ret})")
  if init_runtime:
    ret = rknn.init_runtime()
    if ret != 0:
      raise RuntimeError(f"Failed to init runtime for {onnx_path} (ret={ret})")
  ret = rknn.export_rknn(str(rknn_path))
  if ret != 0:
    raise RuntimeError(f"Failed to export {rknn_path} (ret={ret})")
  rknn.release()


def _write_onnx(model: onnx.ModelProto, path: Path) -> None:
  checker.check_model(model)
  onnx.save(model, str(path))


def _make_idiv_onnx(size: int) -> tuple[onnx.ModelProto, list[list[int]]]:
  shape = [1, 1, size, size]
  input_x = helper.make_tensor_value_info("input_x", TensorProto.INT32, shape)
  input_y = helper.make_tensor_value_info("input_y", TensorProto.INT32, shape)
  output = helper.make_tensor_value_info("output", TensorProto.INT32, shape)
  nodes = [helper.make_node("Div", ["input_x", "input_y"], ["output"])]
  graph = helper.make_graph(nodes, f"div_int32_1x{size}", [input_x, input_y], [output])
  model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
  return model, [shape, shape]


def _make_add_int32_onnx(size: int) -> tuple[onnx.ModelProto, list[list[int]]]:
  shape = [1, 1, size, size]
  input_x = helper.make_tensor_value_info("input_x", TensorProto.INT32, shape)
  input_y = helper.make_tensor_value_info("input_y", TensorProto.INT32, shape)
  output = helper.make_tensor_value_info("output", TensorProto.INT32, shape)
  nodes = [helper.make_node("Add", ["input_x", "input_y"], ["output"])]
  graph = helper.make_graph(nodes, f"add_int32_1x{size}", [input_x, input_y], [output])
  model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
  return model, [shape, shape]

def _make_add_int8_onnx(size: int) -> tuple[onnx.ModelProto, list[list[int]]]:
  shape = [1, 1, size, size]
  input_x = helper.make_tensor_value_info("input_x", TensorProto.INT8, shape)
  input_y = helper.make_tensor_value_info("input_y", TensorProto.INT8, shape)
  output = helper.make_tensor_value_info("output", TensorProto.INT32, shape)
  nodes = [
    helper.make_node("Cast", ["input_x"], ["x_i32"], to=TensorProto.INT32),
    helper.make_node("Cast", ["input_y"], ["y_i32"], to=TensorProto.INT32),
    helper.make_node("Add", ["x_i32", "y_i32"], ["output"]),
  ]
  graph = helper.make_graph(nodes, f"add_int8_1x{size}", [input_x, input_y], [output])
  model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
  return model, [shape, shape]


def _make_torch_onnx(op: str, size: int, dtype: torch.dtype) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, size, size)
  input_shapes: list[list[int]] = [list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  base_name = f"{op}_float16"
  if op == "cast":
    base_name = "cast_int32"
  onnx_path = models_dir / f"{base_name}_1x{size}.onnx"

  if op in {"add", "minus", "max", "min", "div"}:
    class Model(torch.nn.Module):
      def forward(self, x, y):
        if op == "add": return x + y
        if op == "minus": return x - y
        if op == "max": return torch.maximum(x, y)
        if op == "min": return torch.minimum(x, y)
        return x / y

    x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
    y = torch.linspace(2, -2, size * size, dtype=dtype).reshape(shape)
    if size * size > 0:
      y = y.clone()
      y.view(-1)[0] = torch.tensor(1.0, dtype=dtype)
    m = Model()
    torch.onnx.export(m, (x, y), str(onnx_path), input_names=["input_x", "input_y"], output_names=["output"], opset_version=17)
    input_shapes = [list(shape), list(shape)]
    return onnx_path, input_shapes

  if op == "recip":
    class Model(torch.nn.Module):
      def forward(self, x, one):
        return one / x

    x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
    one = torch.ones_like(x)
    if size * size > 0:
      x = x.clone()
      x.view(-1)[0] = torch.tensor(1.0, dtype=dtype)
    m = Model()
    torch.onnx.export(m, (x, one), str(onnx_path), input_names=["input_x", "input_one"], output_names=["output"], opset_version=17)
    input_shapes = [list(shape), list(shape)]
    return onnx_path, input_shapes

  if op == "abs":
    class Model(torch.nn.Module):
      def forward(self, x):
        return torch.abs(x)

  elif op == "cast":
    class Model(torch.nn.Module):
      def forward(self, x):
        return x.to(torch.int32)

  elif op == "floor":
    class Model(torch.nn.Module):
      def forward(self, x):
        return torch.floor(x)

  elif op == "ceil":
    class Model(torch.nn.Module):
      def forward(self, x):
        return -torch.floor(-x)

  elif op == "cmplt":
    class Model(torch.nn.Module):
      def forward(self, x, y):
        return (x < y).to(torch.float16)

    x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
    y = torch.linspace(2, -2, size * size, dtype=dtype).reshape(shape)
    m = Model()
    torch.onnx.export(m, (x, y), str(onnx_path), input_names=["input_x", "input_y"], output_names=["output"], opset_version=17)
    input_shapes = [list(shape), list(shape)]
    return onnx_path, input_shapes

  elif op == "cmpne":
    class Model(torch.nn.Module):
      def forward(self, x, y):
        return (x != y).to(torch.float16)

    x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
    y = torch.linspace(2, -2, size * size, dtype=dtype).reshape(shape)
    if size * size > 0:
      y = y.clone()
      y.view(-1)[0] = x.view(-1)[0]
    m = Model()
    torch.onnx.export(m, (x, y), str(onnx_path), input_names=["input_x", "input_y"], output_names=["output"], opset_version=17)
    input_shapes = [list(shape), list(shape)]
    return onnx_path, input_shapes

  else:
    raise ValueError(f"unsupported op: {op}")

  x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
  m = Model()
  torch.onnx.export(m, x, str(onnx_path), input_names=["input_x"], output_names=["output"], opset_version=17)
  return onnx_path, input_shapes


def _make_neg_bool_onnx(size: int) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, size, size)
  input_shapes: list[list[int]] = [list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"neg_bool_1x{size}.onnx"

  flat = [bool(i & 1) for i in range(size * size)]
  x = torch.tensor(flat, dtype=torch.bool).reshape(shape)

  class Model(torch.nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
      return torch.logical_not(inp)

  torch.onnx.export(
    Model(),
    x,
    str(onnx_path),
    input_names=["input_x"],
    output_names=["output"],
    opset_version=17,
  )
  return onnx_path, input_shapes


def _make_relu_onnx(width: int) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, 1, width)
  input_shapes: list[list[int]] = [list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"relu_float16_1x{width}.onnx"

  x = torch.full(shape, 2, dtype=torch.float16)

  class Model(torch.nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
      return torch.relu(inp)

  torch.onnx.export(
    Model(),
    x,
    str(onnx_path),
    input_names=["input_x"],
    output_names=["output"],
    opset_version=15,
    do_constant_folding=True,
  )
  return onnx_path, input_shapes


def _make_sigmoid_onnx(width: int) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, 1, width)
  input_shapes: list[list[int]] = [list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"sigmoid_float16_1x{width}.onnx"

  x = torch.arange(1, width + 1, dtype=torch.float16).reshape(shape)

  class Model(torch.nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
      return torch.sigmoid(inp)

  torch.onnx.export(
    Model(),
    x,
    str(onnx_path),
    input_names=["input_x"],
    output_names=["output"],
    opset_version=15,
    do_constant_folding=True,
  )
  return onnx_path, input_shapes


def _make_silu_onnx(width: int) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, 1, width)
  input_shapes: list[list[int]] = [list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"silu_float16_1x{width}.onnx"

  x = torch.arange(1, width + 1, dtype=torch.float16).reshape(shape)

  class Model(torch.nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
      return F.silu(inp)

  torch.onnx.export(
    Model(),
    x,
    str(onnx_path),
    input_names=["input_x"],
    output_names=["output"],
    opset_version=15,
    do_constant_folding=True,
  )
  return onnx_path, input_shapes


def _make_swiglu_onnx(width: int) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, 1, width)
  input_shapes: list[list[int]] = [list(shape), list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"swiglu_float16_1x{width}.onnx"

  x = torch.linspace(-2, 2, width, dtype=torch.float16).reshape(shape)
  g = torch.linspace(2, -2, width, dtype=torch.float16).reshape(shape)

  class Model(torch.nn.Module):
    def forward(self, inp: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
      return F.silu(inp) * gate

  torch.onnx.export(
    Model(),
    (x, g),
    str(onnx_path),
    input_names=["input_x", "input_g"],
    output_names=["output"],
    opset_version=15,
    do_constant_folding=True,
  )
  return onnx_path, input_shapes


def _make_where_onnx(size: int) -> tuple[Path, list[list[int]]]:
  shape = (1, 1, size, size)
  input_shapes: list[list[int]] = [list(shape), list(shape), list(shape), list(shape)]
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"where_float16_1x{size}.onnx"

  x = torch.linspace(-2, 2, size * size, dtype=torch.float16).reshape(shape)
  y = torch.linspace(2, -2, size * size, dtype=torch.float16).reshape(shape)
  a = torch.linspace(-1, 1, size * size, dtype=torch.float16).reshape(shape)
  b = torch.linspace(1, -1, size * size, dtype=torch.float16).reshape(shape)

  class Model(torch.nn.Module):
    def forward(self, x_val: torch.Tensor, y_val: torch.Tensor, a_val: torch.Tensor, b_val: torch.Tensor) -> torch.Tensor:
      return torch.where(x_val < y_val, a_val, b_val)

  torch.onnx.export(
    Model(),
    (x, y, a, b),
    str(onnx_path),
    input_names=["input_x", "input_y", "input_a", "input_b"],
    output_names=["output"],
    opset_version=17,
  )
  return onnx_path, input_shapes


class PoolModel(torch.nn.Module):
  def __init__(self, mode: str, kernel_size: int | None = None, stride: int = 1, output_size: Tuple[int, int] | None = None):
    super().__init__()
    self.mode = mode
    self.kernel_size = kernel_size
    self.stride = stride
    self.output_size = output_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    use_fp32 = x.dtype == torch.float16 and self.mode in {"avg", "adaptive_avg"}
    inp = x.float() if use_fp32 else x
    if self.mode == "max":
      out = F.max_pool2d(inp, self.kernel_size, stride=self.stride)
    elif self.mode == "avg":
      out = F.avg_pool2d(inp, self.kernel_size, stride=self.stride)
    elif self.mode == "min":
      out = -F.max_pool2d(-inp, self.kernel_size, stride=self.stride)
    elif self.mode == "adaptive_avg":
      out = F.adaptive_avg_pool2d(inp, self.output_size)
    elif self.mode == "adaptive_max":
      out = F.adaptive_max_pool2d(inp, self.output_size)
    else:
      raise ValueError(f"unsupported mode: {self.mode}")
    if use_fp32:
      out = out.to(x.dtype)
    return out


def _export_pool_case(
  name: str,
  mode: str,
  shape: Tuple[int, int, int, int],
  kernel: int | None = None,
  stride: int = 1,
  output_size: Tuple[int, int] | None = None,
  target: str = "rk3588",
  force: bool = False,
  onnx_only: bool = False,
) -> None:
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"{name}.onnx"
  rknn_path = models_dir / f"{name}.rknn"

  if not force and rknn_path.exists():
    print(f"skip: {rknn_path} exists")
    return

  model = PoolModel(mode, kernel_size=kernel, stride=stride, output_size=output_size)
  dummy_input = torch.arange(1, shape[2] * shape[3] + 1, dtype=torch.float16).reshape(shape)

  torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["input_x"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True,
  )
  print(f"wrote: {onnx_path}")

  if not onnx_only:
    _export_rknn(onnx_path, [[shape[0], shape[1], shape[2], shape[3]]], target)
    print(f"wrote: {rknn_path}")


def _generate_pool_models(target: str, force: bool, onnx_only: bool) -> None:
  cases = [
    dict(name="avg_pool2d_float16_1x4", mode="avg", shape=(1, 1, 4, 4), kernel=2, stride=1),
    dict(name="min_pool2d_float16_1x4", mode="min", shape=(1, 1, 4, 4), kernel=2, stride=1),
    dict(name="adaptive_avg_pool2d_float16_1x4_to_2x2", mode="adaptive_avg", shape=(1, 1, 4, 4), output_size=(2, 2)),
    dict(name="global_avg_pool2d_float16_1x4", mode="adaptive_avg", shape=(1, 1, 4, 4), output_size=(1, 1)),
    dict(name="global_max_pool2d_float16_1x4", mode="adaptive_max", shape=(1, 1, 4, 4), output_size=(1, 1)),
  ]
  for cfg in cases:
    _export_pool_case(target=target, force=force, onnx_only=onnx_only, **cfg)


def _create_conv1d_model(
  in_channels: int,
  out_channels: int,
  kernel_size: int,
  kernel_values: Optional[np.ndarray],
  groups: int,
) -> nn.Module:
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


def _export_onnx_simple(model: nn.Module, input_shape: Sequence[int], onnx_path: Path, dtype: torch.dtype = torch.float32) -> None:
  dummy_input = torch.randn(*input_shape, dtype=dtype)
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
  print(f"wrote: {onnx_path}")


def _build_rknn_simple(onnx_path: Path, input_shape: Sequence[int], target: str = "rk3588") -> None:
  rknn = RKNN()
  rknn.config(target_platform=target)
  rknn.load_onnx(model=str(onnx_path), input_size_list=[list(input_shape)])
  rknn.build(do_quantization=False, dataset=None)
  rknn.export_rknn(str(onnx_path.with_suffix(".rknn")))
  rknn.release()
  print(f"wrote: {onnx_path.with_suffix('.rknn')}")


def _generate_conv1d_models(target: str, onnx_only: bool) -> None:
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
    print(f"\nGenerating {case['name']}...")
    model = _create_conv1d_model(case["in_channels"], case["out_channels"], case["kernel"], kernel_values, case["groups"])
    input_shape = (case["batch"], case["in_channels"], 1, case["input_length"])
    _export_onnx_simple(model, input_shape, onnx_path)
    if not onnx_only:
      _build_rknn_simple(onnx_path, input_shape, target)


def _create_conv2d_model(in_channels: int, out_channels: int, kernel_size: Tuple[int, int], groups: int = 1, bias: bool = False) -> nn.Module:
  conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    groups=groups,
    bias=bias,
  )
  return conv


def _generate_conv2d_models(target: str, onnx_only: bool) -> None:
  n = 6
  test_cases = [
    (n, 3, 2, 1, 1),
    (n, 3, 2, 3, 1),
    (n, 3, 2, 3, 1),
    (n, 3, 2, 5, 1),
    (n, 3, 3, 1, 1),
    (n, 3, 3, 3, 1),
    (n, 3, 3, 3, 3),
    (n, 3, 3, 5, 1),
    (n, 3, 1, 1, 1),
  ]

  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(exist_ok=True)

  dummy_input = torch.randn(1, 3, 5, 7, dtype=torch.float16)
  input_elems = dummy_input.numel()

  for out_ch, in_ch, k_h, k_w, groups in test_cases:
    print(f"\nGenerating model: weight=({out_ch},{in_ch},{k_h},{k_w})")
    if groups > 1:
      print(f"  groups={groups} (depthwise)")

    model = _create_conv2d_model(in_ch, out_ch, (k_h, k_w), groups)
    rng = Mt19937()
    _mt_seed(rng, 0)
    for _ in range(input_elems):
      _mt_uniform(rng, -2.0, 2.0)
    in_ch_per_group = in_ch // groups
    total_weights = out_ch * in_ch_per_group * k_h * k_w
    weight_vals = [_mt_uniform(rng, -2.0, 2.0) for _ in range(total_weights)]
    with torch.no_grad():
      weight_tensor = torch.tensor(weight_vals, dtype=torch.float32).view(out_ch, in_ch_per_group, k_h, k_w)
      model.weight.copy_(weight_tensor)
    model = model.half()
    model.eval()

    model_name = f"conv2d_{k_h}x{k_w}"
    if groups > 1:
      model_name += f"_g{groups}"
    onnx_path = models_dir / f"{model_name}.onnx"
    _export_onnx_simple(model, dummy_input.shape, onnx_path, dtype=dummy_input.dtype)
    if onnx_only:
      print(f"generated: {model_name}")
    else:
      try:
        _build_rknn_simple(onnx_path, dummy_input.shape, target)
        print(f"generated: {model_name}")
      except Exception as exc:
        print(f"failed: {model_name}: {exc}")


def _create_conv2d_model_with_pattern(in_channels: int, out_channels: int, kernel_size: Tuple[int, int], groups: int, bias: bool) -> nn.Module:
  conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    groups=groups,
    bias=bias,
  )
  k_h, k_w = kernel_size
  with torch.no_grad():
    in_ch_per_group = in_channels // groups
    for oc in range(out_channels):
      for ic in range(in_ch_per_group):
        for kh in range(k_h):
          for kw in range(k_w):
            val = 1.0 if (oc + kh + kw) % 3 == 0 else (-1.0 if (oc + kh + kw) % 3 == 1 else 0.0)
            conv.weight[oc, ic, kh, kw] = val
  return conv


def _generate_missing_conv2d(target: str, onnx_only: bool) -> None:
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(exist_ok=True)

  def generate_case(name: str, kernel: Tuple[int, int], groups: int) -> None:
    model = _create_conv2d_model_with_pattern(3, 6, kernel, groups=groups, bias=False)
    dummy_input = torch.randn(1, 3, 5, 7)
    onnx_path = models_dir / f"{name}.onnx"
    _export_onnx_simple(model, dummy_input.shape, onnx_path)
    if not onnx_only:
      _build_rknn_simple(onnx_path, dummy_input.shape, target)
    print(f"generated: {name}")

  generate_case("conv2d_3x3_g3", (3, 3), groups=3)
  generate_case("conv2d_3x5", (3, 5), groups=1)


class ConvOnlyMatMulModel(torch.nn.Module):
  def __init__(self, b_matrix: torch.Tensor) -> None:
    super().__init__()
    b_rows, b_cols = b_matrix.shape
    weight = b_matrix.t().contiguous().view(b_cols, 1, 1, b_rows).to(torch.float16)
    self.register_buffer("weight", weight)

  def forward(self, a: torch.Tensor) -> torch.Tensor:
    return F.conv2d(a.to(torch.float16), self.weight)


def _make_b_matrix(shape: Tuple[int, int], rng: Mt19937) -> torch.Tensor:
  rows, cols = shape
  values = []
  for _ in range(rows * cols):
    values.append(_mt_uniform(rng, -2.0, 2.0))
  return torch.tensor(values, dtype=torch.float16).reshape(rows, cols)


def _generate_matmul_models(target: str, onnx_only: bool) -> None:
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
      raise ValueError(f"incompatible shapes: A {case['a_shape']} vs B {case['b_shape']}")

    name = f"matmul_a{a_rows}x{a_cols}_b{b_rows}x{b_cols}"
    onnx_path = models_dir / f"{name}.onnx"
    print(f"\nGenerating {name}...")

    rng = Mt19937()
    _mt_seed(rng, 0)
    for _ in range(a_rows * a_cols):
      _mt_uniform(rng, -2.0, 2.0)
    b_matrix = _make_b_matrix(case["b_shape"], rng)
    input_shape = (a_rows, 1, 1, a_cols)
    model = ConvOnlyMatMulModel(b_matrix)
    _export_onnx_simple(model, input_shape, onnx_path, dtype=torch.float16)
    if not onnx_only:
      _build_rknn_simple(onnx_path, input_shape, target)

def _make_activation_onnx(op: str, width: int, dtype: torch.dtype) -> tuple[Path, list[list[int]]]:
  if op not in ACTIVATION_FNS:
    raise ValueError(f"unsupported activation: {op}")
  shape = (1, 1, 1, width)
  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = models_dir / f"{op}_float16_1x{width}.onnx"
  fn = ACTIVATION_FNS[op]

  x = torch.linspace(-2.0, 2.0, steps=width, dtype=dtype).reshape(shape)
  expected = fn(x)
  print(f"{op}: expected head: {expected.flatten().tolist()[:4]}")

  class Model(torch.nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
      return fn(inp)

  torch.onnx.export(
    Model(),
    x,
    str(onnx_path),
    input_names=["input_x"],
    output_names=["output"],
    opset_version=15,
    do_constant_folding=True,
  )
  return onnx_path, [list(shape)]


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate RKNN models for ops_rknn")
  parser.add_argument(
    "op",
    choices=[
      "abs",
      "cast",
      "ceil",
      "floor",
      "recip",
      "div",
      "add",
      "minus",
      "max",
      "min",
      "cmplt",
      "cmpne",
      "neg_bool",
      "relu",
      "sigmoid",
      "silu",
      "swiglu",
      "where",
      "conv1d_models",
      "conv2d_models",
      "pool_models",
      "matmul_rknn",
      "missing_conv2d",
      "idiv",
      "add_int",
      *ACTIVATION_FNS,
    ],
  )
  parser.add_argument("size", nargs="?", type=int, default=1)
  parser.add_argument("--target", default="rk3588")
  parser.add_argument("--onnx-only", action="store_true")
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--int8-input", action="store_true", help="for add_int: emit an INT8-input ONNX (RKNN export may not work)")
  parser.add_argument("--rknn-log-level", help="for activations: set RKNN_LOG_LEVEL (default 5)")
  args = parser.parse_args()

  if args.size <= 0:
    raise SystemExit("size must be > 0")

  models_dir = Path(__file__).resolve().parent / "models"
  models_dir.mkdir(parents=True, exist_ok=True)

  if args.op == "idiv":
    onnx_path = models_dir / f"div_int32_1x{args.size}.onnx"
    if onnx_path.exists() and not args.force:
      print(f"skip: {onnx_path} exists")
      shape = [1, 1, args.size, args.size]
      input_shapes = [shape, shape]
    else:
      model, input_shapes = _make_idiv_onnx(args.size)
      _write_onnx(model, onnx_path)
      print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      _export_rknn(onnx_path, input_shapes, args.target)
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op == "add_int":
    if args.int8_input:
      onnx_path = models_dir / f"add_int8_1x{args.size}.onnx"
    else:
      onnx_path = models_dir / f"add_int32_1x{args.size}.onnx"
    if onnx_path.exists() and not args.force:
      print(f"skip: {onnx_path} exists")
      shape = [1, 1, args.size, args.size]
      input_shapes = [shape, shape]
    else:
      if args.int8_input:
        model, input_shapes = _make_add_int8_onnx(args.size)
      else:
        model, input_shapes = _make_add_int32_onnx(args.size)
      _write_onnx(model, onnx_path)
      print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      if args.int8_input:
        raise SystemExit("add_int --int8-input: RKNNToolkit2 2.3.2 crashes exporting INT8-input graphs; use --onnx-only or omit --int8-input")
      _export_rknn(onnx_path, input_shapes, args.target)
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op in ACTIVATION_FNS:
    onnx_path = models_dir / f"{args.op}_float16_1x{args.size}.onnx"
    if onnx_path.exists() and not args.force:
      print(f"skip: {onnx_path} exists")
      input_shapes = [[1, 1, 1, args.size]]
    else:
      onnx_path, input_shapes = _make_activation_onnx(args.op, args.size, torch.float16)
      print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      if args.rknn_log_level is not None:
        os.environ["RKNN_LOG_LEVEL"] = str(args.rknn_log_level)
      else:
        os.environ.setdefault("RKNN_LOG_LEVEL", "5")
      _export_rknn(
        onnx_path,
        input_shapes,
        args.target,
        config_kwargs={
          "single_core_mode": True,
          "remove_reshape": True,
          "disable_rules": ["conv_eltwise_activation_fuse"],
        },
        init_runtime=True,
      )
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op == "neg_bool":
    onnx_path = models_dir / f"neg_bool_1x{args.size}.onnx"
    if onnx_path.exists() and not args.force:
      print(f"skip: {onnx_path} exists")
      input_shapes = [[1, 1, args.size, args.size]]
    else:
      onnx_path, input_shapes = _make_neg_bool_onnx(args.size)
      print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      _export_rknn(onnx_path, input_shapes, args.target)
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op in {"relu", "sigmoid", "silu"}:
    if args.op == "relu":
      onnx_path = models_dir / f"relu_float16_1x{args.size}.onnx"
      if onnx_path.exists() and not args.force:
        print(f"skip: {onnx_path} exists")
        input_shapes = [[1, 1, 1, args.size]]
      else:
        onnx_path, input_shapes = _make_relu_onnx(args.size)
        print(f"wrote: {onnx_path}")
    elif args.op == "sigmoid":
      onnx_path = models_dir / f"sigmoid_float16_1x{args.size}.onnx"
      if onnx_path.exists() and not args.force:
        print(f"skip: {onnx_path} exists")
        input_shapes = [[1, 1, 1, args.size]]
      else:
        onnx_path, input_shapes = _make_sigmoid_onnx(args.size)
        print(f"wrote: {onnx_path}")
    else:
      onnx_path = models_dir / f"silu_float16_1x{args.size}.onnx"
      if onnx_path.exists() and not args.force:
        print(f"skip: {onnx_path} exists")
        input_shapes = [[1, 1, 1, args.size]]
      else:
        onnx_path, input_shapes = _make_silu_onnx(args.size)
        print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      if args.rknn_log_level is not None:
        os.environ["RKNN_LOG_LEVEL"] = str(args.rknn_log_level)
      else:
        os.environ.setdefault("RKNN_LOG_LEVEL", "5")
      _export_rknn(
        onnx_path,
        input_shapes,
        args.target,
        config_kwargs={
          "single_core_mode": True,
          "remove_reshape": True,
          "disable_rules": ["conv_eltwise_activation_fuse"],
        },
        init_runtime=True,
      )
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op == "swiglu":
    onnx_path = models_dir / f"swiglu_float16_1x{args.size}.onnx"
    if onnx_path.exists() and not args.force:
      print(f"skip: {onnx_path} exists")
      input_shapes = [[1, 1, 1, args.size]] * 2
    else:
      onnx_path, input_shapes = _make_swiglu_onnx(args.size)
      print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      if args.rknn_log_level is not None:
        os.environ["RKNN_LOG_LEVEL"] = str(args.rknn_log_level)
      else:
        os.environ.setdefault("RKNN_LOG_LEVEL", "5")
      _export_rknn(
        onnx_path,
        input_shapes,
        args.target,
        config_kwargs={
          "single_core_mode": True,
          "remove_reshape": True,
          "disable_rules": ["conv_eltwise_activation_fuse"],
        },
        init_runtime=True,
      )
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op == "where":
    if args.size < 2:
      raise SystemExit("where: RKNN Toolkit2 crashes for 1x1; use size >= 2")
    onnx_path = models_dir / f"where_float16_1x{args.size}.onnx"
    if onnx_path.exists() and not args.force:
      print(f"skip: {onnx_path} exists")
      input_shapes = [[1, 1, args.size, args.size]] * 4
    else:
      onnx_path, input_shapes = _make_where_onnx(args.size)
      print(f"wrote: {onnx_path}")
    if not args.onnx_only:
      _export_rknn(onnx_path, input_shapes, args.target)
      print(f"wrote: {onnx_path.with_suffix('.rknn')}")
    return 0

  if args.op == "conv1d_models":
    _generate_conv1d_models(args.target, args.onnx_only)
    return 0

  if args.op == "conv2d_models":
    _generate_conv2d_models(args.target, args.onnx_only)
    return 0

  if args.op == "pool_models":
    _generate_pool_models(args.target, args.force, args.onnx_only)
    return 0

  if args.op == "matmul_rknn":
    _generate_matmul_models(args.target, args.onnx_only)
    return 0

  if args.op == "missing_conv2d":
    _generate_missing_conv2d(args.target, args.onnx_only)
    return 0

  base_name = f"{args.op}_float16"
  if args.op == "cast":
    base_name = "cast_int32"
  onnx_path = models_dir / f"{base_name}_1x{args.size}.onnx"
  if onnx_path.exists() and not args.force:
    print(f"skip: {onnx_path} exists")
    input_shapes = [[1, 1, args.size, args.size]]
    if args.op in {"recip", "div", "add", "minus", "max", "min", "cmplt", "cmpne"}:
      input_shapes = [input_shapes[0], input_shapes[0]]
  else:
    onnx_path, input_shapes = _make_torch_onnx(args.op, args.size, torch.float16)
    print(f"wrote: {onnx_path}")

  if not args.onnx_only:
    if args.op == "cast":
      print(
        "note: RKNNToolkit2 on rk3588 typically runs float->int Cast on CPU (NPU dataconvert unsupported); this is a toolkit limitation.",
        file=sys.stderr,
      )
    _export_rknn(onnx_path, input_shapes, args.target)
    print(f"wrote: {onnx_path.with_suffix('.rknn')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
