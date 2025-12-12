import argparse
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from rknn.api import RKNN


class PoolModel(torch.nn.Module):
    def __init__(self, mode: str, kernel_size: int = None, stride: int = 1, output_size: Tuple[int, int] = None):
        super().__init__()
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size

    def forward(self, x):
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
            raise ValueError(f"Unsupported mode: {self.mode}")
        if use_fp32:
            out = out.to(x.dtype)
        return out


def export_case(name: str, mode: str, shape: Tuple[int, int, int, int], kernel: int = None, stride: int = 1,
                output_size: Tuple[int, int] = None, target: str = "rk3588", force: bool = False):
    os.makedirs("models", exist_ok=True)
    onnx_path = os.path.join("models", f"{name}.onnx")
    rknn_path = os.path.join("models", f"{name}.rknn")

    if not force and os.path.exists(rknn_path):
        print(f"[skip] {rknn_path} already exists")
        return

    model = PoolModel(mode, kernel_size=kernel, stride=stride, output_size=output_size)
    dummy_input = torch.arange(1, shape[2] * shape[3] + 1, dtype=torch.float16).reshape(shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_x"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"[onnx] wrote {onnx_path}")

    rknn = RKNN()
    rknn.config(target_platform=target)
    ret = rknn.load_onnx(model=onnx_path, input_size_list=[[shape[0], shape[1], shape[2], shape[3]]])
    if ret != 0:
        raise RuntimeError(f"Failed to load ONNX for {name}, ret={ret}")
    ret = rknn.build(do_quantization=False, dataset=None)
    if ret != 0:
        raise RuntimeError(f"Failed to build RKNN for {name}, ret={ret}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"Failed to export RKNN for {name}, ret={ret}")
    rknn.release()
    print(f"[rknn] wrote {rknn_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate pooling ONNX/RKNN models used by pool.cpp tests.")
    parser.add_argument("--target", default="rk3588", help="Target platform for RKNN build (default: rk3588)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing RKNN files")
    args = parser.parse_args()

    cases = [
        dict(name="avg_pool2d_float16_1x4", mode="avg", shape=(1, 1, 4, 4), kernel=2, stride=1),
        dict(name="min_pool2d_float16_1x4", mode="min", shape=(1, 1, 4, 4), kernel=2, stride=1),
        dict(name="adaptive_avg_pool2d_float16_1x4_to_2x2", mode="adaptive_avg", shape=(1, 1, 4, 4), output_size=(2, 2)),
        dict(name="global_max_pool2d_float16_1x4", mode="adaptive_max", shape=(1, 1, 4, 4), output_size=(1, 1)),
    ]

    for cfg in cases:
        export_case(target=args.target, force=args.force, **cfg)


if __name__ == "__main__":
    main()
