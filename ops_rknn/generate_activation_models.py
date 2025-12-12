import os
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from rknn.api import RKNN

# Dump detailed runtime layer info for every activation build
os.environ.setdefault("RKNN_LOG_LEVEL", "5")


ACTIVATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # "relu": torch.relu,
    # "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.01),
    # "celu": lambda x: F.celu(x, alpha=1.0),
    # "selu": F.selu,
    # "silu": F.silu,
    # "swish": lambda x: x * torch.sigmoid(x),
    # "softsign": F.softsign,
    # "sigmoid": torch.sigmoid,
    # "logsigmoid": F.logsigmoid,
    # "hardsigmoid": F.hardsigmoid,
    # "softplus": F.softplus,
    # "gelu": lambda x: F.gelu(x, approximate="tanh"),
    # "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    # "elu": lambda x: F.elu(x, alpha=1.0),
    # "relu6": F.relu6,
    # "hardswish": F.hardswish,
    # "mish": F.mish,
    # "softmax": lambda x: F.softmax(x, dim=-1),
    # "log_softmax": lambda x: F.log_softmax(x, dim=-1),
    "tanh": torch.tanh,
}

# WIDTHS: List[int] = [1, 2, 4, 16, 64, 4096]
WIDTHS: List[int] = [16]


def build_rknn_model(name: str, width: int) -> None:
    shape = (1, 1, 1, width)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = f"{models_dir}/{name}_float16_1x{width}.onnx"
    rknn_path = onnx_path.replace(".onnx", ".rknn")

    x = torch.linspace(-2.0, 2.0, steps=width, dtype=torch.float16).reshape(shape)
    print(f"\n{name}: exporting ONNX (width={width}, shape={shape})")
    expected = ACTIVATIONS[name](x)
    print(f"  Expected head: {expected.flatten().tolist()[:4]}")

    class ActivationModel(torch.nn.Module):
        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return ACTIVATIONS[name](inp)

    torch.onnx.export(
        ActivationModel(),
        x,
        onnx_path,
        input_names=["input_x"],
        output_names=["output"],
        opset_version=15,
        do_constant_folding=True,
    )

    rknn = RKNN()
    rknn.config(
        target_platform="rk3588",
        single_core_mode=True,
        remove_reshape=True,
        disable_rules=["conv_eltwise_activation_fuse"],
    )
    ret = rknn.load_onnx(model=onnx_path, input_size_list=[[1, 1, 1, width]])
    if ret != 0:
        raise RuntimeError(f"Failed to load {onnx_path} (ret={ret})")
    ret = rknn.build(do_quantization=False, dataset=None)
    if ret != 0:
        raise RuntimeError(f"Failed to build {onnx_path} (ret={ret})")
    # Re-init runtime so each model emits layer info under RKNN_LOG_LEVEL=5
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"Failed to init runtime for {onnx_path} (ret={ret})")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"Failed to export {rknn_path} (ret={ret})")
    rknn.release()
    print(f"  RKNN exported: {rknn_path}")


def main() -> None:
    for name in ACTIVATIONS:
        for width in WIDTHS:
            build_rknn_model(name, width)


if __name__ == "__main__":
    main()
