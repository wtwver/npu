 import sys
import os
import torch
from rknn.api import RKNN


class SiLUModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.sigmo(x)


def main():
    width = 16
    if len(sys.argv) > 1:
        width = int(sys.argv[1])

    shape = (1, 1, 1, width)  # NHWC: N=1, H=1, W=1, C=width
    ops = "silu"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = f"{models_dir}/{ops}_float16_1x{width}.onnx"
    rknn_path = onnx_path.replace(".onnx", ".rknn")

    x = torch.arange(1, width + 1, dtype=torch.float16).reshape(shape)
    print(f"Input shape: {shape}, width={width}")
    print(f"Expected output: {torch.nn.functional.silu(x)}")

    model = SiLUModel()
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["input_x"],
        output_names=["output"],
        # ONNX SiLU op was added in opset 15; use it so the exported graph contains SiLU
        opset_version=15,
        do_constant_folding=True,
    )
    print(f"ONNX saved to {onnx_path}")

    rknn = RKNN()
    # disable_rules can turn off specific fusion passes; disable conv_eltwise_activation_fuse
    rknn.config(
        target_platform="rk3588",
        single_core_mode=True,
        remove_reshape=True,
        disable_rules=["conv_eltwise_activation_fuse"],
    )
    ret = rknn.load_onnx(model=onnx_path, input_size_list=[[1, 1, 1, width]])
    if ret != 0:
        raise RuntimeError(f"Failed to load ONNX, ret={ret}")
    ret = rknn.build(do_quantization=False, dataset=None)
    if ret != 0:
        raise RuntimeError(f"Failed to build RKNN, ret={ret}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"Failed to export RKNN, ret={ret}")
    rknn.release()
    print(f"RKNN saved to {rknn_path}")


if __name__ == "__main__":
    main()
