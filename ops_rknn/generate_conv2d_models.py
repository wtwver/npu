#!/usr/bin/env python3
"""
Generic Conv2D Model Generator for RKNN
Generates ONNX and RKNN models with different kernel sizes for testing
"""
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

def create_conv2d_model(in_channels, out_channels, kernel_size, groups=1, bias=False):
    """Create a PyTorch Conv2d model with specified parameters"""

    class SimpleConv(nn.Module):
        def __init__(self, in_ch, out_ch, k_size, groups_val, bias_val):
            super(SimpleConv, self).__init__()
            # Calculate kernel height and width
            if isinstance(k_size, tuple):
                k_h, k_w = k_size
            else:
                k_h = k_w = k_size

            self.conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(k_h, k_w),
                groups=groups_val,
                bias=bias_val
            )

            # Set deterministic weights for testing
            with torch.no_grad():
                weight = self.conv.weight
                in_ch_per_group = in_ch // groups_val

                # For each output channel
                for oc in range(out_ch):
                    # For each input channel within the group
                    for ic in range(in_ch_per_group):
                        # For each kernel position
                        for kh in range(k_h):
                            for kw in range(k_w):
                                # Create a simple pattern for verification
                                # Use a mix of values: 1, 0, -1 based on position
                                val = 1.0 if (oc + kh + kw) % 3 == 0 else ( -1.0 if (oc + kh + kw) % 3 == 1 else 0.0)
                                weight[oc, ic, kh, kw] = val

        def forward(self, x):
            return self.conv(x)

    return SimpleConv(in_channels, out_channels, kernel_size, groups, bias)

def generate_onnx(model, dummy_input, output_path):
    """Export PyTorch model to ONNX format"""
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
    )
    print(f"ONNX model saved: {output_path}")

def generate_rknn(onnx_path, target_platform='rk3588'):
    """Compile ONNX model to RKNN format"""
    try:
        from rknn.api import RKNN

        rknn_path = onnx_path.replace('.onnx', '.rknn')

        rknn = RKNN()
        rknn.config(target_platform=target_platform)

        # Get input shape from dummy input
        input_size = [[1, 3, 5, 7]]  # Default for testing

        rknn.load_onnx(model=onnx_path, input_size_list=input_size)
        rknn.build(do_quantization=False, dataset=None)
        rknn.export_rknn(rknn_path)

        print(f"RKNN model exported: {rknn_path}")
        return rknn_path

    except ImportError:
        print("RKNN toolkit not available. Please install rknn-toolkit2")
        return None
    except Exception as e:
        print(f"Error compiling to RKNN: {e}")
        return None

def main():
    # Define all the test cases
    n = 6
    test_cases = [
        # (out_channels, in_channels, kernel_h, kernel_w, groups)
        (n, 3, 2, 1, 1),   
        (n, 3, 2, 3, 1),   
        (n, 3, 2, 3, 1),   
        (n, 3, 2, 5, 1),   
        (n, 3, 3, 1, 1),   
        (n, 3, 3, 3, 1),   
        # (n, 3, 3, 3, 3),   # (6,3,3,3) with groups=3 (depthwise: 6 output channels, 1 input channel per group)
        (n, 3, 3, 5, 1),   
        (n, 3, 1, 1, 1)
    ]

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    dummy_input = torch.randn(1, 3, 5, 7, dtype=torch.float16)

    for out_ch, in_ch, k_h, k_w, groups in test_cases:
        print(f"\n{'='*60}")
        print(f"Generating model: weight=({out_ch},{in_ch},{k_h},{k_w})")
        if groups > 1:
            print(f"  groups={groups} (depthwise)")
        print(f"{'='*60}")

        # Create model
        model = create_conv2d_model(in_ch, out_ch, (k_h, k_w), groups)
        model = model.half()
        model.eval()

        # Define output paths
        model_name = f"conv2d_{k_h}x{k_w}"
        if groups > 1:
            model_name += f"_g{groups}"
        onnx_path = models_dir / f"{model_name}.onnx"
        rknn_path = models_dir / f"{model_name}.rknn"

        # Generate ONNX
        try:
            generate_onnx(model, dummy_input, str(onnx_path))

            # Generate RKNN (if toolkit available)
            if generate_rknn(str(onnx_path)):
                print(f"✓ Successfully generated {model_name}")
            else:
                print(f"⚠ ONNX generated, but RKNN compilation failed for {model_name}")

        except Exception as e:
            print(f"✗ Error generating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Model generation complete!")
    print(f"Generated in: {models_dir.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
