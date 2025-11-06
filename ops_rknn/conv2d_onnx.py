#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import helper, TensorProto
import torch
import torch.nn as nn

# Create a simple Conv2d layer with the weights we want
class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(2, 3), bias=False)
        # Set weights manually
        with torch.no_grad():
            # Output channel 0: [1, 0] for each input channel
            self.conv.weight[0, 0, 0, 0] = 1.0
            self.conv.weight[0, 0, 1, 0] = 0.0
            self.conv.weight[0, 1, 0, 0] = 1.0
            self.conv.weight[0, 1, 1, 0] = 0.0
            self.conv.weight[0, 2, 0, 0] = 1.0
            self.conv.weight[0, 2, 1, 0] = 0.0

            # Output channel 1: [0, 1] for each input channel
            self.conv.weight[1, 0, 0, 0] = 0.0
            self.conv.weight[1, 0, 1, 0] = 1.0
            self.conv.weight[1, 1, 0, 0] = 0.0
            self.conv.weight[1, 1, 1, 0] = 1.0
            self.conv.weight[1, 2, 0, 0] = 0.0
            self.conv.weight[1, 2, 1, 0] = 1.0

            # Output channel 2: [-1, 0] for each input channel
            self.conv.weight[2, 0, 0, 0] = -1.0
            self.conv.weight[2, 0, 1, 0] = 0.0
            self.conv.weight[2, 1, 0, 0] = -1.0
            self.conv.weight[2, 1, 1, 0] = 0.0
            self.conv.weight[2, 2, 0, 0] = -1.0
            self.conv.weight[2, 2, 1, 0] = 0.0

            # Output channel 3: [0, -1] for each input channel
            self.conv.weight[3, 0, 0, 0] = 0.0
            self.conv.weight[3, 0, 1, 0] = -1.0
            self.conv.weight[3, 1, 0, 0] = 0.0
            self.conv.weight[3, 1, 1, 0] = -1.0
            self.conv.weight[3, 2, 0, 0] = 0.0
            self.conv.weight[3, 2, 1, 0] = -1.0

            # Output channel 4: [1, 1] for each input channel
            self.conv.weight[4, 0, 0, 0] = 1.0
            self.conv.weight[4, 0, 1, 0] = 1.0
            self.conv.weight[4, 1, 0, 0] = 1.0
            self.conv.weight[4, 1, 1, 0] = 1.0
            self.conv.weight[4, 2, 0, 0] = 1.0
            self.conv.weight[4, 2, 1, 0] = 1.0

            # Output channel 5: [-1, -1] for each input channel
            self.conv.weight[5, 0, 0, 0] = -1.0
            self.conv.weight[5, 0, 1, 0] = -1.0
            self.conv.weight[5, 1, 0, 0] = -1.0
            self.conv.weight[5, 1, 1, 0] = -1.0
            self.conv.weight[5, 2, 0, 0] = -1.0
            self.conv.weight[5, 2, 1, 0] = -1.0

    def forward(self, x):
        return self.conv(x)

# Create dummy input with the right shape
dummy_input = torch.randn(1, 3, 5, 7)

# Export to ONNX
model = SimpleConv()
torch.onnx.export(
    model,
    dummy_input,
    'conv2d_simple.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    # NO dynamic axes - use fixed shape [1, 3, 5, 7]
)
print('Saved conv2d_simple.onnx')
