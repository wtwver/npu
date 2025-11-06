from math import exp
import torch, sys
from rknn.api import RKNN

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding=0)

    def forward(self, x, y):
        return self.conv1(x)

size = 2
if sys.argv[1:]:
    size = int(sys.argv[1])
dtype = torch.float16
ops = "conv"
model_path = f"models/{ops}_float16_1x{size}.onnx"

x = torch.full((1, size), 2, dtype=dtype)
y = torch.full((1, size), 1, dtype=dtype)
print(f"Input x: {x}")
print(f"Input y: {y}")
print(f"Expected output: {x + y}")

# Export to ONNX 
m = Model()
torch.onnx.export(m, (x, y), model_path,
                #   opset_version=11,
                  input_names=['input_x', 'input_y'],
                  output_names=['output'])

#### what is dynamo_export
# torch.onnx.dynamo_export(m, (x, y), model_path,
#                 #   opset_version=11,
#                   input_names=['input_x', 'input_y'],
#                   output_names=['output'])

# ONNX to RKNN
model_path = "/home/orangepi/tinygrad/npu/ops_rknn/conv2d_simple.onnx"
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=model_path)
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(model_path.replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")