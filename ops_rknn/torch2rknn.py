from math import exp
import torch, sys
from rknn.api import RKNN

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # return x+y
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        # Perform addition in FP32
        return x + y

size = 2
if sys.argv[1:]:
    size = int(sys.argv[1])
dtype = torch.float16
ops = "add"
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
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=model_path)
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(model_path.replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")