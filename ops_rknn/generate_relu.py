import torch
import sys
from rknn.api import RKNN

class Model(torch.nn.Module):
  def forward(self, x):
    return torch.relu(x)

size = 2
if sys.argv[1:]:
  size = int(sys.argv[1])
dtype = torch.float16
ops = "relu"
model_path = f"models/{ops}_float16_1x{size}.onnx"

shape = (1, 1, 1, size)
x = torch.full(shape, 2, dtype=dtype)
print(f"Input x: {x}")
print(f"Expected output: {torch.relu(x)}")

# Export to ONNX 
m = Model()
torch.onnx.export(m, x, model_path,
                  input_names=['input_x'],
                  output_names=['output'])

# ONNX to RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=model_path, input_size_list=[[1, 1, 1, size]])
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(model_path.replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")
