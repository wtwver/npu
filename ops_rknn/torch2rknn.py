import torch
import sys
from rknn.api import RKNN

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()

  def forward(self, x, y):
    return x / y

size = 4
kernel_size = 2
if sys.argv[1:]:
  size = int(sys.argv[1])
if len(sys.argv) > 2:
  kernel_size = int(sys.argv[2])
dtype = torch.float16
ops = "div"
model_path = f"models/{ops}_float16_1x{size}.onnx"

shape = (1, 1, size, size)
x = torch.arange(1, size * size + 1, dtype=dtype).reshape(shape)
y = torch.arange(1, size * size + 1, dtype=dtype).reshape(shape)
print(f"Input x: {x} \nInput y: {y}")
# print(f"Expected output: {torch.nn.functional.max_pool2d(x, kernel_size, stride=1)}")

# Export to ONNX
m = Model()
torch.onnx.export(m, (x, y), model_path,
                  input_names=['input_x', 'input_y'],
                  output_names=['output'])

# ONNX to RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=model_path, input_size_list=[[1, 1, size, size]])
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(model_path.replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")
