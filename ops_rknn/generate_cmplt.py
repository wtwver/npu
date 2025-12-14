import torch
import sys
from rknn.api import RKNN
from pathlib import Path

class Model(torch.nn.Module):
  def forward(self, x, y):
    return (x < y).to(torch.float16)

size = 1
if sys.argv[1:]:
  size = int(sys.argv[1])
dtype = torch.float16
ops = "cmplt"
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / f"{ops}_float16_1x{size}.onnx"

shape = (1, 1, size, size)
x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
y = torch.linspace(2, -2, size * size, dtype=dtype).reshape(shape)
print(f"Input x: {x} \nInput y: {y}")
print(f"Expected output: {(x < y).to(torch.float16)}")

m = Model()
torch.onnx.export(m, (x, y), str(model_path),
                  input_names=['input_x', 'input_y'],
                  output_names=['output'])

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=str(model_path), input_size_list=[[1, 1, size, size]])
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(str(model_path).replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")
