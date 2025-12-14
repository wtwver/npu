import torch
import sys
from rknn.api import RKNN
from pathlib import Path

class Model(torch.nn.Module):
  def forward(self, x, y, a, b):
    return torch.where(x < y, a, b)

size = 2
if sys.argv[1:]:
  size = int(sys.argv[1])
if size < 2:
  print("RKNN Toolkit2 crashes compiling Where for 1x1; use size>=2 (e.g. `python3 generate_where.py 2`).", file=sys.stderr)
  raise SystemExit(2)
dtype = torch.float16
ops = "where"
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / f"{ops}_float16_1x{size}.onnx"

shape = (1, 1, size, size)
x = torch.linspace(-2, 2, size * size, dtype=dtype).reshape(shape)
y = torch.linspace(2, -2, size * size, dtype=dtype).reshape(shape)
a = torch.linspace(-1, 1, size * size, dtype=dtype).reshape(shape)
b = torch.linspace(1, -1, size * size, dtype=dtype).reshape(shape)
print(f"Input x: {x} \nInput y: {y}\nInput a: {a}\nInput b: {b}")
print(f"Expected output: {torch.where(x < y, a, b)}")

m = Model()
torch.onnx.export(m, (x, y, a, b), str(model_path),
                  input_names=['input_x', 'input_y', 'input_a', 'input_b'],
                  output_names=['output'])

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=str(model_path), input_size_list=[[1, 1, size, size], [1, 1, size, size], [1, 1, size, size], [1, 1, size, size]])
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(str(model_path).replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")
