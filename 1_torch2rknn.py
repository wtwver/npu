import torch, sys
from rknn.api import RKNN

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return x * y
        
size = 1
dtype = torch.int32
model_path = f"models/mul_int32_{size}x{size}.onnx"

x = torch.full((1, size), 2, dtype=dtype)
y = torch.full((1, size), 1, dtype=dtype)
print(f"Input x: {x}")
print(f"Input y: {y}")
print(f"Expected output: {x + y}")

# Export to ONNX 
m = Model()
torch.onnx.export(m, (x, y), model_path,
                  opset_version=11,
                  input_names=['input_x', 'input_y'],
                  output_names=['output'])

# ONNX to RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=model_path)
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(model_path.replace('.onnx', '.rknn'))
print("RKNN model exported successfully!")
