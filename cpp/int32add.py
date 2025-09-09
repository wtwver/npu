# create rknn model for int32 add operation
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Simple addition operation between x and y (int32)
        return x * y  
        
        
# Create int32 tensors for testing
x = torch.full((1, 2), 1, dtype=torch.int32)
y = torch.full((1, 2), 3, dtype=torch.int32)
print(x)
print(y)


print(f"Input x: {x}")
print(f"Input y: {y}")
print(f"Expected output: {x + y}")

m = Model()

# Export to ONNX with proper int32 support
torch.onnx.export(m, (x, y), "int32add.onnx", 
                  opset_version=11,
                  input_names=['input_x', 'input_y'],
                  output_names=['output'])

## Convert generated ONNX model to RKNN
from rknn.api import RKNN
rknn = RKNN()

rknn.config(target_platform='rk3588')
rknn.load_onnx(model='int32add.onnx')

ret = rknn.build(do_quantization=False, dataset=None)
ret = rknn.export_rknn('int32add.rknn')

print("RKNN model exported successfully!")

# run = rknn.eval_perf('int32add.rknn')
# print(run)