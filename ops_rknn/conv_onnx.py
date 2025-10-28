import onnx
from onnx import helper, TensorProto

# I/O tensors (N, C, W)
input_tensor  = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 1, 5])
weight_tensor = helper.make_tensor_value_info('weight', TensorProto.FLOAT, [1, 1, 3])  # [out_c, in_c, k]
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 3])  # 5-3+1 = 3

# Conv node: stride=1, no padding (omit pads)  ── OR ── use pads=[0, 0]
conv_node = helper.make_node(
    'Conv',
    inputs=['input', 'weight'],
    outputs=['output'],
    strides=[1],
    # pads=[0, 0],  # <- also valid; pick one approach
)

# Graph & model (opset 13+ is fine)
graph = helper.make_graph([conv_node], 'Conv1DGraph', [input_tensor, weight_tensor], [output_tensor])
model = helper.make_model(graph, producer_name='simple-1d-conv', opset_imports=[helper.make_opsetid('', 13)])

onnx.checker.check_model(model)
onnx.save(model, 'conv1d_simple.onnx')
print("Saved conv1d_simple.onnx")
