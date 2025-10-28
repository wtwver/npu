import numpy as np
import onnx
from onnx import helper, TensorProto

input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])

weight_values = np.array([
    1., 0., -1.,
    1., 0., -1.,
    1., 0., -1.
], dtype=np.float32).reshape(1, 1, 3, 3)

weight_initializer = helper.make_tensor(
    name='weight',
    data_type=TensorProto.FLOAT,
    dims=weight_values.shape,
    vals=weight_values.flatten().tolist(),
)

conv_node = helper.make_node(
    'Conv',
    inputs=['input', 'weight'],
    outputs=['output'],
    strides=[1, 1],
)

graph = helper.make_graph(
    [conv_node],
    'Conv2DGraph',
    [input_tensor],
    [output_tensor],
    initializer=[weight_initializer],
)

model = helper.make_model(
    graph,
    producer_name='simple-2d-conv',
    opset_imports=[helper.make_opsetid('', 13)],
)

onnx.checker.check_model(model)
onnx.save(model, 'conv2d_simple.onnx')
print('Saved conv2d_simple.onnx')
