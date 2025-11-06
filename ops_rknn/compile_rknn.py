#!/usr/bin/env python3
from rknn.api import RKNN

model_path = "/home/orangepi/tinygrad/npu/ops_rknn/conv2d_simple.onnx"
rknn_path = model_path.replace('.onnx', '.rknn')

rknn = RKNN()
rknn.config(target_platform='rk3588')
# Specify the input shape explicitly to avoid dynamic input issue
rknn.load_onnx(model=model_path, input_size_list=[[1, 3, 5, 7]])
rknn.build(do_quantization=False, dataset=None)
rknn.export_rknn(rknn_path)
print("RKNN model exported successfully!")
