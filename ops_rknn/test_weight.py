#!/usr/bin/env python3
import numpy as np
import onnx

# Load the ONNX model
model = onnx.load('conv2d_simple.onnx')

# Find the Conv node
conv_node = None
for node in model.graph.node:
    if node.op_type == 'Conv':
        conv_node = node
        break

if conv_node is None:
    print("ERROR: No Conv node found")
    exit(1)

print(f"Conv node: {conv_node.name}")
print(f"Inputs: {conv_node.input}")
print(f"Outputs: {conv_node.output}")

# Find the weight initializer
weight_name = None
for inp in conv_node.input:
    if inp != conv_node.input[0]:  # Skip input tensor
        weight_name = inp
        break

if weight_name is None:
    print("ERROR: No weight found")
    exit(1)

print(f"\nWeight name: {weight_name}")

# Find the weight tensor
for init in model.graph.initializer:
    if init.name == weight_name:
        weight_data = np.frombuffer(init.raw_data, dtype=np.float32)
        print(f"Weight shape: {init.dims}")
        print(f"Weight size: {weight_data.size}")
        print(f"First 20 weights: {weight_data[:20]}")

        # Reshape to expected format [6, 3, 2, 1]
        weight_reshaped = weight_data.reshape(6, 3, 2, 1)
        print(f"\nWeight (6, 3, 2, 1):")
        for oc in range(6):
            print(f"  Output Channel {oc}:")
            for ic in range(3):
                print(f"    Input Channel {ic}:")
                for h in range(2):
                    print(f"      [{weight_reshaped[oc, ic, h, 0]}]")
        break
