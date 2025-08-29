#!/usr/bin/env python3
"""
Create ONNX Equal model for testing
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_equal_model():
    """Create Equal model with two float inputs by converting to int first"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 32, 32])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, 32, 32])
    
    # Create Cast nodes to convert float to int
    cast1_node = helper.make_node(
        "Cast",
        inputs=["input_1"],
        outputs=["input_1_int"],
        to=TensorProto.INT32
    )
    
    cast2_node = helper.make_node(
        "Cast",
        inputs=["input_2"],
        outputs=["input_2_int"],
        to=TensorProto.INT32
    )
    
    # Create Equal node with integer inputs
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_1_int", "input_2_int"],
        outputs=["output"]
    )
    
    # Create output (Equal produces boolean output)
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 32, 32])
    
    # Create graph and model
    graph = helper.make_graph([cast1_node, cast2_node, equal_node], "equal_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/15_eq.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created Equal model: {model_path}")
    
    return model_path

def create_equal_with_int_inputs():
    """Create Equal model with integer inputs"""
    
    # Create input tensors with integer types
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 32, 32])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 32, 32])
    
    # Create Equal node
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 32, 32])
    
    # Create graph and model
    graph = helper.make_graph([equal_node], "equal_int_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/15_eq_int.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created Equal model with integer inputs: {model_path}")
    
    return model_path

def create_equal_simple():
    """Create simple Equal model with small integer inputs"""
    
    # Create input tensors with small integer types
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 8, 8])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 8, 8])
    
    # Create Equal node
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 8, 8])
    
    # Create graph and model
    graph = helper.make_graph([equal_node], "equal_simple_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/15_eq_simple.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created simple Equal model: {model_path}")
    
    return model_path

def create_equal_int8():
    """Create Equal model with UINT8 inputs (RK3588 compatible)"""
    
    # Create input tensors with UINT8 types
    input1 = helper.make_tensor_value_info("input_1", TensorProto.UINT8, [1, 8, 8])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.UINT8, [1, 8, 8])
    
    # Create Equal node
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 8, 8])
    
    # Create graph and model
    graph = helper.make_graph([equal_node], "equal_uint8_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/15_eq_uint8.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created Equal model with UINT8 inputs: {model_path}")
    
    return model_path

if __name__ == "__main__":
    print("Creating Equal models...")
    
    # Create all versions
    float_path = create_equal_model()
    int_path = create_equal_with_int_inputs()
    simple_path = create_equal_simple()
    int8_path = create_equal_int8()
    
    print(f"\nEqual models created:")
    print(f"1. Float inputs (with cast): {float_path}")
    print(f"2. Integer inputs: {int_path}")
    print(f"3. Simple integer: {simple_path}")
    print(f"4. INT8 inputs: {int8_path}")
    
    print("\nNow you can test these models on CPU and convert to RKNN")
