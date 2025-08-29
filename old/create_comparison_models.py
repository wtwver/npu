#!/usr/bin/env python3
"""
Create ONNX comparison operation models for testing
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_less_model():
    """Create Less model with INT32 inputs"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 8, 8])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 8, 8])
    
    # Create Less node
    less_node = helper.make_node(
        "Less",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Create output (Less produces boolean output)
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 8, 8])
    
    # Create graph and model
    graph = helper.make_graph([less_node], "less_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/17_lt.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created Less model: {model_path}")
    
    return model_path

def create_less_equal_model():
    """Create LessOrEqual model using Less + Equal + Or"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 8, 8])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 8, 8])
    
    # Create Less node
    less_node = helper.make_node(
        "Less",
        inputs=["input_1", "input_2"],
        outputs=["less_output"]
    )
    
    # Create Equal node
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_1", "input_2"],
        outputs=["equal_output"]
    )
    
    # Create Or node to combine Less and Equal
    or_node = helper.make_node(
        "Or",
        inputs=["less_output", "equal_output"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 8, 8])
    
    # Create graph and model
    graph = helper.make_graph([less_node, equal_node, or_node], "less_equal_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/18_le.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created LessOrEqual model: {model_path}")
    
    return model_path

def create_greater_model():
    """Create Greater model with INT32 inputs"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 8, 8])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 8, 8])
    
    # Create Greater node
    greater_node = helper.make_node(
        "Greater",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 8, 8])
    
    # Create graph and model
    graph = helper.make_graph([greater_node], "greater_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/19_gt.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created Greater model: {model_path}")
    
    return model_path

def create_greater_equal_model():
    """Create GreaterOrEqual model using Greater + Equal + Or"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 8, 8])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 8, 8])
    
    # Create Greater node
    greater_node = helper.make_node(
        "Greater",
        inputs=["input_1", "input_2"],
        outputs=["greater_output"]
    )
    
    # Create Equal node
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_1", "input_2"],
        outputs=["equal_output"]
    )
    
    # Create Or node to combine Greater and Equal
    or_node = helper.make_node(
        "Or",
        inputs=["greater_output", "equal_output"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 8, 8])
    
    # Create graph and model
    graph = helper.make_graph([greater_node, equal_node, or_node], "greater_equal_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/20_ge.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created GreaterOrEqual model: {model_path}")
    
    return model_path

def create_simple_comparison_models():
    """Create simple comparison models with small INT32 inputs"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 4, 4])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 4, 4])
    
    # Create comparison nodes
    less_node = helper.make_node("Less", ["input_1", "input_2"], ["less_output"])
    greater_node = helper.make_node("Greater", ["input_1", "input_2"], ["gt_output"])
    equal_node = helper.make_node("Equal", ["input_1", "input_2"], ["eq_output"])
    
    # Create outputs
    less_output = helper.make_tensor_value_info("less_output", TensorProto.BOOL, [1, 4, 4])
    gt_output = helper.make_tensor_value_info("gt_output", TensorProto.BOOL, [1, 4, 4])
    eq_output = helper.make_tensor_value_info("eq_output", TensorProto.BOOL, [1, 4, 4])
    
    # Create graph and model
    graph = helper.make_graph(
        [less_node, greater_node, equal_node], 
        "simple_comparison_graph", 
        [input1, input2], 
        [less_output, gt_output, eq_output]
    )
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/comparison_simple.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created simple comparison model: {model_path}")
    
    return model_path

if __name__ == "__main__":
    print("Creating comparison operation models...")
    
    # Create all comparison models
    less_path = create_less_model()
    le_path = create_less_equal_model()
    gt_path = create_greater_model()
    ge_path = create_greater_equal_model()
    simple_path = create_simple_comparison_models()
    
    print(f"\nComparison models created:")
    print(f"1. Less: {less_path}")
    print(f"2. LessOrEqual: {le_path}")
    print(f"3. Greater: {gt_path}")
    print(f"4. GreaterOrEqual: {ge_path}")
    print(f"5. Simple comparison: {simple_path}")
    
    print("\nNow you can test these models on CPU and convert to RKNN")
