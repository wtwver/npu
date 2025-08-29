#!/usr/bin/env python3
"""
Fix the Mod operation by adding the required fmod attribute for floating-point types
The error shows: "fmod attribute must be true for floating point types"
Also fix RK3588 limitation: "Mod supports only when divisor has one element"
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_fixed_mod_model():
    """Create mod model with correct fmod attribute"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 32, 32])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, 32, 32])
    
    # Create Mod node with fmod=True attribute
    mod_node = helper.make_node(
        "Mod",
        inputs=["input_1", "input_2"],
        outputs=["output"],
        fmod=1  # True for floating-point mod operation
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])
    
    # Create graph and model
    graph = helper.make_graph([mod_node], "mod_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/mod_fixed.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created fixed mod model: {model_path}")
    
    return model_path

def create_mod_with_scalar_divisor():
    """Create mod model with scalar divisor (RK3588 compatible)"""
    
    # Create input tensor
    input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 32, 32])
    
    # Create a constant scalar divisor
    divisor_data = np.array([2.0], dtype=np.float32)
    divisor_tensor = helper.make_tensor("divisor", TensorProto.FLOAT, [1], divisor_data.flatten())
    
    # Create constant node for scalar divisor
    const_node = helper.make_node("Constant", [], ["divisor"], value=divisor_tensor)
    
    # Create Mod node with scalar divisor
    mod_node = helper.make_node(
        "Mod",
        inputs=["input_1", "divisor"],
        outputs=["output"],
        fmod=1
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])
    
    # Create graph and model
    graph = helper.make_graph([const_node, mod_node], "mod_scalar_graph", [input1], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/mod_scalar.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created mod model with scalar divisor: {model_path}")
    
    return model_path

def create_mod_with_int_inputs():
    """Create mod model with integer inputs (alternative approach)"""
    
    # Create input tensors with integer types
    input1 = helper.make_tensor_value_info("input_1", TensorProto.INT32, [1, 32, 32])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.INT32, [1, 32, 32])
    
    # Create Mod node (no fmod attribute needed for integers)
    mod_node = helper.make_node(
        "Mod",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.INT32, [1, 32, 32])
    
    # Create graph and model
    graph = helper.make_graph([mod_node], "mod_int_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/mod_int.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created mod model with integer inputs: {model_path}")
    
    return model_path

def create_mod_with_computation():
    """Create mod model using computational approach (like we did with empty)"""
    
    # Create input tensors
    input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 32, 32])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, 32, 32])
    
    # Create a constant tensor with small positive values to avoid division by zero
    const_data = np.ones([1, 32, 32], dtype=np.float32) * 0.001
    const_tensor = helper.make_tensor("epsilon", TensorProto.FLOAT, [1, 32, 32], const_data.flatten())
    
    # Create constant node
    const_node = helper.make_node("Constant", [], ["epsilon"], value=const_tensor)
    
    # Create Add to avoid division by zero
    add_node = helper.make_node(
        "Add",
        inputs=["input_2", "epsilon"],
        outputs=["safe_divisor"]
    )
    
    # Create Div operation (division)
    div_node = helper.make_node(
        "Div",
        inputs=["input_1", "safe_divisor"],
        outputs=["quotient"]
    )
    
    # Create Floor operation
    floor_node = helper.make_node(
        "Floor",
        inputs=["quotient"],
        outputs=["floor_quotient"]
    )
    
    # Create Mul operation (floor_quotient * divisor)
    mul_node = helper.make_node(
        "Mul",
        inputs=["floor_quotient", "safe_divisor"],
        outputs=["product"]
    )
    
    # Create Sub operation (input_1 - product = remainder)
    sub_node = helper.make_node(
        "Sub",
        inputs=["input_1", "product"],
        outputs=["output"]
    )
    
    # Create output
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])
    
    # Create graph and model
    graph = helper.make_graph(
        [const_node, add_node, div_node, floor_node, mul_node, sub_node], 
        "mod_computational_graph", 
        [input1, input2], 
        [output]
    )
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Save model
    model_path = "models/mod_computational.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created computational mod model: {model_path}")
    
    return model_path

if __name__ == "__main__":
    print("Creating fixed mod models...")
    
    # Create all versions
    fixed_path = create_fixed_mod_model()
    scalar_path = create_mod_with_scalar_divisor()
    int_path = create_mod_with_int_inputs()
    comp_path = create_mod_with_computation()
    
    print(f"\nFixed mod models created:")
    print(f"1. Fixed fmod attribute: {fixed_path}")
    print(f"2. Scalar divisor (RK3588 compatible): {scalar_path}")
    print(f"3. Integer inputs: {int_path}")
    print(f"4. Computational approach: {comp_path}")
    
    print("\nNow you can test these models on CPU and convert to RKNN")
