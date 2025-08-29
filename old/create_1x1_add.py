#!/usr/bin/env python3
"""
Create a simple 1x1 add ONNX model for RK3588 NPU testing
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_1x1_add_model():
    """Create a minimal 1x1 add ONNX model using float32 for compatibility"""
    
    # Use float32 instead of uint8 to avoid type issues in ONNX Runtime
    input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 1])
    input2 = helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, 1])
    
    # Create Add node
    add_node = helper.make_node(
        "Add",
        inputs=["input_1", "input_2"],
        outputs=["output"]
    )
    
    # Output is also float32
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    
    # Create graph and model
    graph = helper.make_graph([add_node], "add_1x1_graph", [input1, input2], [output])
    model = helper.make_model(graph, ir_version=10)
    model.opset_import[0].version = 10
    
    # Ensure directory exists
    import os
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/add_1x1.onnx"
    onnx.save(model, model_path)
    
    # Validate
    onnx.checker.check_model(model)
    print(f"✓ Created 1x1 add model: {model_path}")
    
    return model_path


def test_1x1_add_model():
    """Test the created model with sample inputs"""
    try:
        import onnxruntime as ort
        
        # Load model
        session = ort.InferenceSession('models/add_1x1.onnx')
        
        # Prepare test inputs (seeded for consistency)
        np.random.seed(42)
        input1_data = np.random.uniform(0, 10, size=(1, 1)).astype(np.float32)
        input2_data = np.random.uniform(0, 10, size=(1, 1)).astype(np.float32)
        
        inputs = {
            'input_1': input1_data,
            'input_2': input2_data
        }
        
        # Run inference
        outputs = session.run(None, inputs)
        result = outputs[0]
        
        # Verify result
        expected = input1_data + input2_data
        is_correct = np.allclose(result, expected, rtol=1e-5)
        
        print(f"✓ Model test passed: {is_correct}")
        print(f"  Input 1: {input1_data.flatten()[0]:.3f}")
        print(f"  Input 2: {input2_data.flatten()[0]:.3f}")
        print(f"  Output:  {result.flatten()[0]:.3f}")
        print(f"  Expected: {expected.flatten()[0]:.3f}")
        
        return is_correct
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Creating 1x1 add ONNX model...")
    model_path = create_1x1_add_model()
    if model_path:
        test_1x1_add_model()
    print("Done!")