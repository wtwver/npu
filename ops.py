#!/usr/bin/env python3
"""
Simple ONNX addition model creator for RK3588 NPU
"""

import sys
from pathlib import Path

def create_addition_model(size=32, output_dir="models"):
    """Create ONNX addition model"""
    Path(output_dir).mkdir(exist_ok=True)

    try:
        import onnx
        from onnx import helper, TensorProto

        # Create inputs/outputs
        input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, size, size])
        input2 = helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, size, size])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, size, size])

        # Create addition node
        add_node = helper.make_node("Add", ["input_1", "input_2"], ["output"])

        # Create graph and model
        graph = helper.make_graph([add_node], "addition", [input1, input2], [output])
        model = helper.make_model(graph, ir_version=10)
        model.opset_import[0].version = 10

        # Save model
        model_path = f"{output_dir}/add_{size}.onnx"
        onnx.save(model, model_path)

        # Validate
        onnx.checker.check_model(model)
        print(f"✓ Created: {model_path} ({Path(model_path).stat().st_size} bytes)")

        return model_path

    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 ops.py <size> [output_dir]")
        print("Example: python3 ops.py 32 models")
        sys.exit(1)

    size = int(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models"

    create_addition_model(size, output_dir)

if __name__ == "__main__":
    main()
