#!/usr/bin/env python3
"""
Comprehensive ONNX operations exporter for RK3588 NPU
Exports all operations defined in ops.md to individual ONNX models
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

def create_base_model(op_name: str, inputs: List, outputs: List, nodes: List, output_dir: str = "models") -> str:
    """Create base ONNX model structure"""
    Path(output_dir).mkdir(exist_ok=True)

    try:
        import onnx
        from onnx import helper

        # Create graph and model
        graph = helper.make_graph(nodes, f"{op_name}_graph", inputs, outputs)
        model = helper.make_model(graph, ir_version=10)
        model.opset_import[0].version = 10

        # Save model
        model_path = f"{output_dir}/{op_name}.onnx"
        onnx.save(model, model_path)

        # Validate
        onnx.checker.check_model(model)
        print(f"✓ Created: {model_path} ({Path(model_path).stat().st_size} bytes)")

        return model_path

    except Exception as e:
        print(f"✗ Error creating {op_name}: {e}")
        return None

# Tensor Creation Operations
def create_tensor_creation_ops(output_dir: str = "models") -> List[str]:
    """Create tensor creation operation models"""
    ops = []
    from onnx import helper, TensorProto
    import numpy as np

    tensor_ops = ["zeros", "ones", "empty", "full", "eye", "arange", "linspace"]

    for op in tensor_ops:
        try:
            # Create different shapes for variety
            shapes = [[1, 32], [1, 64, 64], [2, 3, 32, 32]]

            for i, shape in enumerate(shapes):
                # Create a constant tensor with the desired shape and values
                if op == "zeros":
                    data = np.zeros(shape, dtype=np.float32)
                elif op == "ones":
                    data = np.ones(shape, dtype=np.float32)
                elif op == "empty":
                    data = np.zeros(shape, dtype=np.float32)  # empty is typically zeros
                elif op == "full":
                    data = np.full(shape, 1.0, dtype=np.float32)
                elif op == "eye":
                    # For eye, create an identity matrix (simplified version)
                    data = np.eye(min(shape[-2:]), dtype=np.float32)
                    if len(shape) > 2:
                        data = np.broadcast_to(data, shape)
                elif op == "arange":
                    # For arange, create a range tensor
                    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                elif op == "linspace":
                    # For linspace, create evenly spaced values
                    data = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)

                # Create constant node
                tensor = helper.make_tensor("data", TensorProto.FLOAT, shape, data.flatten())
                node = helper.make_node("Constant", [], ["output"], value=tensor)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

                model_path = create_base_model(f"{op}_{i}", [], [output], [node], output_dir)
                if model_path:
                    ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Arithmetic Operations
def create_arithmetic_ops(output_dir: str = "models") -> List[str]:
    """Create arithmetic operation models"""
    ops = []
    from onnx import helper, TensorProto

    arithmetic_ops = ["add", "sub", "mul", "div", "mod", "pow", "neg"]

    for op in arithmetic_ops:
        try:
            input1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 32, 32])
            input2 = helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, 32, 32])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])

            if op == "neg":
                node = helper.make_node("Neg", ["input_1"], ["output"])
            else:
                node = helper.make_node(op.capitalize(), ["input_1", "input_2"], ["output"])

            model_path = create_base_model(f"{op}", [input1, input2], [output], [node], output_dir)
            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Math Functions
def create_math_ops(output_dir: str = "models") -> List[str]:
    """Create math function models"""
    ops = []
    from onnx import helper, TensorProto

    math_ops = ["sin", "cos", "tan", "asin", "acos", "atan", "exp", "exp2", "log", "log2",
                "sqrt", "rsqrt", "abs", "sign", "floor", "ceil", "trunc", "round"]

    for op in math_ops:
        try:
            input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])

            node = helper.make_node(op.capitalize(), ["input"], ["output"])

            model_path = create_base_model(f"{op}", [input1], [output], [node], output_dir)
            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Activation Functions
def create_activation_ops(output_dir: str = "models") -> List[str]:
    """Create activation function models"""
    ops = []
    from onnx import helper, TensorProto

    activation_ops = ["relu", "leaky_relu", "celu", "selu", "silu", "gelu", "elu",
                      "relu6", "hardswish", "hardsigmoid", "sigmoid", "logsigmoid",
                      "softplus", "erf", "quick_gelu"]

    for op in activation_ops:
        try:
            input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])

            # Handle special cases
            if op == "leaky_relu":
                node = helper.make_node("LeakyRelu", ["input"], ["output"], alpha=0.1)
            elif op in ["celu", "selu"]:
                node = helper.make_node(op.upper(), ["input"], ["output"])
            else:
                node = helper.make_node(op.capitalize(), ["input"], ["output"])

            model_path = create_base_model(f"{op}", [input1], [output], [node], output_dir)
            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Reduction Operations
def create_reduction_ops(output_dir: str = "models") -> List[str]:
    """Create reduction operation models"""
    ops = []
    from onnx import helper, TensorProto

    reduction_ops = ["sum", "max", "min", "mean"]

    for op in reduction_ops:
        try:
            input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])

            node = helper.make_node(f"Reduce{op.capitalize()}", ["input"], ["output"])

            model_path = create_base_model(f"reduce_{op}", [input1], [output], [node], output_dir)
            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping reduce_{op}: {e}")

    return ops

# Shape Operations
def create_shape_ops(output_dir: str = "models") -> List[str]:
    """Create shape operation models"""
    ops = []
    from onnx import helper, TensorProto

    shape_ops = ["reshape", "transpose", "flatten", "squeeze", "cat", "split",
                 "repeat", "expand", "unfold", "meshgrid"]

    for op in shape_ops:
        try:
            if op == "reshape":
                input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32])
                shape_input = helper.make_tensor_value_info("shape", TensorProto.INT64, [4])
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 8, 8])
                node = helper.make_node("Reshape", ["input", "shape"], ["output"])
                model_path = create_base_model(f"{op}", [input1, shape_input], [output], [node], output_dir)

            elif op == "transpose":
                input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32])
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])
                node = helper.make_node("Transpose", ["input"], ["output"], perm=[0, 2, 1])
                model_path = create_base_model(f"{op}", [input1], [output], [node], output_dir)

            elif op == "flatten":
                input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32])
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1024])
                node = helper.make_node("Flatten", ["input"], ["output"])
                model_path = create_base_model(f"{op}", [input1], [output], [node], output_dir)

            elif op == "cat":
                input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 32, 32])
                input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 32, 32])
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 32, 32])
                node = helper.make_node("Concat", ["input1", "input2"], ["output"], axis=0)
                model_path = create_base_model(f"{op}", [input1, input2], [output], [node], output_dir)

            else:
                # Skip complex operations for now
                continue

            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Matrix Operations
def create_matrix_ops(output_dir: str = "models") -> List[str]:
    """Create matrix operation models"""
    ops = []
    from onnx import helper, TensorProto

    matrix_ops = ["matmul", "dot", "gemm"]

    for op in matrix_ops:
        try:
            input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 32, 32])
            input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 32, 32])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32])

            if op == "gemm":
                node = helper.make_node("Gemm", ["input1", "input2"], ["output"])
            else:
                node = helper.make_node(op.capitalize(), ["input1", "input2"], ["output"])

            model_path = create_base_model(f"{op}", [input1, input2], [output], [node], output_dir)
            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Convolution Operations
def create_conv_ops(output_dir: str = "models") -> List[str]:
    """Create convolution operation models"""
    ops = []
    from onnx import helper, TensorProto

    conv_ops = ["conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d", "conv1d"]

    for op in conv_ops:
        try:
            if op == "conv2d":
                input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
                weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [16, 3, 3, 3])
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 30, 30])
                node = helper.make_node("Conv", ["input", "weight"], ["output"],
                                      kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
                model_path = create_base_model(f"{op}", [input1, weight], [output], [node], output_dir)

            elif op == "conv1d":
                input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32])
                weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [16, 3, 3])
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 30])
                node = helper.make_node("Conv", ["input", "weight"], ["output"],
                                      kernel_shape=[3], strides=[1], pads=[0, 0])
                model_path = create_base_model(f"{op}", [input1, weight], [output], [node], output_dir)

            else:
                # Skip 3D and transpose convolutions for now
                continue

            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

# Pooling Operations
def create_pooling_ops(output_dir: str = "models") -> List[str]:
    """Create pooling operation models"""
    ops = []
    from onnx import helper, TensorProto

    pooling_ops = ["max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d"]

    for op in pooling_ops:
        try:
            input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])

            if "adaptive" in op:
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 16, 16])
                pool_type = "AveragePool" if "avg" in op else "MaxPool"
                node = helper.make_node(pool_type, ["input"], ["output"],
                                      kernel_shape=[2, 2], strides=[2, 2])
            else:
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 16, 16])
                pool_type = "AveragePool" if "avg" in op else "MaxPool"
                node = helper.make_node(pool_type, ["input"], ["output"],
                                      kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

            model_path = create_base_model(f"{op}", [input1], [output], [node], output_dir)
            if model_path:
                ops.append(model_path)

        except Exception as e:
            print(f"Skipping {op}: {e}")

    return ops

def export_all_operations(output_dir: str = "models") -> Dict[str, List[str]]:
    """Export all operations to ONNX models"""
    print(f"🚀 Exporting all operations to {output_dir}...")

    all_ops = {}

    # Export all operation categories
    print("\n📦 Creating Tensor Creation Operations...")
    all_ops["tensor_creation"] = create_tensor_creation_ops(output_dir)

    print("\n🔢 Creating Arithmetic Operations...")
    all_ops["arithmetic"] = create_arithmetic_ops(output_dir)

    print("\n📐 Creating Math Operations...")
    all_ops["math"] = create_math_ops(output_dir)

    print("\n🔥 Creating Activation Operations...")
    all_ops["activations"] = create_activation_ops(output_dir)

    print("\n📉 Creating Reduction Operations...")
    all_ops["reductions"] = create_reduction_ops(output_dir)

    print("\n🔄 Creating Shape Operations...")
    all_ops["shape"] = create_shape_ops(output_dir)

    print("\n📊 Creating Matrix Operations...")
    all_ops["matrix"] = create_matrix_ops(output_dir)

    print("\n🎯 Creating Convolution Operations...")
    all_ops["convolution"] = create_conv_ops(output_dir)

    print("\n🌊 Creating Pooling Operations...")
    all_ops["pooling"] = create_pooling_ops(output_dir)

    # Summary
    total_ops = sum(len(ops) for ops in all_ops.values())
    print(f"\n✅ Export complete! Created {total_ops} operation models")

    for category, ops in all_ops.items():
        if ops:
            print(f"  • {category}: {len(ops)} models")

    return all_ops

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 ops.py <command> [output_dir]")
        print("Commands:")
        print("  export_all    - Export all operations")
        print("  add <size>    - Create addition model (legacy)")
        print("Example: python3 ops.py export_all models")
        sys.exit(1)

    command = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models"

    if command == "export_all":
        export_all_operations(output_dir)
    elif command == "add":
        if len(sys.argv) < 3:
            print("Usage: python3 ops.py add <size> [output_dir]")
            sys.exit(1)
        size = int(sys.argv[2])
        create_addition_model(size, output_dir)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
