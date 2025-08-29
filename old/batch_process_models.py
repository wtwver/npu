#!/usr/bin/env python3
import os
import subprocess
import sys
import glob

def convert_and_benchmark(model_name):
    """Convert ONNX to RKNN and run benchmark"""
    onnx_file = f"models/{model_name}.onnx"
    rknn_file = f"models/{model_name}.rknn"

    if not os.path.exists(onnx_file):
        print(f"❌ {onnx_file} not found")
        return False

    print(f"🔄 Converting {model_name}...")

    # Convert to RKNN
    convert_cmd = f"python3 -m rknn.api.rknn_convert -t rk3588 -i {onnx_file} -o models/"
    result = subprocess.run(convert_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Conversion failed for {model_name}: {result.stderr}")
        return False

    print(f"✅ Conversion successful for {model_name}")

    # Run benchmark
    benchmark_cmd = f"./rknn_benchmark {rknn_file}"
    result = subprocess.run(benchmark_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Benchmark failed for {model_name}: {result.stderr}")
        return False

    # Extract performance metrics
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Avg Time' in line and 'Avg FPS' in line:
            print(f"📊 {model_name}: {line.strip()}")
            break

    print(f"✅ Benchmark successful for {model_name}")
    return True

def main():
    # Get all ONNX files
    onnx_files = glob.glob("models/*.onnx")
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in onnx_files]

    # Skip models that already have RKNN files
    existing_rknn = glob.glob("models/*.rknn")
    existing_names = [os.path.splitext(os.path.basename(f))[0] for f in existing_rknn]
    model_names = [name for name in model_names if name not in existing_names]

    print(f"📋 Found {len(model_names)} models to process")

    successful = 0
    failed = 0

    for i, model_name in enumerate(model_names):
        print(f"\n🚀 Processing {i+1}/{len(model_names)}: {model_name}")
        try:
            if convert_and_benchmark(model_name):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Exception processing {model_name}: {e}")
            failed += 1

    print(f"\n🎉 Batch processing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    main()
