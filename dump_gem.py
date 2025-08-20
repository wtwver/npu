#!/usr/bin/env python3
"""
RK3588 NPU GEM Memory Dumper
Python wrapper that uses the existing hello.c program to dump GEM objects
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required programs exist"""
    hello_path = "/home/orangepi/rk3588/rknpu-reverse-engineering/hello"
    if not os.path.exists(hello_path):
        print("Error: hello program not found!")
        print("Please compile it first:")
        print("cd /home/orangepi/rk3588/rknpu-reverse-engineering")
        print("gcc -o hello hello.c -I.")
        return False

    # Check if DRM device exists
    if not os.path.exists("/dev/dri/card1"):
        print("Error: DRM device /dev/dri/card1 not found!")
        print("Make sure the RKNPU driver is loaded")
        return False

    return True

def dump_gem_memory(output_dir="dump"):
    """Dump GEM memory using the hello program"""
    print("=== Dumping GEM Memory for RK3588 NPU ===")

    hello_path = "/home/orangepi/rk3588/rknpu-reverse-engineering/hello"

    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Run the hello program from the output directory
        result = subprocess.run([hello_path],
                              capture_output=True,
                              text=True,
                              cwd=output_dir)

        print("=== GEM Memory Dump Output ===")
        print(result.stdout)

        if result.stderr:
            print("=== Errors ===")
            print(result.stderr)

        # Check for generated dump files
        dump_files = []
        for file in os.listdir(output_dir):
            if file.endswith('.bin') or 'dump' in file.lower():
                dump_files.append(file)

        if dump_files:
            print(f"\n=== Generated Dump Files in {output_dir}/ ===")
            for file in dump_files:
                file_path = Path(output_dir) / file
                size = file_path.stat().st_size
                print(f"  {file}: {size} bytes")
        else:
            print("\nNo dump files were generated")

    except Exception as e:
        print(f"Error running hello program: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Dump GEM objects from RK3588 NPU")
    parser.add_argument("--gems", type=int, nargs='+',
                       help="GEM flink names to dump (e.g., 1 2 3)")
    parser.add_argument("--all", action="store_true",
                       help="Dump all known GEM objects (1, 2, 3)")
    parser.add_argument("--output-dir", default="dump",
                       help="Output directory for dump files")

    args = parser.parse_args()

    if not args.gems and not args.all:
        print("Error: Must specify either --gems or --all")
        parser.print_help()
        return 1

    print("RK3588 NPU GEM Memory Dumper")
    print("=" * 40)

    if not check_dependencies():
        return 1

    print("Dependencies check passed!")
    print(f"\nNote: GEM objects only exist while RKNN models are running")
    print("If you get 'gem open got -1', it means no GEM objects are currently active")
    print("\nTo create GEM objects, you need to:")
    print("1. Run an RKNN model/demo")
    print("2. Or load a neural network model")
    print("3. Then run this script to dump the GEM memory")

    print(f"\nDumping GEM objects to {args.output_dir}/...")

    if dump_gem_memory(args.output_dir):
        print(f"\n=== GEM Memory Dump Complete ===")
        print(f"Dump files saved to: {args.output_dir}/")
    else:
        print(f"\n=== GEM Memory Dump Failed ===")

    return 0

if __name__ == "__main__":
    sys.exit(main())
