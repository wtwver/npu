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
    # Use local hello program instead of hardcoded path
    hello_path = "./hello"
    if not os.path.exists(hello_path):
        print("Error: hello program not found!")
        print("Please compile it first:")
        print("gcc -o hello hello.c -I.")
        return False

    # Check if DRM device exists
    if not os.path.exists("/dev/dri/card1"):
        print("Error: DRM device /dev/dri/card1 not found!")
        print("Make sure the RKNPU driver is loaded")
        return False

    return True

def dump_gem_memory(gem_numbers, output_dir="dump"):
    """Dump GEM memory using the hello program"""
    print("=== Dumping GEM Memory for RK3588 NPU ===")

    # Use absolute path to hello program
    hello_path = os.path.abspath("./hello")

    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Clean up any existing dump files for the requested GEM numbers
        for gem_num in gem_numbers:
            for pattern in [f"gem{gem_num}-dump", f"gem{gem_num}_regdump.bin"]:
                existing_file = Path(".") / pattern
                if existing_file.exists():
                    existing_file.unlink()
                    print(f"Cleaned up existing {pattern}")

        # Build command with GEM numbers
        cmd = [hello_path] + [str(gem) for gem in gem_numbers]
        print(f"Running command: {' '.join(cmd)}")

        # Run the hello program from the current directory (not output_dir)
        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              cwd=".")  # Run from current directory

        print("=== GEM Memory Dump Output ===")
        print(result.stdout)

        if result.stderr:
            print("=== Errors ===")
            print(result.stderr)

        # Move created dump files to output directory
        dump_files = []
        # Only look for dump files for the requested GEM numbers
        for gem_num in gem_numbers:
            for pattern in [f"gem{gem_num}-dump", f"gem{gem_num}_regdump.bin"]:
                dump_file = Path(".") / pattern
                if dump_file.exists():
                    dump_files.append(dump_file)
        
        # Remove duplicates and sort
        dump_files = sorted(set(dump_files), key=lambda x: x.name)
        
        if dump_files:
            print(f"\n=== Moving dump files to {output_dir}/ ===")
            moved_count = 0
            for dump_file in dump_files:
                if dump_file.exists():
                    target_path = Path(output_dir) / dump_file.name
                    try:
                        dump_file.rename(target_path)
                        print(f"Moved: {dump_file.name} -> {output_dir}/")
                        moved_count += 1
                    except Exception as e:
                        print(f"Error moving {dump_file.name}: {e}")
            
            print(f"Successfully moved {moved_count} file(s)")
            
            # Show which GEM objects were successfully dumped
            dumped_gems = set()
            for dump_file in dump_files:
                if dump_file.name.startswith("gem") and "_" in dump_file.name:
                    gem_num = dump_file.name.split("_")[0][3:]  # Extract number from "gem1_regdump.bin"
                    dumped_gems.add(gem_num)
            
            if dumped_gems:
                print(f"Successfully dumped GEM objects: {sorted(dumped_gems)}")
        else:
            print("No dump files found to move")
            print("This may indicate that the requested GEM objects are not currently active")
            print("Make sure to run an RKNN model first to create GEM objects")

        # Check what files are now in the output directory
        output_files = list(Path(output_dir).glob("*"))
        if output_files:
            print(f"\n=== Files in {output_dir}/ ===")
            for f in output_files:
                print(f"  {f.name}")
        else:
            print(f"\nNo files found in {output_dir}/")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running hello program: {e}")
        return False

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

    # Determine which GEMs to dump
    gems_to_dump = []
    if args.gems:
        gems_to_dump = args.gems
    elif args.all:
        gems_to_dump = [1, 2, 3]  # Default GEM objects
    
    print(f"Will dump GEM objects: {gems_to_dump}")

    if dump_gem_memory(gems_to_dump, args.output_dir):
        print(f"\n=== GEM Memory Dump Complete ===")
        print(f"Dump files saved to: {args.output_dir}/")
    else:
        print(f"\n=== GEM Memory Dump Failed ===")

    return 0

if __name__ == "__main__":
    sys.exit(main())
