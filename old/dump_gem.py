#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    if not os.path.exists("./hello"):
        print("Error: hello program not found!")
        print("Please compile it first:")
        print("gcc -o hello hello.c -I.")
        return False
    if not os.path.exists("/dev/dri/card1"):
        print("Error: DRM device /dev/dri/card1 not found!")
        print("Make sure the RKNPU driver is loaded")
        return False
    return True

def dump_gem_memory(gem_numbers, output_dir="dump"):
    print("=== Dumping GEM Memory for RK3588 NPU ===")
    hello_path = os.path.abspath("./hello")
    try:
        Path(output_dir).mkdir(exist_ok=True)
        for gem_num in gem_numbers:
            for pattern in [f"gem{gem_num}-dump", f"gem{gem_num}_regdump.bin"]:
                p = Path(".") / pattern
                if p.exists():
                    p.unlink()
                    print(f"Cleaned up existing {pattern}")
        cmd = [hello_path] + [str(gem) for gem in gem_numbers]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        print("=== GEM Memory Dump Output ===")
        print(result.stdout)
        if result.stderr:
            print("=== Errors ===")
            print(result.stderr)
        dump_files = []
        for gem_num in gem_numbers:
            for pattern in [f"gem{gem_num}-dump", f"gem{gem_num}_regdump.bin"]:
                f = Path(".") / pattern
                if f.exists():
                    dump_files.append(f)
        dump_files = sorted(set(dump_files), key=lambda x: x.name)
        if dump_files:
            print(f"\n=== Moving dump files to {output_dir}/ ===")
            moved_count = 0
            for f in dump_files:
                if f.exists():
                    target = Path(output_dir) / f.name
                    try:
                        f.rename(target)
                        print(f"Moved: {f.name} -> {output_dir}/")
                        moved_count += 1
                    except Exception as e:
                        print(f"Error moving {f.name}: {e}")
            print(f"Successfully moved {moved_count} file(s)")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running hello program: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python dump_gem.py <gem_number>")
        print("Example: python dump_gem.py 1")
        return 1
    try:
        gem_number = int(sys.argv[1])
    except ValueError:
        print("Error: GEM number must be an integer.")
        return 1
    output_dir = "dump"
    print("RK3588 NPU GEM Memory Dumper")
    print("=" * 40)
    if not check_dependencies():
        return 1
    print(f"\nDumping GEM object {gem_number} to {output_dir}/...")
    gems_to_dump = [gem_number]
    print(f"Will dump GEM object: {gems_to_dump}")
    if dump_gem_memory(gems_to_dump, output_dir):
        print(f"\n=== GEM Memory Dump Complete ===")
        print(f"Dump files saved to: {output_dir}/")
    else:
        print(f"\n=== GEM Memory Dump Failed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
