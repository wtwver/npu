#!/usr/bin/env bash
set -e

# Always regenerate the model
python3 int32add.py

# Create and enter build directory
BUILD_DIR=build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure and build
cmake ..
make

# Run the executable with any passed arguments
echo "Running ./int32add ..."
# LD_LIBRARY_PATH="/root/.pyenv/versions/3.11.4/lib" 
#  LD_LIBRARY_PATH="/usr/lib/python3.10" LD_PRELOAD="/home/orangepi/add/rk3588/rknn-sniff/preload_python.so" 
 ./int32add "$@" 