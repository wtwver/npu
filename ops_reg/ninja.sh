#!/bin/sh
set -eu

clear
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

if [ ! -d build ]; then
  meson setup build
fi

ninja -C build
if [ $# -gt 0 ]; then
  gdb -q -x test.gdb ./build/ops_reg
else
  ninja -C build test
fi
