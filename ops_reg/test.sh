#!/bin/sh
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

gcc -o main main.c -ldrm -lm -I../include

while IFS= read -r test_entry; do
  case "$test_entry" in
    ""|\#*) continue ;;
  esac
    echo "=== ${test_entry} ==="
    set +e
    ./main "${test_entry}" | tee /tmp/test_ops_reg.log

    if cat /tmp/test_ops_reg.log | rg "matches CPU -> NO|timed out|failed"; then
      echo "Stopping: ${test_entry} reported CPU mismatch or timeout"
      exit 1
    fi
done <<'EOF'
conv1d
conv2d
matmul
add
minus
mul
div
abs
neg
max
cmple
cmpgt
cmpge
cmplt
cmpeq
cmpneq
roundoff
where
sigmoid
relu
# silu
EOF
