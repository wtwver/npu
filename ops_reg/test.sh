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

    if ! rg -q "matches CPU -> YES" /tmp/test_ops_reg.log; then
      echo "Stopping: ${test_entry} did not report matches CPU -> YES"
      exit 1
      if rg -q "matches CPU -> NO|timed out|failed|not found" /tmp/test_ops_reg.log; then
        echo "Stopping: ${test_entry} did not report matches CPU -> YES"
        exit 1
      fi
    fi
done <<'EOF'
conv1d
conv2d
matmul
add
minus
mul
div
idiv
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
rounddown
where
sigmoid
relu
# maxpool
silu
EOF
