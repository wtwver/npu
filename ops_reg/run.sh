# REGENERATE_CONV1D=${REGENERATE_CONV1D:-0}
# CONV1D_DATA_DIR=${CONV1D_DATA_DIR:-../ops_rknn/conv1d_simple_data}
# GDB_SCRIPT=${GDB_SCRIPT:-reg.gdb}
# GDB_FLAGS=${GDB_FLAGS:-}

# if [ "$REGENERATE_CONV1D" != "0" ]; then
#   python3 ../ops_rknn/generate_conv1d_inputs.py --case conv1d_simple_bs8
# fi

# gcc -o main main.c -ldrm -lm -I../include && gdb $GDB_FLAGS -x "$GDB_SCRIPT" --args ./main --data-dir "$CONV1D_DATA_DIR" 1

# gcc -o main main.c -ldrm -lm -I../include && CONV1D_DATA_DIR="$CONV1D_DATA_DIR" ./main
gcc -o main main.c -ldrm -lm -I../include && gdb -q -x test.gdb ./main
