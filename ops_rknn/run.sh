# g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 add 1x1
# g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 relu 1x1
# g++ -o ops_float16_relu ops_float16_relu.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16_relu relu 1x2

# g++ -o conv1d_simple conv1d_simple.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./conv1d_simple | tee run_output.txt
# else
#    ./conv1d_simple | tee run_output.txt
# fi

# g++ -o conv2d_multi conv2d_multi.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./conv2d_multi | tee run_output.txt
# else
#    ./conv2d_multi | tee run_output.txt
# fi

# g++ -o matmul_multi matmul_multi.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./matmul_multi | tee run_output.txt
# else
#    ./matmul_multi | tee run_output.txt
# fi

# g++ -o matmul_api matmul_api.cpp -I../include -lrknnrt -std=c++11
# if [ $# > 0 ]; then
#    gdb -x matmul.gdb ./matmul_api | tee run_output.txt
# else
#    ./matmul_api | tee run_output.txt
# fi

# Ensure ReLU models exist for default shapes (1x2, 1x4, 1x4096)
# python3 generate_relu.py 25

# g++ -o relu relu.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./relu | tee run_output.txt
# else
#    ./relu | tee run_output.txt
# fi

# g++ -o sigmoid sigmoid.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./sigmoid | tee run_output.txt
# else
#    ./sigmoid | tee run_output.txt
# fi

# Ensure SiLU model exists for the 1x1x1x16 input used in silu.cpp
# if [ ! -f models/silu_float16_1x16.rknn ]; then
#   python3 generate_silu.py 16
# fi

# g++ -o silu silu.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./silu | tee run_output.txt
# else
#    ./silu | tee run_output.txt
# fi

# g++ -o activation activation.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    ./activation "$@" | tee run_output.txt
# else
#    # default: exercise all supported activations at width 64
#    : > run_output.txt
#    # ACTIVATIONS="relu leaky_relu celu selu silu swish softsign sigmoid logsigmoid hardsigmoid softplus gelu quick_gelu elu relu6 hardswish mish softmax log_softmax"
#    ACTIVATIONS="log_softmax"
#    for act in $ACTIVATIONS; do
#      echo "=== running $act ===" | tee -a run_output.txt
#      ./activation "$act" $WIDTH | tee -a run_output.txt
#    done
# fi

# g++ -o pool pool.cpp -I../include -lrknnrt -std=c++11
# if [ $# -gt 0 ]; then
#    gdb -x rknn.gdb ./pool | tee run_output.txt
# else
#    ./pool | tee run_output.txt
# fi

g++ -o div div.cpp -I../include -lrknnrt -std=c++11
if [ $# -gt 0 ]; then
   gdb -x rknn.gdb ./div | tee run_output.txt
else
   ./div | tee run_output.txt
fi
