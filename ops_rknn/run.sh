# g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 add 1x1
# g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 relu 1x1
# g++ -o ops_float16_relu ops_float16_relu.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16_relu relu 1x2

g++ -o conv1d_simple conv1d_simple.cpp -I../include -lrknnrt -std=c++11
if [ $# -gt 0 ]; then
   gdb -x rknn.gdb ./conv1d_simple | tee run_output.txt
else
   ./conv1d_simple | tee run_output.txt
fi

# g++ -o conv2d_multi conv2d_multi.cpp -I../include -lrknnrt -std=c++11 &&  gdb -x rknn.gdb ./conv2d_multi | tee run_output.txt
# conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 2, 1)
# conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 2, 3)
# conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 2, 5)
# conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 3, 1)
# conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 3, 3)
# conv2d input shape (1, 3, 5, 7), weight shape (6, 1, 3, 3)
# conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 3, 5)