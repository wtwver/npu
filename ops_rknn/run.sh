# g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 add 1x1
# g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 relu 1x1
# g++ -o ops_float16_relu ops_float16_relu.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16_relu relu 1x2


# g++ -o conv1d_simple conv1d_simple.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb ./conv1d_simple 
g++ -o conv2d_simple conv2d_simple.cpp -I../include -lrknnrt -std=c++11 &&  gdb -x ops.gdb ./conv2d_simple 
