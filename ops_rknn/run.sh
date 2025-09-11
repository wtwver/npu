g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11 && gdb -x ops.gdb --args ./ops_float16 add 1x3
