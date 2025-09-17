gcc -o main main.c -ldrm -lm -I../include && ./main 1
# gcc -o main_int16 main_int16.c -ldrm -lm -I../include && gdb -x ops.gdb --args ./main_int16 1
