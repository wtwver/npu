gcc -o main main.c -ldrm -lm -I../include
gdb -x ops.gdb --args ./main add 1x2
