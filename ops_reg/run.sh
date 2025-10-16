# gcc -o main main.c -ldrm -lm -I../include && gdb -x ops.gdb --args ./main 1

gcc -o main main.c -ldrm -lm -I../include && ./main 3
