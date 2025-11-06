gcc -o main main.c -ldrm -lm -I../include && gdb -x ops.gdb --args ./main 1 | tee run_output.txt

# gcc -o main main.c -ldrm -lm -I../include && ./main 3
