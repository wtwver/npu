clear

gcc -o main main.c -ldrm -lm -I../include 
if [ $# -gt 0 ]; then
  gdb -q -x test.gdb ./main
else
  ./main
fi