if [ "$1" = "long" ]; then
    gcc -o matmul_long matmul_long.c -I ../../include/ && ./matmul_long 32 32 32 fp16
else
    gcc -o matmul matmul.c -I ../../include/ && ./matmul 32 32 32 fp16
fi

# gcc -o matmul_long matmul_long.c -I ../../include/ && ./matmul_long 32 32 32 fp16 