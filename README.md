# How 

How to convert onnx to rknn
- python3 -m rknn.api.rknn_convert -t rk3588 -i /home/orangepi/npu/models/8_add.onnx -o /home/orangepi/npu/models/

How to build matmul / alu
- cd ops_reg/matmul/ && gcc -o matmul matmul.c -I ../../include -ldrm && ./matmul 32 32 32

# Ref
https://github.com/liej6799/rk3588
