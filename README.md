# How to do matmul in RK3588

## Input and output dimension 

In matmul, we have C(M,N) = A(M,K) * B(K,N)
c[m][n] = Σₖ A[m][k] · B[k][n]
                                                <----------- N columns ----------->
                                          ^   ┌────────────────────────────────────────┐
                                          |   │ b[0][0] b[0][1] ... b[0][N-1]          │
                                          |   │ b[1][0] b[1][1] ... b[1][N-1]          │
                                   K rows |   │ b[2][0] b[2][1] ... b[2][N-1]          │
                                          |   │     .        .        .                │
                                          |   │ b[K-1][0] b[K-1][1] ... b[K-1][N-1]    │
                                          V   └────────────────────────────────────────┘
        <----------- K columns ----------->
^      ┌────────────────────────────────────┐ ┌────────────────────────────────────────┐      
|      │ a[0][0] a[0][1] ... a[0][K-1]      │ │ c[0][0] c[0][1] ... c[0][N-1]          │
|      │ a[1][0] a[1][1] ... a[1][K-1]      │ │ c[1][0] c[1][1] ... c[1][N-1]          │
M rows │ a[2][0] a[2][1] ... a[2][K-1]      │ │ c[2][0] c[2][1] ... c[2][N-1]          │
|      │     .        .        .            │ │     .        .        .                │
|      │ a[M-1][0] a[M-1][1] ... a[M-1][K-1]│ | c[M-1][0] c[M-1][1] ... c[M-1][N-1]    │
V      └────────────────────────────────────┘ └────────────────────────────────────────┘

As RK3588 is a NVDLA-based NPU which designed for convulution, we can use a special convulution config to do matmul.
For normal 2d conv, 

        Feature Data (H,W=1)   Kernel(R=1,S=1)
        ┌────────────┐         ┌─────┐
        │ a[0][0]    │         │  k  │
        │ a[2][0]    │         └─────┘
        │   ...      │
        │ a[H-1][0]  │
        └────────────┘

We need 3d conv to do matmul, and N kernels
                                                                       Kernel (N=N, R=1, S=1, C=K)
        Feature Map A  (H=M, W=1, C=K)                                  kernel 0
                  +───────── front-face channels C ─────────────+.         +───────── front-face channels C ──────+
        W=1      /                                             /|     S=1 /                                      /|                    
                +─────────────────────────────────────────────+ |        +──────────────────────────────────────+ |
        ^       | a[0][0][0]   a[0][0][1]   ... a[0][0][N-1]  | |   R=1  | b[0][0][0]   b[0][1]   ... b[0][K-1] | |
        |       | a[1][0][0]   a[1][0][1]   ... a[1][0][N-1]  | |        +──────────────────────────────────────+/
        |       | a[2][0][0]   a[2][0][1]   ... a[2][0][N-1]  | |       kernel 1
        H rows  |        .        .           .               | |          +───────── front-face channels C ──────+
        |       | a[H-1][0][0] a[H-1][1] ... a[H-1][0][N-1]   | /     S=1 /                                      /|
        V       +─────────────────────────────────────────────+/        +──────────────────────────────────────+ |
                                                                    R=1 | b[0][0][0]   b[0][1]   ... b[0][K-1] | |
                                                                        +──────────────────────────────────────+/
                                                                        .
                                                                        .

                                                                        kernel N-1
                                                                          +───────── front-face channels C ──────+
                                                                     S=1 /                                      /|
                                                                        +──────────────────────────────────────+ |
                                                                    R=1 | b[0][0][0]   b[0][1]   ... b[0][K-1] | |
                                                                        +──────────────────────────────────────+/

Input vector:  A[h, :]
Kernel n:      B[:, n]
out[h][n] = Σ_k A[h][k] * B[k][n]

        Output Feature Map C  (H=M, W=1, C=N)                                  
                  +───────── front-face channels C=N ───────────+.  
        W=1      /                                             /|                
                +─────────────────────────────────────────────+ |    
        ^       | a[0][0][0]   a[0][0][1]   ... a[0][0][N-1]  | |  
        |       | a[1][0][0]   a[1][0][1]   ... a[1][0][N-1]  | |   
        |       | a[2][0][0]   a[2][0][1]   ... a[2][0][N-1]  | |     
        M rows  |        .        .           .               | |   
        |       | a[M-1][0][0] a[M-1][1] ... a[M-1][0][N-1]   | /    
        V       +─────────────────────────────────────────────+/     

## CNA registers

so now we have

Feature Map A  (H=M, W=1, C=K) 
```
datain_height = M
datain_width = 1
datain_channel = K
```

Kernel (N=N, R=1, S=1, C=K)
```
weight_width = 1
weight_height = 1
```

# FAQ 
How to convert onnx to rknn
- python3 -m rknn.api.rknn_convert -t rk3588 -i /home/orangepi/npu/models/8_add.onnx -o /home/orangepi/npu/models/

How to build matmul / alu
- cd ops_reg/matmul/ && gcc -o matmul matmul.c -I ../../include -ldrm && ./matmul 32 32 32


# Reference
https://github.com/nvdla/sw
https://github.com/ONNC/onnc
https://github.com/mtx512/rk3588-npu
https://github.com/liej6799/rk3588