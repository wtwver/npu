# How 

How to convert onnx to rknn
- python3 -m rknn.api.rknn_convert -t rk3588 -i /home/orangepi/npu/models/8_add.onnx -o /home/orangepi/npu/models/

How to build matmul / alu
- cd ops_reg/matmul/ && gcc -o matmul matmul.c -I ../../include -ldrm && ./matmul 32 32 32

# Ref
https://github.com/liej6799/rk3588


# matmul 
        K columns
      ┌───────────────┐
m=0   │ a00 a01 ... a0K│
m=1   │ a10 a11 ... a1K│
m=2   │ a20 a21 ... a2K│
      │  .   .       .│
m=M-1 │ aM0 aM1 ... aMK│
      └───────────────┘
         ↑ rows = M

          N columns
        ┌─────────────────┐
k=0     │ b00 b01 ... b0N  │
k=1     │ b10 b11 ... b1N  │
k=2     │ b20 b21 ... b2N  │
        │  .   .       .  │
k=K-1   │ bK0 bK1 ... bKN  │
        └─────────────────┘
           ↑ rows = K

        ┌─────────────────┐
m=0     │ c00 c01 ... c0N  │
m=1     │ c10 c11 ... c1N  │
        │  .   .       .  │
m=M-1   │ cM0 cM1 ... cMN  │
        └─────────────────┘

c[m][n] = Σ_k A[m][k] * B[k][n]

# CNA View

## Feature Map (A)
H = M
W = 1
C = K

H (rows)
┌──────────────────────────────┐
│ [a00 a01 ... a0K]  ← C=K     │  h=0
│ [a10 a11 ... a1K]            │  h=1
│ [a20 a21 ... a2K]            │  h=2
│  .                           │
│ [aM0 aM1 ... aMK]            │  h=M-1
└──────────────────────────────┘
           W = 1

## Weights (B)
Kernel size: 1 × 1
C = K
Number of kernels = N

Kernel 0 (n=0):  [b00 b10 b20 ... bK0]
Kernel 1 (n=1):  [b01 b11 b21 ... bK1]
Kernel 2 (n=2):  [b02 b12 b22 ... bK2]
...
Kernel N-1:      [b0N b1N b2N ... bKN]

      Kernel 0      Kernel 1       Kernel N-1
     ┌─────────┐  ┌─────────┐    ┌─────────┐
     │ b00     │  │ b01     │    │ b0N     │
     │ b10     │  │ b11     │    │ b1N     │
C=K  │ b20     │  │ b21     │    │ b2N     │
     │  .      │  │  .      │    │  .      │
     │ bK0     │  │ bK1     │    │ bKN     │
     └─────────┘  └─────────┘    └─────────┘
       1×1           1×1             1×1

## CNA Computation (1×1 convolution)

Input vector:  A[h, :]
Kernel n:      B[:, n]
out[h][n] = Σ_k A[h][k] * B[k][n]

At h = 2:

Feature vector:
[a20 a21 a22 ... a2K]

Kernel 3:
[b03 b13 b23 ... bK3]

Dot:
a20*b03 + a21*b13 + ... + a2K*bK3

## CNA Output Feature Map (C)
H = M
W = 1
C = N

Output Feature Map (C):

┌──────────────────────────────┐
│ [c00 c01 ... c0N]            │ h=0
│ [c10 c11 ... c1N]            │ h=1
│ [c20 c21 ... c2N]            │ h=2
│  .                           │
│ [cM0 cM1 ... cMN]            │ h=M-1
└──────────────────────────────┘