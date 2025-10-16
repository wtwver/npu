#include <stdio.h>
#include "rknnops.h"

int test_alu(int argc, char **argv) {
    int size = 1 ;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    __fp16* a = (__fp16*)malloc(size * sizeof(__fp16));
    __fp16* b = (__fp16*)malloc(size * sizeof(__fp16));
    // printf("size: %d %d\n", sizeof(a), sizeof(b));

    for (size_t i = 0; i < size; i++) {
        a[i] = 2.3f;
        b[i] = 2.1f;
    }
    // 4'd0: Max;
    // 4'd1: Min;
    // 4'd2: Add;
    // 4'd3: Div; # overflow issue
    // 4'd4: Minus;
    // CUSTOM 9: MUL
    // __fp16* result = float16_alu_op(a, b, 2, size);
    float* result = (float*)float16_alu_op(a, b, 2, size);
    
    printf("Input0: "); for (size_t i = 0; i < size; i++) printf("%f ", a[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < size; i++) printf("%f ", b[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < size; i++) printf("fp16: %f fp32: %f \n", (__fp16)result[i], result[i]); printf("\n");
    
    free(a);
    free(b);
    return 0;
}

int test_matmul(int argc, char **argv) {
    // if (argc > 1) {
        // size = atoi(argv[1]);
    // }
    int M = 32; int K = 32; int N=32; 

    __fp16* a = (__fp16*)malloc( M*K * sizeof(__fp16));
    __fp16* b = (__fp16*)malloc( N*K * sizeof(__fp16));

    for (size_t i = 0; i < M*K ; i++) {a[i] = (int)2.0f;}
    for (size_t i = 0; i < N*K ; i++) {b[i] = (int)3.0f;}
    // 4'd0: Max;
    // 4'd1: Min;
    // 4'd2: Add;
    // 4'd3: Div; # overflow issue
    // 4'd4: Minus;
    // CUSTOM 9: MUL
    // CUSTOM 10: RELUls
    // CUSTOM 11: MATMUL
    float* result = float16_matmul(a, b, 11, 32, 32, 32);
    printf("Input0: "); for (size_t i = 0; i < M*K ; i++) printf("%f ", a[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < N*K ; i++) printf("%f ", b[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < M*N ; i++) printf("%f ", result[i]); printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    int fd = getDeviceFd();
    npu_reset(fd);

    test_alu(argc, argv);
    // test_matmul(argc, argv);
    return 0;
}