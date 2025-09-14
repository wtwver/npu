#include <stdio.h>
#include "rknnops.h"

int main(int argc, char **argv) {

    int size = 1 ;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    __fp16* a = (__fp16*)malloc(size * sizeof(__fp16));
    __fp16* b = (__fp16*)malloc(size * sizeof(__fp16));
    printf("size: %d %d\n", sizeof(a), sizeof(b));

    for (size_t i = 0; i < size; i++) {
        a[i] = 68.0f;
        b[i] = 85.0f;
    }
    // 4'd0: Max;
    // 4'd1: Min;
    // 4'd2: Add;
    // 4'd3: Div;
    // 4'd4: Minus;
    // 4'd5: Abs;
    // 4'd6: Neg;
    // 4'd7: Floor;
    // 4'd8: Ceil.
    // CUSTOM 9: MUL
    __fp16* result = float16_alu_op(a, b, 2, size);
    printf("Input0: "); for (size_t i = 0; i < size; i++) printf("%f ", a[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < size; i++) printf("%f ", b[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < size; i++) printf("%f ", result[i]); printf("\n");

    for (size_t i = 0; i < size; i++) {
        b[i] = 2.0f;
    }
    result = float16_alu_op(result, b, 9, size);
    printf("Input1: "); for (size_t i = 0; i < size; i++) printf("%f ", b[i]); printf("\n");
    printf("Result: "); for (size_t i = 0; i < size; i++) printf("%f ", result[i]); printf("\n");

    return 0;
}