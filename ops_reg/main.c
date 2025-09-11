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
    __fp16* result = float16_alu_op(a, b, 2, size);
    printf("Input0: ");
    for (size_t i = 0; i < size; i++) {
        printf("%f ", a[i]);
    };

    printf("\nInput1: ");
    for (size_t i = 0; i < size; i++) {
        printf("%f ", b[i]);
    };

    printf("\nResult: ");
    for (size_t i = 0; i < size; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    // int16_t* a1 = (int16_t*)malloc(10 * sizeof(int16_t));
    // int16_t* b1 = (int16_t*)malloc(10 * sizeof(int16_t));
    // for (size_t i = 0; i < 10; i++) {
    //     a1[i] = 11;
    //     b1[i] = 12;
    // }
    // int16_t* result1 = int16_add_op(a1, b1);
    // for (size_t i = 0; i < 10; i++) {
    //     printf("%d ", result1[i]);
    // }

    // int8_t* a1 = (int8_t*)malloc(10 * sizeof(int8_t));
    // int8_t* b1 = (int8_t*)malloc(10 * sizeof(int8_t));
    // for (size_t i = 0; i < 10; i++) {
    //     a1[i] = 18;
    //     b1[i] = 12;
    // }
    // int8_t* result1 = int8_add_op(a1, b1);
    // for (size_t i = 0; i < 10; i++) {
    //     printf("%d ", result1[i]);
    // }
    return 0;
}