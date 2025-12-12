#include <stdio.h>
#include "rknnops.h"

int main(int argc, char **argv) {

    int size = 1 ;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    int32_t* a = (int32_t*)malloc(size * sizeof(int32_t));
    int32_t* b = (int32_t*)malloc(size * sizeof(int32_t));
    printf("size: %d %d\n", sizeof(a), sizeof(b));

    for (size_t i = 0; i < size; i++) {
        // a[i] = 68.0f;
        a[i] = 30;
        // b[i] = 85.0f;
        b[i] = 15;
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
    int32_t* result = float16_alu_op(a, b, 7, size);
    printf("Input0: "); for (size_t i = 0; i < size; i++) printf("%d ", a[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < size; i++) printf("%d ", b[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < size; i++) printf("%d ", result[i]); printf("\n");

    // for (size_t i = 0; i < size; i++) {
        // b[i] = 2.0f;
    // }
    // result = float16_alu_op(result, b, 9, size);
    // printf("Input1: "); for (size_t i = 0; i < size; i++) printf("%d ", b[i]); printf("\n");
    // printf("Result: "); for (size_t i = 0; i < size; i++) printf("%d ", result[i]); printf("\n");

    return 0;
}