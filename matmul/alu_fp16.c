#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include "rknnops.h"



int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <alu_algorithm>\n", argv[0]);
        return 1;
    }
    
    int alu_algorithm = atoi(argv[1]);
    const char* algo_names[] = {"Max", "Min", "Add", "Div", "Subtract", "Abs", "Neg", "Floor", "Ceil", "Unknown", "Unknown"};
    const char* algo_name = (alu_algorithm >= 0 && alu_algorithm <= 10) ? algo_names[alu_algorithm] : "Invalid";
    
    printf("Testing _Float16 operation with ALU Algorithm %d (%s)\n", alu_algorithm, algo_name);
    printf("================================================\n");

    // Get DRM device file descriptor
    int fd = getDeviceFd();
    if (fd < 0) {
        printf("Failed to get device file descriptor\n");
        return 1;
    }
    printf("Device fd: %d\n", fd);

    _Float16* a = (_Float16*)malloc(5 * sizeof(_Float16));
    _Float16* b = (_Float16*)malloc(5 * sizeof(_Float16));

    if (a == NULL || b == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize test data
    printf("sizeof(_Float16) = %zu\n", sizeof(_Float16));
    for (size_t i = 0; i < 5; i++) {
        a[i] = 18.0f;
        b[i] = 2.0f;
        printf("Setting a[%zu] = %.1f, b[%zu] = %.1f\n", i, (float)a[i], i, (float)b[i]);
    }

    // Display input values
    printf("Input A: ");
    for (size_t i = 0; i < 5; i++) {
        printf("%.1f ", (float)a[i]);
    }
    printf("\n");

    printf("Input B: ");
    for (size_t i = 0; i < 5; i++) {
        printf("%.1f ", (float)b[i]);
    }
    printf("\n");

    // Calculate expected results based on algorithm
    printf("Expected result (%s): ", algo_name);
    for (size_t i = 0; i < 5; i++) {
        int result;
        switch (alu_algorithm) {
            case 0:  // Max
                result = (a[i] > b[i]) ? a[i] : b[i];
                break;
            case 1:  // Min
                result = (a[i] < b[i]) ? a[i] : b[i];
                break;
            case 2:  // Add
                result = a[i] + b[i];
                break;
            case 3:  // Div
                result = (b[i] != 0) ? a[i] / b[i] : 0;
                break;
            case 4:  // Subtract
                result = a[i] - b[i];
                break;
            case 5:  // Abs (unary operation on a)
                result = (a[i] < 0) ? -a[i] : a[i];
                break;
            case 6:  // Neg (unary operation on a)
                result = -a[i];
                break;
            case 7:  // Floor (unary operation on a)
                result = a[i];
                break;
            case 8:  // Ceil (unary operation on a)
                result = a[i];
                break;
            default:  // 9-10 (unknown)
                result = 0;
                break;
        }
        printf("%d ", result);
    }
    printf("\n");

    printf("Calling NPU float16_alu_op...\n");
    _Float16* result = float16_alu_op(a, b, alu_algorithm);

    if (result != NULL) {
        printf("NPU results: ");
        for (size_t i = 0; i < 5; i++) {
            printf("%.1f ", (float)result[i]);
        }
        printf("\n");
    } else {
        printf("NPU operation failed\n");
    }

    free(a);
    free(b);

    return 0;
}