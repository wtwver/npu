#include <stdio.h>
#include "rknnops.h"

static void unpack_nc1hwc2_fp16(const __fp16 *src, float *dst,
    int batch, int channels, int height, int width,
    int c2, int width_stride) {
  int c1 = (channels + c2 - 1) / c2;
  size_t plane_stride = (size_t)height * width_stride * c2;
  for (int n = 0; n < batch; n++) {
    for (int c = 0; c < channels; c++) {
      int plane = c / c2;
      int offset = c % c2;
      size_t src_plane_base = ((size_t)n * c1 + plane) * plane_stride;
      for (int h = 0; h < height; h++) {
        size_t src_row_base = src_plane_base + (size_t)h * width_stride * c2;
        size_t dst_row_base = ((((size_t)n * channels + c) * height) + h) * width;
        for (int w = 0; w < width; w++) {
          size_t src_idx = src_row_base + (size_t)w * c2 + offset;
          dst[dst_row_base + w] = (float)src[src_idx];
        }
      }
    }
  }
}

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
    int M = 32; int K = 32; int N = 32; 

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
    // CUSTOM 10: RELU
    // CUSTOM 11: MATMUL
    float* result = float16_matmul(a, b, 11, 32, 32, 32);
    printf("Input0: "); for (size_t i = 0; i < M*K ; i++) printf("%f ", a[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < N*K ; i++) printf("%f ", b[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < M*N ; i++) printf("fp16: %f fp32: %f \n", (__fp16)result[i], result[i]); printf("\n");
    return 0;
}

int test_conv(int argc, char **argv) {
    int input_size = 5;
    int kernel_size = 3;
    int output_size = input_size - kernel_size + 1;  // 5 - 3 + 1 = 3
    
    __fp16* input = (__fp16*)malloc(input_size * sizeof(__fp16));
    __fp16* kernel = (__fp16*)malloc(kernel_size * sizeof(__fp16));
    
    // Initialize input: [1, 2, 3, 4, 5]
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f; input[3] = 4.0f; input[4] = 5.0f;
    // Initialize kernel: [1, 0, -1]
    kernel[0] = 1.0f; kernel[1] = 0.0f; kernel[2] = -1.0f;
    
    // CUSTOM 12: CONV
    __fp16* result = float16_conv(input, kernel, 12, input_size, kernel_size);
    printf("Input0: "); for (size_t i = 0; i < input_size ; i++) printf("%f ", input[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < kernel_size ; i++) printf("%f ", kernel[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < output_size ; i++) printf("fp16: %f fp32: %f \n", (__fp16)result[i], result[i]); printf("\n");
    return 0;
    
    // Expected output should be: [-2.0, -2.0, -2.0]
    // This matches conv1d_simple.cpp calculation
    
    free(input);
    free(kernel);
    return 0;
}

int test_conv2d(int argc, char **argv) {
  const int batch = 1;
  const int in_channels = 3;
  const int in_height = 5;
  const int in_width = 7;
  const int out_channels = 6;
  const int kernel_h = 2;
  const int kernel_w = 3;
  const int out_height = in_height - kernel_h + 1;
  const int out_width = in_width - kernel_w + 1;
  const int align_c = 8;
  const int align_out_c = 8;
  const int width_stride = 8;
  const int out_width_stride = 5;

  size_t input_elems = (size_t)batch * in_channels * in_height * in_width;
  size_t weight_elems = (size_t)out_channels * in_channels * kernel_h * kernel_w;
  size_t packed_input_elems =
      (size_t)batch * ((in_channels + align_c - 1) / align_c) * in_height * width_stride * align_c;
  __fp16 *input = (__fp16*)malloc(input_elems * sizeof(__fp16));
  __fp16 *kernel = (__fp16*)malloc(weight_elems * sizeof(__fp16));
  __fp16 *input_packed = (__fp16*)malloc(packed_input_elems * sizeof(__fp16));

  if (!input || !kernel || !input_packed) {
    printf("failed to allocate conv2d buffers\n");
    free(input);
    free(kernel);
    free(input_packed);
    return -1;
  }

  size_t idx = 0;
  for (int n = 0; n < batch; n++) {
    for (int c = 0; c < in_channels; c++) {
      for (int h = 0; h < in_height; h++) {
        float hv = (float)((h + 1) * (h + 1));
        for (int w = 0; w < in_width; w++) {
          float wv = (float)((w + 1) * (w + 1));
          float base = hv + 0.7f * wv;
          float ch = 0.3f * (c + 1);
          float nb = 0.1f * (n + 1);
          input[idx++] = (__fp16)(base * ch + nb);
        }
      }
    }
  }

  idx = 0;
  for (int oc = 0; oc < out_channels; oc++) {
    for (int ic = 0; ic < in_channels; ic++) {
      for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
          int pattern = (oc + kh + kw) % 3;
          float val = (pattern == 0) ? 1.0f : ((pattern == 1) ? -1.0f : 0.0f);
          kernel[idx++] = (__fp16)val;
        }
      }
    }
  }

  for (size_t i = 0; i < packed_input_elems; i++) {
    input_packed[i] = (__fp16)0;
  }
  pack_nc1hwc2_fp16(input_packed, input,
      batch, in_channels, in_height, in_width, align_c, width_stride);

  size_t expected_elems = (size_t)out_channels * out_height * out_width;
  float *expected = (float*)malloc(expected_elems * sizeof(float));
  if (!expected) {
    printf("failed to allocate expected buffer\n");
    free(input);
    free(kernel);
    free(input_packed);
    return -1;
  }
  for (size_t i = 0; i < expected_elems; i++) expected[i] = 0.0f;

  for (int oc = 0; oc < out_channels; oc++) {
    for (int oh = 0; oh < out_height; oh++) {
      for (int ow = 0; ow < out_width; ow++) {
        float acc = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
          for (int kh = 0; kh < kernel_h; kh++) {
            int ih = oh + kh;
            for (int kw = 0; kw < kernel_w; kw++) {
              int iw = ow + kw;
              size_t in_idx = (((size_t)ic * in_height) + ih) * in_width + iw;
              size_t wt_idx = ((((size_t)oc * in_channels) + ic) * kernel_h + kh) * kernel_w + kw;
              acc += (float)kernel[wt_idx] * (float)input[in_idx];
            }
          }
        }
        expected[(size_t)oc * out_height * out_width + oh * out_width + ow] = acc;
      }
    }
  }

  __fp16 *result = float16_conv2d(input, kernel, 13, (int)input_elems, (int)weight_elems);
  float *output_nchw = (float*)malloc((size_t)out_channels * out_height * out_width * sizeof(float));
  if (!output_nchw) {
    printf("failed to allocate output buffer\n");
    free(input);
    free(kernel);
    free(input_packed);
    free(expected);
    return -1;
  }
  unpack_nc1hwc2_fp16(result, output_nchw,
      batch, out_channels, out_height, out_width, align_out_c, out_width_stride);

  printf("Input tensor (NCHW)\n");
  for (int n = 0; n < batch; n++) {
    printf("  n=%d\n", n);
    for (int c = 0; c < in_channels; c++) {
      printf("    c=%d\n", c);
      for (int h = 0; h < in_height; h++) {
        printf("      h=%d: ", h);
        size_t row_base = (((size_t)n * in_channels + c) * in_height + h) * in_width;
        for (int w = 0; w < in_width; w++) {
          printf("%.6f ", (float)input[row_base + w]);
        }
        printf("\n");
      }
    }
  }

  printf("Packed input (first 16 elements): ");
  for (size_t i = 0; i < 16 && i < packed_input_elems; i++) {
    printf("%f ", input_packed[i]);
  }
  printf("\n");

  printf("Kernel (first 16 elements): ");
  for (size_t i = 0; i < 16 && i < weight_elems; i++) {
    printf("%f ", kernel[i]);
  }
  printf("\n");

  printf("Expected Output (CPU computed):\n");
  for (int oc = 0; oc < out_channels; oc++) {
    printf("  Output Channel %d:\n", oc);
    for (int oh = 0; oh < out_height; oh++) {
      printf("    ");
      for (int ow = 0; ow < out_width; ow++) {
        float val = expected[(size_t)oc * out_height * out_width + oh * out_width + ow];
        printf("%.5f ", val);
      }
      printf("\n");
    }
  }

  printf("\nActual Output (RKNN):\n");
  for (int oc = 0; oc < out_channels; oc++) {
    printf("  Output Channel %d:\n", oc);
    for (int oh = 0; oh < out_height; oh++) {
      printf("    ");
      for (int ow = 0; ow < out_width; ow++) {
        size_t idx_out = (size_t)oc * out_height * out_width + oh * out_width + ow;
        printf("%.5f ", output_nchw[idx_out]);
      }
      printf("\n");
    }
  }

  free(input);
  free(kernel);
  free(input_packed);
  free(expected);
  free(output_nchw);
  return 0;
}

int main(int argc, char **argv) {
    int fd = getDeviceFd();
    npu_reset(fd);

    // test_alu(argc, argv);
    // test_matmul(argc, argv);
    // test_conv(argc, argv);
    test_conv2d(argc, argv);
    return 0;
}
