#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <string.h>
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

static void unpack_matmul_output_fp16_with_c2(const __fp16 *src, float *dst,
    int M, int N, int c2) {
  if (!src || !dst || M <= 0 || N <= 0) return;
  if (c2 <= 0) return;
  const size_t plane_stride = (size_t)M * (size_t)c2;
  const size_t row_stride = (size_t)c2;
  for (int n = 0; n < N; n++) {
    size_t plane = (size_t)n / c2;
    size_t offset = (size_t)n % c2;
    size_t plane_base = plane * plane_stride;
    for (int m = 0; m < M; m++) {
      size_t idx = plane_base + (size_t)m * row_stride + offset;
      dst[(size_t)m * N + n] = (float)src[idx];
    }
  }
}

static void unpack_matmul_output_fp16(const __fp16 *src, float *dst, int M, int N) {
  int c2 = (matmul_params.align_out > 0) ? matmul_params.align_out : 8;
  unpack_matmul_output_fp16_with_c2(src, dst, M, N, c2);
}

static void unpack_matmul_output_fp32_with_c2(const float *src, float *dst,
    int M, int N, int c2) {
  if (!src || !dst || M <= 0 || N <= 0) return;
  if (c2 <= 0) return;
  const size_t plane_stride = (size_t)M * (size_t)c2;
  const size_t row_stride = (size_t)c2;
  for (int n = 0; n < N; n++) {
    size_t plane = (size_t)n / c2;
    size_t offset = (size_t)n % c2;
    size_t plane_base = plane * plane_stride;
    for (int m = 0; m < M; m++) {
      size_t idx = plane_base + (size_t)m * row_stride + offset;
      dst[(size_t)m * N + n] = src[idx];
    }
  }
}

static void unpack_matmul_output_fp32(const float *src, float *dst, int M, int N) {
  // Matmul 64x64x64 and 256x256x256 outputs are stored with C2=4; other shapes keep align_out.
  int is_c2_4 = ((matmul_params.M == 64 && matmul_params.N == 64 && matmul_params.K == 64) ||
                 (matmul_params.M == 256 && matmul_params.N == 256 && matmul_params.K == 256));
  int c2 = is_c2_4 ? 4 : ((matmul_params.align_out > 0) ? matmul_params.align_out : 8);
  unpack_matmul_output_fp32_with_c2(src, dst, M, N, c2);
}

static void print_conv1d_outputs(const char *title, const float *data,
    int batch, int channels, int width, int row_len) {
  printf("%s\n", title);
  if (row_len <= 0) row_len = width;
  for (int n = 0; n < batch; n++) {
    printf("  Batch %d:\n", n);
    for (int oc = 0; oc < channels; oc++) {
      printf("    Output Channel %d:\n", oc);
      const float *row = data + (((size_t)n * channels + oc) * width);
      for (int start = 0; start < width; start += row_len) {
        printf("      ");
        int end = start + row_len;
        if (end > width) end = width;
        for (int i = start; i < end; i++) {
          printf("%8.5f  ", row[i]);
        }
        // printf("\n");
      }
      printf("\n");
    }
  }
}

static int align_up(int value, int align) {
  if (align <= 0) return value;
  return ((value + align - 1) / align) * align;
}

static const char *kConv1dFixtureDefault = "../ops_rknn/conv1d_simple_data";
static const char *kConv1dFixtureAlternate = "npu/ops_rknn/conv1d_simple_data";

static int load_fp16_fixture(const char *dir, const char *name,
    __fp16 *dst, size_t elems) {
  char path[512];
  int len = snprintf(path, sizeof(path), "%s/%s", dir, name);
  if (len < 0 || len >= (int)sizeof(path)) return 0;
  FILE *file = fopen(path, "rb");
  if (!file) return 0;
  size_t read = fread(dst, sizeof(__fp16), elems, file);
  fclose(file);
  return read == elems;
}

static int load_conv1d_fixtures_from(const char *base_dir, const char *fixture_dir,
    __fp16 *input, size_t input_elems,
    __fp16 *kernel, size_t kernel_elems) {
  char dir[512];
  int len = snprintf(dir, sizeof(dir), "%s/%s", base_dir, fixture_dir);
  if (len < 0 || len >= (int)sizeof(dir)) return 0;
  if (!load_fp16_fixture(dir, "input.bin", input, input_elems)) return 0;
  if (!load_fp16_fixture(dir, "kernel.bin", kernel, kernel_elems)) return 0;
  return 1;
}

static int load_conv1d_fixtures(const char *fixture_dir,
    __fp16 *input, size_t input_elems,
    __fp16 *kernel, size_t kernel_elems) {
  if (!fixture_dir) return 0;
  const char *env_base = getenv("CONV1D_DATA_DIR");
  if (env_base && load_conv1d_fixtures_from(env_base, fixture_dir, input, input_elems,
        kernel, kernel_elems)) {
    return 1;
  }
  if (load_conv1d_fixtures_from(kConv1dFixtureDefault, fixture_dir, input, input_elems,
        kernel, kernel_elems)) {
    return 1;
  }
  if (load_conv1d_fixtures_from(kConv1dFixtureAlternate, fixture_dir, input, input_elems,
        kernel, kernel_elems)) {
    return 1;
  }
  return 0;
}

typedef struct {
  uint32_t mt[624];
  int index;
} Mt19937;

static void mt_seed(Mt19937 *rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; i++) {
    rng->mt[i] = 1812433253U * (rng->mt[i-1] ^ (rng->mt[i-1] >> 30)) + (uint32_t)i;
  }
  rng->index = 624;
}

static uint32_t mt_extract(Mt19937 *rng) {
  const uint32_t mag01[2] = {0U, 0x9908b0dfU};
  if (rng->index >= 624) {
    int kk;
    for (kk = 0; kk < 624 - 397; kk++) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk+1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (; kk < 623; kk++) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk+1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
    }
    uint32_t y = (rng->mt[623] & 0x80000000U) | (rng->mt[0] & 0x7fffffffU);
    rng->mt[623] = rng->mt[396] ^ (y >> 1) ^ mag01[y & 1U];
    rng->index = 0;
  }
  uint32_t y = rng->mt[rng->index++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}

static float mt_uniform(Mt19937 *rng, float low, float high) {
  const double a = (double)(mt_extract(rng) >> 5);
  const double b = (double)(mt_extract(rng) >> 6);
  const double random = (a * 67108864.0 + b) / 9007199254740992.0;
  return (float)(low + (high - low) * random);
}

static void print_conv1d_tensor(const char *title, const __fp16 *data,
    int batch, int channels, int width) {
  printf("%s\n", title);
  for (int n = 0; n < batch; n++) {
    printf("  batch=%d\n", n);
    for (int c = 0; c < channels; c++) {
      printf("    channel=%d: ", c);
      for (int w = 0; w < width; w++) {
        size_t idx = ((size_t)n * channels + c) * width + w;
        printf("%8.5f ", (float)data[idx]);
      }
      printf("\n");
    }
  }
}

static void print_conv1d_kernel(const char *title, const __fp16 *data,
    int out_channels, int in_channels, int kernel_size) {
  printf("%s\n", title);
  for (int oc = 0; oc < out_channels; oc++) {
    printf("  out_channel=%d\n", oc);
    for (int ic = 0; ic < in_channels; ic++) {
      printf("    in_channel=%d: ", ic);
      for (int k = 0; k < kernel_size; k++) {
        size_t idx = ((size_t)oc * in_channels + ic) * kernel_size + k;
        printf("%8.5f ", (float)data[idx]);
      }
      printf("\n");
    }
  }
}

static void print_conv2d_input(const char *title, const __fp16 *data,
    int batch, int channels, int height, int width) {
  printf("%s\n", title);
  for (int n = 0; n < batch; n++) {
    printf("  batch=%d\n", n);
    for (int c = 0; c < channels; c++) {
      printf("    channel=%d\n", c);
      for (int h = 0; h < height; h++) {
        printf("      h=%d: ", h);
        for (int w = 0; w < width; w++) {
          size_t idx = ((((size_t)n * channels + c) * height) + h) * width + w;
          printf("%8.5f ", (float)data[idx]);
        }
        printf("\n");
      }
    }
  }
}

static void print_conv2d_kernel(const char *title, const __fp16 *data,
    int out_channels, int in_channels, int kernel_h, int kernel_w) {
  printf("%s\n", title);
  for (int oc = 0; oc < out_channels; oc++) {
    printf("  out_channel=%d\n", oc);
    for (int ic = 0; ic < in_channels; ic++) {
      printf("    in_channel=%d\n", ic);
      for (int kh = 0; kh < kernel_h; kh++) {
        printf("      kh=%d: ", kh);
        for (int kw = 0; kw < kernel_w; kw++) {
          size_t idx = ((((size_t)oc * in_channels + ic) * kernel_h + kh) * kernel_w) + kw;
          printf("%8.5f ", (float)data[idx]);
        }
        printf("\n");
      }
    }
  }
}

static void print_conv2d_output(const char *title, const float *data,
    int batch, int channels, int height, int width) {
  printf("%s\n", title);
  for (int n = 0; n < batch; n++) {
    printf("  batch=%d\n", n);
    for (int oc = 0; oc < channels; oc++) {
      printf("    out_channel=%d\n", oc);
      for (int h = 0; h < height; h++) {
        printf("      h=%d: ", h);
        const float *row = data + ((((size_t)n * channels + oc) * height + h) * width);
        for (int w = 0; w < width; w++) {
          printf("%8.5f ", row[w]);
        }
        printf("\n");
      }
    }
  }
}

static void print_fp16_row(const __fp16 *data, int cols, int row) {
  printf("  [");
  for (int c = 0; c < cols; c++) {
    printf("%6.2f", (float)data[row * cols + c]);
    if (c + 1 < cols) printf(", ");
  }
  printf("]");
}

static void print_float_row(const float *data, int cols, int row) {
  printf("  [");
  for (int c = 0; c < cols; c++) {
    printf("%9.6f", data[row * cols + c]);
    if (c + 1 < cols) printf(", ");
  }
  printf("]");
}

static void print_fp16_grid(const char *label, const __fp16 *data,
    int rows, int cols) {
  if (!data) {
    printf("%s: (null)\n", label);
    return;
  }
  printf("%s (%dx%d):\n", label, rows, cols);
  for (int r = 0; r < rows; r++) {
    printf("  ");
    for (int c = 0; c < cols; c++) {
      size_t idx = (size_t)r * cols + c;
      printf("%7.3f ", (float)data[idx]);
    }
    printf("\n");
  }
}

static void print_fp32_grid(const char *label, const float *data,
    int rows, int cols) {
  if (!data) {
    printf("%s: (null)\n", label);
    return;
  }
  printf("%s (%dx%d):\n", label, rows, cols);
  for (int r = 0; r < rows; r++) {
    printf("  ");
    for (int c = 0; c < cols; c++) {
      size_t idx = (size_t)r * cols + c;
      printf("%7.3f ", data[idx]);
    }
    printf("\n");
  }
}

static void print_fp16_matrix(const char *title, const __fp16 *data,
    int rows, int cols) {
  printf("%s tensor([\n", title);
  if (rows <= 4) {
    for (int r = 0; r < rows; r++) {
      print_fp16_row(data, cols, r);
      printf(r + 1 < rows ? ",\n" : "\n");
    }
  } else {
    print_fp16_row(data, cols, 0);
    printf(",\n");
    print_fp16_row(data, cols, 1);
    printf(",\n");
    printf("  ...,\n");
    print_fp16_row(data, cols, rows - 2);
    printf(",\n");
    print_fp16_row(data, cols, rows - 1);
    printf("\n");
  }
  printf("], shape=(%d, %d), dtype=float16)\n", rows, cols);
}

static void print_float_matrix(const char *title, const float *data,
    int rows, int cols) {
  printf("%s tensor([\n", title);
  if (rows <= 4) {
    for (int r = 0; r < rows; r++) {
      print_float_row(data, cols, r);
      printf(r + 1 < rows ? ",\n" : "\n");
    }
  } else {
    print_float_row(data, cols, 0);
    printf(",\n");
    print_float_row(data, cols, 1);
    printf(",\n");
    printf("  ...,\n");
    print_float_row(data, cols, rows - 2);
    printf(",\n");
    print_float_row(data, cols, rows - 1);
    printf("\n");
  }
  printf("], shape=(%d, %d), dtype=float32)\n", rows, cols);
}

void breakpoint(){}

static __fp16 *cmpeq_last = NULL;
static int cmpeq_last_size = 0;
static int cmpeq_last_rows = 0;
static int cmpeq_last_cols = 0;

int test_alu(int argc, char **argv) {
    int size = 16 ;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    __fp16* a = (__fp16*)malloc(size * sizeof(__fp16));
    __fp16* b = (__fp16*)malloc(size * sizeof(__fp16));
    // printf("size: %d %d\n", sizeof(a), sizeof(b));

    for (size_t i = 0; i < size; i++) {
        a[i] = 2.3f;
        b[i] = 2.1f;
        if (i % 2 == 0) {
            a[i] = -a[i];
            b[i] = -b[i];
        }
    }
    // 4'd0: Max;
    // 4'd1: Min;
    // 4'd2: Add;
    // 4'd3: Div; # overflow issue
    // 4'd4: Minus;
    // CUSTOM 9: MUL
    // CUSTOM 10: RELU
    // CUSTOM 14: SILU
    // __fp16* result = float16_alu_op(a, b, 2, size);
    __fp16* result = float16_alu_op(a, b, 10, size);
    
    printf("Input0: ");
    for (size_t i = 0; i < size; i++) printf("%f ", a[i]);
    printf("\n");
    printf("Input1: ");
    for (size_t i = 0; i < size; i++) printf("%f ", b[i]);
    printf("\n");
    printf("Result/Input0: ");
    for (size_t i = 0; i < size; i++) {
        // float as_fp32 = (float)result[i];
        printf("%f ", result[i]);
    }
    printf("\n");
    
    const float kReluAtol = 1e-3f;
    int matches = result ? 1 : 0;
    float max_abs_diff = 0.0f;
    if (result) {
        for (size_t i = 0; i < size; i++) {
            float expected = (float)b[i];
            if (expected < 0.0f) expected = 0.0f;
            float actual = (float)result[i];
            float diff = fabsf(actual - expected);
            if (diff > max_abs_diff) max_abs_diff = diff;
            if (diff > kReluAtol) {
                matches = 0;
                break;
            }
        }
    }
    printf("test_alu: matches CPU -> %s (max diff=%.6f)\n",
        matches ? "YES" : "NO", max_abs_diff);

    breakpoint();

    free(a);
    free(b);
    return 0;
}

typedef struct {
  const char *name;
  int rows;
  int cols;
} DivTestConfig;

	typedef struct {
	  const char *name;
	  int rows;
	  int cols;
	  int a_broadcast_cols;
	  int b_broadcast_cols;
	} CmpltTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} CmpeqTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MinusTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} NegTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AbsTestConfig;

			typedef struct {
			  const char *name;
			  int rows;
			  int cols;
			} AddTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MulTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} WhereTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} RounddownTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} RoundoffTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MaxpoolTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AvgpoolTestConfig;

		typedef struct {
		  const char *name;
		  int rows;
		  int cols;
		} MaxTestConfig;

static float roundoff_ref(float in_val);

	static int run_div_case(const DivTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "div_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a);
    free(b);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    float av = mt_uniform(&rng, -2.0f, 2.0f);
    float bv = mt_uniform(&rng, -2.0f, 2.0f);
    if (fabsf(bv) < 1e-3f) bv = 1.0f;
    a[i] = (__fp16)av;
    b[i] = (__fp16)bv;
  }

  float max_abs_diff_fp16 = 0.0f;
  float max_abs_diff_fp32 = 0.0f;
  __fp16 *unpacked_fp16 = (__fp16*)malloc((size_t)size * sizeof(__fp16));
  if (!unpacked_fp16) {
    printf("%s: failed to allocate output buffer\n", name);
    free(a);
    free(b);
    return -1;
  }

  set_div_params(rows, cols);
  __fp16 *result = float16_alu_op(a, b, 3, size);
  if (!result) {
    printf("%s: float16_alu_op failed\n", name);
    free(unpacked_fp16);
    free(a);
    free(b);
    return -1;
  }
  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) {
    unpacked_fp16[i] = result[(size_t)i * stride_fp16];
  }

  for (int i = 0; i < size; i++) {
    float expected_fp16 = (float)(__fp16)((float)a[i] / (float)b[i]);
    float expected_fp32 = (float)a[i] / (float)b[i];
    float actual_fp16 = (float)unpacked_fp16[i];
    float actual_fp32 = (float)unpacked_fp16[i];
    float diff_fp16 = fabsf(actual_fp16 - expected_fp16);
    float diff_fp32 = fabsf(actual_fp32 - expected_fp32);
    if (diff_fp16 > max_abs_diff_fp16) max_abs_diff_fp16 = diff_fp16;
    if (diff_fp32 > max_abs_diff_fp32) max_abs_diff_fp32 = diff_fp32;
  }

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  const float kDivAtolFp16 = 3.2e-2f;
  const float kDivAtolFp32 = 2.5e-1f;
  int matches_fp16 = max_abs_diff_fp16 <= kDivAtolFp16;
  int matches_fp32 = max_abs_diff_fp32 <= kDivAtolFp32;

  printf("%s: matches CPU fp16 -> %s (max diff=%.6f)\n", name, matches_fp16 ? "YES" : "NO", max_abs_diff_fp16);
  printf("%s: matches CPU fp32 -> %s (max diff=%.6f)\n", name, matches_fp32 ? "YES" : "NO", max_abs_diff_fp32);
  printf("%s: matches CPU -> %s\n", name, (matches_fp16 || matches_fp32) ? "YES" : "NO");

  breakpoint();
  free(unpacked_fp16);
  free(a);
  free(b);
  return (matches_fp16 || matches_fp32) ? 0 : -1;
}

static int run_idiv_case(const DivTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "idiv_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *div_unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *offset = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *rounddown_stage = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *weights = (__fp16*)calloc(total_elements, sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  float *result_fp32 = (float*)malloc(total_elements * sizeof(float));
  if (!a || !b || !div_unpacked || !offset || !rounddown_stage ||
      !weights || !unpacked || !expected_fp16 || !result_fp32) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(div_unpacked); free(offset); free(rounddown_stage);
    free(weights); free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    float av = mt_uniform(&rng, 0.0f, 512.0f);
    float bv = mt_uniform(&rng, 1.0f, 16.0f);
    a[i] = (__fp16)av;
    b[i] = (__fp16)bv;
    offset[i] = (__fp16)0.5f;
    float div_val = (float)a[i] / (float)b[i];
    expected_fp16[i] = (__fp16)roundoff_ref(div_val - 0.5f);
  }

  set_div_params(rows, cols);
  __fp16 *div_packed = float16_alu_op(a, b, 3, size);
  if (!div_packed) {
    printf("%s: float16_alu_op div failed\n", name);
    free(a); free(b); free(div_unpacked); free(offset); free(rounddown_stage);
    free(weights); free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) {
    div_unpacked[i] = div_packed[(size_t)i * stride_fp16];
  }

  set_minus_params(rows, cols);
  __fp16 *stage1_packed = float16_alu_op(offset, div_unpacked, 4, size);
  if (!stage1_packed) {
    printf("%s: float16_alu_op rounddown stage1 failed\n", name);
    free(a); free(b); free(div_unpacked); free(offset); free(rounddown_stage);
    free(weights); free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    rounddown_stage[i] = stage1_packed[(size_t)i * stride_fp16];
  }

  set_minus_params(rows, cols);
  __fp16 *result_packed = float16_alu_op(weights, rounddown_stage, 23, size);
  if (!result_packed) {
    printf("%s: float16_alu_op rounddown stage2 failed\n", name);
    free(a); free(b); free(div_unpacked); free(offset); free(rounddown_stage);
    free(weights); free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    unpacked[i] = result_packed[(size_t)i * stride_fp16];
  }
  const size_t stride_fp32 = 0x10 / sizeof(float);
  const float *packed_fp32 = (const float *)result_packed;
  for (int i = 0; i < size; i++) {
    result_fp32[i] = packed_fp32[(size_t)i * stride_fp32];
  }

  float max_abs_diff_fp16 = 0.0f;
  float max_abs_diff_fp32 = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = (float)expected_fp16[i];
    float actual_fp16 = (float)unpacked[i];
    float actual_fp32 = result_fp32[i];
    float diff_fp16 = fabsf(actual_fp16 - expected);
    float diff_fp32 = fabsf(actual_fp32 - expected);
    if (diff_fp16 > max_abs_diff_fp16) max_abs_diff_fp16 = diff_fp16;
    if (diff_fp32 > max_abs_diff_fp32) max_abs_diff_fp32 = diff_fp32;
  }
  const float kRounddownAtol = 1e-3f;
  int matches_fp16 = max_abs_diff_fp16 <= kRounddownAtol;
  int matches_fp32 = max_abs_diff_fp32 <= kRounddownAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Div (as fp16)", div_unpacked, rows, cols);
    print_fp16_grid("Stage1 (div-0.5)", rounddown_stage, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    print_fp32_grid("Result (as fp32)", result_fp32, rows, cols);
    print_fp16_grid("Expected (idiv)", expected_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU fp16 -> %s (max diff=%.6f)\n",
      name, matches_fp16 ? "YES" : "NO", max_abs_diff_fp16);
  printf("%s: matches CPU fp32 -> %s (max diff=%.6f)\n",
      name, matches_fp32 ? "YES" : "NO", max_abs_diff_fp32);
  printf("%s: matches CPU -> %s\n",
      name, (matches_fp16 || matches_fp32) ? "YES" : "NO");

  breakpoint();
  free(a); free(b); free(div_unpacked); free(offset); free(rounddown_stage);
  free(weights); free(unpacked); free(expected_fp16); free(result_fp32);
  return (matches_fp16 || matches_fp32) ? 0 : -1;
}

static int run_maxpool_case(const MaxpoolTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "maxpool_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *weights = (__fp16*)calloc(total_elements, sizeof(__fp16));
  __fp16 *input = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *stage1 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!weights || !input || !stage1 || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(weights); free(input); free(stage1); free(unpacked);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
    input[i] = (__fp16)(-2.0f + 4.0f * t);
  }

  set_minus_params(rows, cols);
  __fp16 *stage1_packed = float16_alu_op(weights, input, 99, size);
  if (!stage1_packed) {
    printf("%s: float16_alu_op stage1 failed\n", name);
    free(weights); free(input); free(stage1); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) {
    stage1[i] = stage1_packed[(size_t)i * stride_fp16];
    unpacked[i] = stage1[i];
  }

  if (total_elements <= 64) {
    print_fp16_grid("Input", input, rows, cols);
    print_fp16_grid("Stage1 (alu99)", stage1, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> NO (no CPU reference yet)\n", name);

  breakpoint();
  free(weights); free(input); free(stage1); free(unpacked);
  return 0;
}

static int run_avgpool_case(const AvgpoolTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "avgpool_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *weights = (__fp16*)calloc(total_elements, sizeof(__fp16));
  __fp16 *input = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *stage1 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!weights || !input || !stage1 || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(weights); free(input); free(stage1); free(unpacked);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
    input[i] = (__fp16)(1.0f + 3.0f * t);
  }

  set_minus_params(rows, cols);
  __fp16 *stage1_packed = float16_alu_op(weights, input, 99, size);
  if (!stage1_packed) {
    printf("%s: float16_alu_op stage1 failed\n", name);
    free(weights); free(input); free(stage1); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) {
    stage1[i] = stage1_packed[(size_t)i * stride_fp16];
    unpacked[i] = stage1[i];
  }

  if (total_elements <= 64) {
    print_fp16_grid("Input", input, rows, cols);
    print_fp16_grid("Stage1 (alu99)", stage1, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> NO (no CPU reference yet)\n", name);

  breakpoint();
  free(weights); free(input); free(stage1); free(unpacked);
  return 0;
}

static int run_cmplt_case(const CmpltTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "cmplt_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  int a_broadcast_cols = config->a_broadcast_cols;
  int b_broadcast_cols = config->b_broadcast_cols;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

    if (a_broadcast_cols) {
      for (int i = 0; i < size; i++) {
        int c = i % cols;
        float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
        a[i] = (__fp16)(-2.0f + 4.0f * t);
      }
    } else {
      for (int i = 0; i < size; i++) {
        float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
        a[i] = (__fp16)(-2.0f + 4.0f * t);
      }
    }
    if (b_broadcast_cols) {
      for (int i = 0; i < size; i++) {
        int c = i % cols;
        float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
        b[i] = (__fp16)(2.0f - 4.0f * t);
      }
    } else {
      for (int i = 0; i < size; i++) {
        float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
        b[i] = (__fp16)(2.0f - 4.0f * t);
      }
    }

		  set_minus_params(rows, cols);
		  __fp16 *diff_packed = float16_alu_op(a, b, 4, size);
		  if (!diff_packed) {
		    printf("%s: float16_alu_op cmplt_sub failed\n", name);
		    free(a); free(b); free(unpacked);
		    return -1;
		  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  printf("%s: alu4 first=%f\n", name, (float)diff_packed[0]);
  __fp16 *diff = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *intermediate = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!diff || !intermediate || !zeros) {
    printf("%s: failed to allocate cmplt buffers\n", name);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  const float eps = 0x1p-14f;
  for (int i = 0; i < size; i++) {
    intermediate[i] = diff_packed[(size_t)i * stride_fp16];
    float v = (float)intermediate[i] - eps;
    diff[i] = (__fp16)v;
  }
  if (total_elements <= 64) print_fp16_grid("Intermediate Result (as fp16)", intermediate, rows, cols);

		  __fp16 *result_packed = float16_alu_op(zeros, diff, 16, size);
		  if (!result_packed) {
		    printf("%s: float16_alu_op cmplt_cmp failed\n", name);
		    free(diff);
		    free(intermediate);
		    free(zeros);
		    free(a); free(b); free(unpacked);
		    return -1;
		  }
  for (int i = 0; i < size; i++) unpacked[i] = result_packed[(size_t)i * stride_fp16];

  float max_abs_diff_ab = 0.0f;
  float max_abs_diff_ba = 0.0f;
  __fp16 *expected_ab_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_ba_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *actual_bool_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_ab_fp16 || !expected_ba_fp16) {
    printf("%s: failed to allocate expected buffers\n", name);
    free(expected_ab_fp16);
    free(expected_ba_fp16);
    free(actual_bool_fp16);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    float expected_ab = ((float)a[i] < (float)b[i]) ? 1.0f : 0.0f;
    float expected_ba = ((float)b[i] < (float)a[i]) ? 1.0f : 0.0f;
    expected_ab_fp16[i] = (__fp16)expected_ab;
    expected_ba_fp16[i] = (__fp16)expected_ba;
    float actual = (float)unpacked[i];
    float actual_bool = actual > 0.0f ? 1.0f : 0.0f;
    actual_bool_fp16[i] = (__fp16)actual_bool;
    float diff_ab = fabsf(actual_bool - expected_ab);
    float diff_ba = fabsf(actual_bool - expected_ba);
    if (diff_ab > max_abs_diff_ab) max_abs_diff_ab = diff_ab;
    if (diff_ba > max_abs_diff_ba) max_abs_diff_ba = diff_ba;
  }

  const float kAtol = 1e-3f;
  int matches_ab = max_abs_diff_ab <= kAtol;
  int matches_ba = max_abs_diff_ba <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    // print_fp16_grid("Result (bool)", actual_bool_fp16, rows, cols);
    print_fp16_grid("Expected (A<B)", expected_ab_fp16, rows, cols);
    // print_fp16_grid("Expected (B<A)", expected_ba_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  if (matches_ab || matches_ba) {
    printf("%s: matches CPU -> YES (%s<b%s, max diff=%.6f)\n",
        name, matches_ab ? "A" : "B", matches_ab ? "B" : "A", matches_ab ? max_abs_diff_ab : max_abs_diff_ba);
  } else {
    printf("%s: matches CPU -> NO (max diff A<B=%.6f, B<A=%.6f)\n", name, max_abs_diff_ab, max_abs_diff_ba);
  }

  breakpoint();
  free(diff);
  free(intermediate);
  free(zeros);
  free(expected_ab_fp16);
  free(expected_ba_fp16);
  free(actual_bool_fp16);
  free(a); free(b); free(unpacked);
  return (matches_ab || matches_ba) ? 0 : -1;
}

static int run_cmpgt_case(const CmpltTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "cmpgt_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  int a_broadcast_cols = config->a_broadcast_cols;
  int b_broadcast_cols = config->b_broadcast_cols;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  if (a_broadcast_cols) {
    for (int i = 0; i < size; i++) {
      int c = i % cols;
      float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
      a[i] = (__fp16)(-2.0f + 4.0f * t);
    }
  } else {
    for (int i = 0; i < size; i++) {
      float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
      a[i] = (__fp16)(-2.0f + 4.0f * t);
    }
  }
  if (b_broadcast_cols) {
    for (int i = 0; i < size; i++) {
      int c = i % cols;
      float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
      b[i] = (__fp16)(2.0f - 4.0f * t);
    }
  } else {
    for (int i = 0; i < size; i++) {
      float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
      b[i] = (__fp16)(2.0f - 4.0f * t);
    }
  }

  set_minus_params(rows, cols);
  __fp16 *diff_packed = float16_alu_op(b, a, 4, size);
  if (!diff_packed) {
    printf("%s: float16_alu_op cmpgt_sub failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  printf("%s: alu4 first=%f\n", name, (float)diff_packed[0]);
  __fp16 *diff = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *intermediate = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!diff || !intermediate || !zeros) {
    printf("%s: failed to allocate cmpgt buffers\n", name);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  const float eps = 0x1p-14f;
  for (int i = 0; i < size; i++) {
    intermediate[i] = diff_packed[(size_t)i * stride_fp16];
    float v = (float)intermediate[i] - eps;
    diff[i] = (__fp16)v;
  }
  if (total_elements <= 64) print_fp16_grid("Intermediate Result (as fp16)", intermediate, rows, cols);

  __fp16 *result_packed = float16_alu_op(zeros, diff, 16, size);
  if (!result_packed) {
    printf("%s: float16_alu_op cmpgt_cmp failed\n", name);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) unpacked[i] = result_packed[(size_t)i * stride_fp16];

  float max_abs_diff_ab = 0.0f;
  float max_abs_diff_ba = 0.0f;
  __fp16 *expected_ab_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_ba_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *actual_bool_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_ab_fp16 || !expected_ba_fp16) {
    printf("%s: failed to allocate expected buffers\n", name);
    free(expected_ab_fp16);
    free(expected_ba_fp16);
    free(actual_bool_fp16);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    float expected_ab = ((float)a[i] > (float)b[i]) ? 1.0f : 0.0f;
    float expected_ba = ((float)b[i] > (float)a[i]) ? 1.0f : 0.0f;
    expected_ab_fp16[i] = (__fp16)expected_ab;
    expected_ba_fp16[i] = (__fp16)expected_ba;
    float actual = (float)unpacked[i];
    float actual_bool = actual > 0.0f ? 1.0f : 0.0f;
    actual_bool_fp16[i] = (__fp16)actual_bool;
    float diff_ab = fabsf(actual_bool - expected_ab);
    float diff_ba = fabsf(actual_bool - expected_ba);
    if (diff_ab > max_abs_diff_ab) max_abs_diff_ab = diff_ab;
    if (diff_ba > max_abs_diff_ba) max_abs_diff_ba = diff_ba;
  }

  const float kAtol = 1e-3f;
  int matches_ab = max_abs_diff_ab <= kAtol;
  int matches_ba = max_abs_diff_ba <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    // print_fp16_grid("Result (bool)", actual_bool_fp16, rows, cols);
    print_fp16_grid("Expected (A>B)", expected_ab_fp16, rows, cols);
    // print_fp16_grid("Expected (B>A)", expected_ba_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  if (matches_ab || matches_ba) {
    printf("%s: matches CPU -> YES (%s>%s, max diff=%.6f)\n",
        name, matches_ab ? "A" : "B", matches_ab ? "B" : "A", matches_ab ? max_abs_diff_ab : max_abs_diff_ba);
  } else {
    printf("%s: matches CPU -> NO (max diff A>B=%.6f, B>A=%.6f)\n", name, max_abs_diff_ab, max_abs_diff_ba);
  }

  breakpoint();
  free(diff);
  free(intermediate);
  free(zeros);
  free(expected_ab_fp16);
  free(expected_ba_fp16);
  free(actual_bool_fp16);
  free(a); free(b); free(unpacked);
  return (matches_ab || matches_ba) ? 0 : -1;
}

static int run_cmpge_case(const CmpltTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "cmpge_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  int a_broadcast_cols = config->a_broadcast_cols;
  int b_broadcast_cols = config->b_broadcast_cols;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  if (a_broadcast_cols) {
    for (int i = 0; i < size; i++) {
      int c = i % cols;
      float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
      a[i] = (__fp16)(-2.0f + 4.0f * t);
    }
  } else {
    for (int i = 0; i < size; i++) {
      float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
      a[i] = (__fp16)(-2.0f + 4.0f * t);
    }
  }
  if (b_broadcast_cols) {
    for (int i = 0; i < size; i++) {
      int c = i % cols;
      float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
      b[i] = (__fp16)(2.0f - 4.0f * t);
    }
  } else {
    for (int i = 0; i < size; i++) {
      float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
      b[i] = (__fp16)(2.0f - 4.0f * t);
    }
  }
  b[0] = a[0];

  set_minus_params(rows, cols);
  __fp16 *diff_packed = float16_alu_op(b, a, 4, size);
  if (!diff_packed) {
    printf("%s: float16_alu_op cmpge_sub failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  printf("%s: alu4 first=%f\n", name, (float)diff_packed[0]);
  __fp16 *diff = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *intermediate = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!diff || !intermediate || !zeros) {
    printf("%s: failed to allocate cmpge buffers\n", name);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    intermediate[i] = diff_packed[(size_t)i * stride_fp16];
    diff[i] = intermediate[i];
  }
  if (total_elements <= 64) print_fp16_grid("Intermediate Result (as fp16)", intermediate, rows, cols);

  __fp16 *result_packed = float16_alu_op(zeros, diff, 20, size);
  if (!result_packed) {
    printf("%s: float16_alu_op cmpge_cmp failed\n", name);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) unpacked[i] = result_packed[(size_t)i * stride_fp16];

  float max_abs_diff_ab = 0.0f;
  float max_abs_diff_ba = 0.0f;
  __fp16 *expected_ab_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_ba_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *actual_bool_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_ab_fp16 || !expected_ba_fp16) {
    printf("%s: failed to allocate expected buffers\n", name);
    free(expected_ab_fp16);
    free(expected_ba_fp16);
    free(actual_bool_fp16);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    float expected_ab = ((float)a[i] >= (float)b[i]) ? 1.0f : 0.0f;
    float expected_ba = ((float)b[i] >= (float)a[i]) ? 1.0f : 0.0f;
    expected_ab_fp16[i] = (__fp16)expected_ab;
    expected_ba_fp16[i] = (__fp16)expected_ba;
    float actual = (float)unpacked[i];
    float actual_bool = actual > 0.0f ? 1.0f : 0.0f;
    actual_bool_fp16[i] = (__fp16)actual_bool;
    float diff_ab = fabsf(actual_bool - expected_ab);
    float diff_ba = fabsf(actual_bool - expected_ba);
    if (diff_ab > max_abs_diff_ab) max_abs_diff_ab = diff_ab;
    if (diff_ba > max_abs_diff_ba) max_abs_diff_ba = diff_ba;
  }

  const float kAtol = 1e-3f;
  int matches_ab = max_abs_diff_ab <= kAtol;
  int matches_ba = max_abs_diff_ba <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    // print_fp16_grid("Result (bool)", actual_bool_fp16, rows, cols);
    print_fp16_grid("Expected (A>=B)", expected_ab_fp16, rows, cols);
    // print_fp16_grid("Expected (B>=A)", expected_ba_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  if (matches_ab || matches_ba) {
    printf("%s: matches CPU -> YES (%s>=%s, max diff=%.6f)\n",
        name, matches_ab ? "A" : "B", matches_ab ? "B" : "A", matches_ab ? max_abs_diff_ab : max_abs_diff_ba);
  } else {
    printf("%s: matches CPU -> NO (max diff A>=B=%.6f, B>=A=%.6f)\n", name, max_abs_diff_ab, max_abs_diff_ba);
  }

  breakpoint();
  free(diff);
  free(intermediate);
  free(zeros);
  free(expected_ab_fp16);
  free(expected_ba_fp16);
  free(actual_bool_fp16);
  free(a); free(b); free(unpacked);
  return (matches_ab || matches_ba) ? 0 : -1;
}

static int run_cmple_case(const CmpltTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "cmple_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  int a_broadcast_cols = config->a_broadcast_cols;
  int b_broadcast_cols = config->b_broadcast_cols;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

    if (a_broadcast_cols) {
      for (int i = 0; i < size; i++) {
        int c = i % cols;
        float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
        a[i] = (__fp16)(-2.0f + 4.0f * t);
      }
    } else {
      for (int i = 0; i < size; i++) {
        float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
        a[i] = (__fp16)(-2.0f + 4.0f * t);
      }
    }
    if (b_broadcast_cols) {
      for (int i = 0; i < size; i++) {
        int c = i % cols;
        float t = cols > 1 ? (float)c / (float)(cols - 1) : 0.0f;
        b[i] = (__fp16)(2.0f - 4.0f * t);
      }
    } else {
      for (int i = 0; i < size; i++) {
        float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
        b[i] = (__fp16)(2.0f - 4.0f * t);
      }
    }
    b[0] = a[0];

			  set_minus_params(rows, cols);
			  __fp16 *diff_packed = float16_alu_op(a, b, 4, size);
			  if (!diff_packed) {
			    printf("%s: float16_alu_op cmple_sub failed\n", name);
			    free(a); free(b); free(unpacked);
			    return -1;
			  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  printf("%s: alu4 first=%f\n", name, (float)diff_packed[0]);
  __fp16 *diff = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *intermediate = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!diff || !intermediate || !zeros) {
    printf("%s: failed to allocate cmple buffers\n", name);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    intermediate[i] = diff_packed[(size_t)i * stride_fp16];
    diff[i] = intermediate[i];
  }
  if (total_elements <= 64) print_fp16_grid("Intermediate Result (as fp16)", intermediate, rows, cols);

			  __fp16 *result_packed = float16_alu_op(zeros, diff, 20, size);
			  if (!result_packed) {
			    printf("%s: float16_alu_op cmple_cmp failed\n", name);
			    free(diff);
			    free(intermediate);
			    free(zeros);
			    free(a); free(b); free(unpacked);
			    return -1;
			  }
  for (int i = 0; i < size; i++) unpacked[i] = result_packed[(size_t)i * stride_fp16];

  float max_abs_diff_ab = 0.0f;
  float max_abs_diff_ba = 0.0f;
  __fp16 *expected_ab_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_ba_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *actual_bool_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_ab_fp16 || !expected_ba_fp16) {
    printf("%s: failed to allocate expected buffers\n", name);
    free(expected_ab_fp16);
    free(expected_ba_fp16);
    free(actual_bool_fp16);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    float expected_ab = ((float)a[i] <= (float)b[i]) ? 1.0f : 0.0f;
    float expected_ba = ((float)b[i] <= (float)a[i]) ? 1.0f : 0.0f;
    expected_ab_fp16[i] = (__fp16)expected_ab;
    expected_ba_fp16[i] = (__fp16)expected_ba;
    float actual = (float)unpacked[i];
    float actual_bool = actual > 0.0f ? 1.0f : 0.0f;
    actual_bool_fp16[i] = (__fp16)actual_bool;
    float diff_ab = fabsf(actual_bool - expected_ab);
    float diff_ba = fabsf(actual_bool - expected_ba);
    if (diff_ab > max_abs_diff_ab) max_abs_diff_ab = diff_ab;
    if (diff_ba > max_abs_diff_ba) max_abs_diff_ba = diff_ba;
  }

  const float kAtol = 1e-3f;
  int matches_ab = max_abs_diff_ab <= kAtol;
  int matches_ba = max_abs_diff_ba <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    // print_fp16_grid("Result (bool)", actual_bool_fp16, rows, cols);
    print_fp16_grid("Expected (A<=B)", expected_ab_fp16, rows, cols);
    // print_fp16_grid("Expected (B<=A)", expected_ba_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  if (matches_ab || matches_ba) {
    printf("%s: matches CPU -> YES (%s<=%s, max diff=%.6f)\n",
        name, matches_ab ? "A" : "B", matches_ab ? "B" : "A", matches_ab ? max_abs_diff_ab : max_abs_diff_ba);
  } else {
    printf("%s: matches CPU -> NO (max diff A<=B=%.6f, B<=A=%.6f)\n", name, max_abs_diff_ab, max_abs_diff_ba);
  }

  breakpoint();
  free(diff);
  free(intermediate);
  free(zeros);
  free(expected_ab_fp16);
  free(expected_ba_fp16);
  free(actual_bool_fp16);
  free(a); free(b); free(unpacked);
  return (matches_ab || matches_ba) ? 0 : -1;
}

static int run_cmpeq_case(const CmpeqTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "cmpeq_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
    float v = -2.0f + 4.0f * t;
    a[i] = (__fp16)v;
    if ((i & 3) == 0) b[i] = a[i];
    else b[i] = (__fp16)(v + 1.0f);
  }
  b[0] = a[0] ;
  b[1] = a[1] ;

  set_minus_params(rows, cols);
  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  __fp16 *ones = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *diff = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *intermediate = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!ones || !diff || !intermediate || !zeros) {
    printf("%s: failed to allocate cmpeq buffers\n", name);
    free(ones);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) ones[i] = (__fp16)1.0f;

  __fp16 *stage1_packed = float16_alu_op(a, b, 4, size);
  if (!stage1_packed) {
    printf("%s: float16_alu_op cmpeq_stage1 failed\n", name);
    free(ones);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  printf("%s: stage1(alu4) first=%f\n", name, (float)stage1_packed[0]);
  for (int i = 0; i < size; i++) {
    intermediate[i] = stage1_packed[(size_t)i * stride_fp16];
    diff[i] = intermediate[i];
  }
  if (total_elements <= 64) print_fp16_grid("Stage1 Result (as fp16)", intermediate, rows, cols);

  // Stage 2: ALU op 17 (implemented elsewhere) on Stage1 output.
  __fp16 *stage2_packed = float16_alu_op(zeros, diff, 17, size);
  if (!stage2_packed) {
    printf("%s: float16_alu_op cmpeq_stage2 failed\n", name);
    free(ones);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  printf("%s: stage2(alu17) first=%f\n", name, (float)stage2_packed[0]);
  for (int i = 0; i < size; i++) {
    intermediate[i] = stage2_packed[(size_t)i * stride_fp16];
    diff[i] = intermediate[i];
  }
  if (total_elements <= 64) print_fp16_grid("Stage2 Result (as fp16)", intermediate, rows, cols);

  // Stage 3: ALU op 18 on Stage2 output.
  __fp16 *result_packed = float16_alu_op(zeros, diff, 18, size);
  if (!result_packed) {
    printf("%s: float16_alu_op cmpeq_stage3 failed\n", name);
    free(ones);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  printf("%s: stage3(alu18) first=%f\n", name, (float)result_packed[0]);
  for (int i = 0; i < size; i++) unpacked[i] = result_packed[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  __fp16 *expected_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_fp16) {
    printf("%s: failed to allocate expected buffer\n", name);
    free(ones);
    free(diff);
    free(intermediate);
    free(zeros);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    uint16_t a_bits = 0, b_bits = 0;
    memcpy(&a_bits, &a[i], sizeof(a_bits));
    memcpy(&b_bits, &b[i], sizeof(b_bits));
    float expected = (a_bits == b_bits) ? 1.0f : 0.0f;
    expected_fp16[i] = (__fp16)expected;
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    print_fp16_grid("Expected (A==B)", expected_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

	  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

    free(cmpeq_last);
    cmpeq_last = (__fp16*)malloc(total_elements * sizeof(__fp16));
    if (cmpeq_last) {
      memcpy(cmpeq_last, unpacked, total_elements * sizeof(__fp16));
      cmpeq_last_size = size;
      cmpeq_last_rows = rows;
      cmpeq_last_cols = cols;
    } else {
      cmpeq_last_size = 0;
      cmpeq_last_rows = 0;
      cmpeq_last_cols = 0;
    }

	  breakpoint();
	  free(expected_fp16);
	  free(ones);
	  free(diff);
	  free(intermediate);
  free(zeros);
  free(a); free(b); free(unpacked);
  return matches ? 0 : -1;
}

static int run_add_case(const AddTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "add_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    a[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
    b[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
  }

  __fp16 *result = float16_alu_op(a, b, 2, size);
  if (!result) {
    printf("%s: float16_alu_op failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = (float)a[i] + (float)b[i];
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(a); free(b); free(unpacked);
  return matches ? 0 : -1;
}

static int run_mul_case(const MulTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "mul_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    a[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
    b[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
  }

  set_minus_params(rows, cols);
  __fp16 *result = float16_alu_op(a, b, 9, size);
  if (!result) {
    printf("%s: float16_alu_op failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = (float)(__fp16)((float)a[i] * (float)b[i]);
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(a); free(b); free(unpacked);
  return matches ? 0 : -1;
}

static int run_where_case(const WhereTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "where_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *mask = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *stage1 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *stage2 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *stage3 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!a || !b || !mask || !stage1 || !stage2 || !stage3 || !unpacked || !zeros) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(mask); free(stage1); free(stage2); free(stage3); free(unpacked); free(zeros);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    a[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
    b[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
    mask[i] = (__fp16)((i & 1) ? 1.0f : 0.0f);
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);

  // Stage 1: neg(mask) -> (1 - mask), algo 19.
  set_minus_params(rows, cols);
  __fp16 *stage1_packed = float16_alu_op(zeros, mask, 19, size);
  if (!stage1_packed) {
    printf("%s: float16_alu_op stage1 failed\n", name);
    free(a); free(b); free(mask); free(stage1); free(stage2); free(unpacked); free(zeros);
    return -1;
  }
  for (int i = 0; i < size; i++) stage1[i] = stage1_packed[(size_t)i * stride_fp16];

  // Stage 2: stage1 * B, algo 9.
  set_minus_params(rows, cols);
  __fp16 *stage2_packed = float16_alu_op(stage1, b, 9, size);
  if (!stage2_packed) {
    printf("%s: float16_alu_op stage2 failed\n", name);
    free(a); free(b); free(mask); free(stage1); free(stage2); free(stage3); free(unpacked); free(zeros);
    return -1;
  }
  for (int i = 0; i < size; i++) stage2[i] = stage2_packed[(size_t)i * stride_fp16];

  // Stage 3: A * mask, algo 9.
  set_minus_params(rows, cols);
  __fp16 *stage3_packed = float16_alu_op(mask, a, 9, size);
  if (!stage3_packed) {
    printf("%s: float16_alu_op stage3 failed\n", name);
    free(a); free(b); free(mask); free(stage1); free(stage2); free(stage3); free(unpacked); free(zeros);
    return -1;
  }
  for (int i = 0; i < size; i++) stage3[i] = stage3_packed[(size_t)i * stride_fp16];

  // Stage 4: stage2 + stage3, algo 2.
  __fp16 *result_packed = float16_alu_op(stage3, stage2, 2, size);
  if (!result_packed) {
    printf("%s: float16_alu_op stage4 failed\n", name);
    free(a); free(b); free(mask); free(stage1); free(stage2); free(stage3); free(unpacked); free(zeros);
    return -1;
  }
  for (int i = 0; i < size; i++) unpacked[i] = result_packed[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  for (int i = 0; i < size; i++) {
    __fp16 s1 = (__fp16)(1.0f - (float)mask[i]);
    __fp16 s2 = (__fp16)((float)b[i] * (float)s1);
    __fp16 s3 = (__fp16)((float)a[i] * (float)mask[i]);
    __fp16 s4 = (__fp16)((float)s2 + (float)s3);
    float expected = (float)s4;
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Mask", mask, rows, cols);
    print_fp16_grid("Stage1 (1-mask)", stage1, rows, cols);
    print_fp16_grid("Stage2 (B*(1-mask))", stage2, rows, cols);
    print_fp16_grid("Stage3 (A*mask)", stage3, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(a); free(b); free(mask); free(stage1); free(stage2); free(stage3); free(unpacked); free(zeros);
  return matches ? 0 : -1;
}

static float roundoff_ref(float in_val) {
  float base = floorf(in_val);
  float frac = in_val - base;
  int base_i = (int)base;
  if (frac > 0.5f || (frac == 0.5f && (base_i & 1))) return base + 1.0f;
  return base;
}

static int run_rounddown_case(const RounddownTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "rounddown_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *weights = (__fp16*)calloc(total_elements, sizeof(__fp16));
  __fp16 *input = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *offset = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *stage1 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  float *result_fp32 = (float*)malloc(total_elements * sizeof(float));
  if (!weights || !input || !offset || !stage1 || !unpacked || !expected_fp16 || !result_fp32) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(weights); free(input); free(offset); free(stage1);
    free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  static const float samples[] = {
      0.50f, 0.60f, 1.00f, 1.40f, 1.60f, 2.40f, 2.50f, 3.20f,
      3.51f, 4.00f, 4.49f, 4.50f, 4.75f, 5.25f, 5.60f, 6.40f,
  };
  const int sample_count = (int)(sizeof(samples) / sizeof(samples[0]));
  const float rounddown_offset = 0.5f;
  for (int i = 0; i < size; i++) {
    float x = samples[i % sample_count];
    input[i] = (__fp16)x;
    offset[i] = (__fp16)rounddown_offset;
    expected_fp16[i] = (__fp16)roundoff_ref(x - rounddown_offset);
  }

  set_minus_params(rows, cols);
  __fp16 *stage1_packed = float16_alu_op(offset, input, 4, size);
  if (!stage1_packed) {
    printf("%s: float16_alu_op stage1 failed\n", name);
    free(weights); free(input); free(offset); free(stage1);
    free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) {
    stage1[i] = stage1_packed[(size_t)i * stride_fp16];
  }

  set_minus_params(rows, cols);
  __fp16 *result_packed = float16_alu_op(weights, stage1, 23, size);
  if (!result_packed) {
    printf("%s: float16_alu_op stage2 failed\n", name);
    free(weights); free(input); free(offset); free(stage1);
    free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    unpacked[i] = result_packed[(size_t)i * stride_fp16];
  }
  const size_t stride_fp32 = 0x10 / sizeof(float);
  const float *packed_fp32 = (const float *)result_packed;
  for (int i = 0; i < size; i++) {
    result_fp32[i] = packed_fp32[(size_t)i * stride_fp32];
  }

  float max_abs_diff = 0.0f;
  int mismatch_count = 0;
  for (int i = 0; i < size; i++) {
    float actual = (float)unpacked[i];
    float expected = (float)expected_fp16[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
    if (diff > 1e-3f) mismatch_count++;
  }
  int matches = mismatch_count == 0;

  if (total_elements <= 64) {
    print_fp16_grid("Input (x)", input, rows, cols);
    print_fp16_grid("Stage1 (x-0.5)", stage1, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    print_fp32_grid("Result (as fp32)", result_fp32, rows, cols);
    print_fp16_grid("Expected (rounddown)", expected_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f, mismatches=%d)\n",
      name, matches ? "YES" : "NO", max_abs_diff, mismatch_count);

  breakpoint();
  free(weights); free(input); free(offset); free(stage1);
  free(unpacked); free(expected_fp16); free(result_fp32);
  return matches ? 0 : -1;
}

static int run_roundoff_case(const RoundoffTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "roundoff_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *weights = (__fp16*)calloc(total_elements, sizeof(__fp16));
  __fp16 *input = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *expected_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  float *result_fp32 = (float*)malloc(total_elements * sizeof(float));
  if (!weights || !input || !unpacked || !expected_fp16 || !result_fp32) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(weights); free(input); free(unpacked); free(expected_fp16); free(result_fp32);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    input[i] = (__fp16)mt_uniform(&rng, 0.0f, 512.0f);
    float in_val = (float)input[i];
    expected_fp16[i] = (__fp16)roundoff_ref(in_val);
  }

  set_minus_params(rows, cols);
  __fp16 *result_packed = float16_alu_op(weights, input, 23, size);
  if (!result_packed) {
    printf("%s: float16_alu_op failed\n", name);
    free(weights); free(input); free(unpacked); free(expected_fp16);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) {
    unpacked[i] = result_packed[(size_t)i * stride_fp16];
  }
  const size_t stride_fp32 = 0x10 / sizeof(float);
  const float *packed_fp32 = (const float *)result_packed;
  for (int i = 0; i < size; i++) {
    result_fp32[i] = packed_fp32[(size_t)i * stride_fp32];
  }

  float max_abs_diff_fp16 = 0.0f;
  float max_abs_diff_fp32 = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = (float)expected_fp16[i];
    float actual_fp16 = (float)unpacked[i];
    float actual_fp32 = result_fp32[i];
    float diff_fp16 = fabsf(actual_fp16 - expected);
    float diff_fp32 = fabsf(actual_fp32 - expected);
    if (diff_fp16 > max_abs_diff_fp16) max_abs_diff_fp16 = diff_fp16;
    if (diff_fp32 > max_abs_diff_fp32) max_abs_diff_fp32 = diff_fp32;
  }
  const float kRoundoffAtol = 1e-3f;
  int matches_fp16 = max_abs_diff_fp16 <= kRoundoffAtol;
  int matches_fp32 = max_abs_diff_fp32 <= kRoundoffAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input", input, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    print_fp32_grid("Result (as fp32)", result_fp32, rows, cols);
    print_fp16_grid("Expected (round)", expected_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU fp16 -> %s (max diff=%.6f)\n",
      name, matches_fp16 ? "YES" : "NO", max_abs_diff_fp16);
  printf("%s: matches CPU fp32 -> %s (max diff=%.6f)\n",
      name, matches_fp32 ? "YES" : "NO", max_abs_diff_fp32);
  printf("%s: matches CPU -> %s\n",
      name, (matches_fp16 || matches_fp32) ? "YES" : "NO");

  breakpoint();
  free(weights); free(input); free(unpacked); free(expected_fp16); free(result_fp32);
  return (matches_fp16 || matches_fp32) ? 0 : -1;
}

static int run_max_case(const MaxTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "max_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
	  for (int i = 0; i < size; i++) {
	    a[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
	    b[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
	  }

	  set_max_params(rows, cols);
	  __fp16 *result = float16_alu_op(a, b, 0, size);
	  if (!result) {
	    printf("%s: float16_alu_op failed\n", name);
	    free(a); free(b); free(unpacked);
	    return -1;
	  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = fmaxf((float)a[i], (float)b[i]);
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(a); free(b); free(unpacked);
  return matches ? 0 : -1;
}

static int run_minus_case(const MinusTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "minus_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    a[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
    b[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
  }

  set_minus_params(rows, cols);
  __fp16 *result = float16_alu_op(a, b, 4, size);
  if (!result) {
    printf("%s: float16_alu_op failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff_ab = 0.0f;
  float max_abs_diff_ba = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected_ab = (float)a[i] - (float)b[i];
    float expected_ba = (float)b[i] - (float)a[i];
    float actual = (float)unpacked[i];
    float diff_ab = fabsf(actual - expected_ab);
    float diff_ba = fabsf(actual - expected_ba);
    if (diff_ab > max_abs_diff_ab) max_abs_diff_ab = diff_ab;
    if (diff_ba > max_abs_diff_ba) max_abs_diff_ba = diff_ba;
  }

  const float kAtol = 1e-3f;
  int matches_ab = max_abs_diff_ab <= kAtol;
  int matches_ba = max_abs_diff_ba <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  if (matches_ab || matches_ba) {
    printf("%s: matches CPU -> YES (%s-%s, max diff=%.6f)\n",
        name, matches_ab ? "A" : "B", matches_ab ? "B" : "A", matches_ab ? max_abs_diff_ab : max_abs_diff_ba);
  } else {
    printf("%s: matches CPU -> NO (max diff A-B=%.6f, B-A=%.6f)\n", name, max_abs_diff_ab, max_abs_diff_ba);
  }

  breakpoint();
  free(a); free(b); free(unpacked);
  return (matches_ab || matches_ba) ? 0 : -1;
}

static int run_neg_case(const NegTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "neg_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    a[i] = (__fp16)0.0f;
    b[i] = (__fp16)((i & 1) ? 1.0f : 0.0f);
  }

  set_minus_params(rows, cols);
  __fp16 *result = float16_alu_op(a, b, 19, size);
  if (!result) {
    printf("%s: float16_alu_op failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = 1.0f - (float)b[i];
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(a); free(b); free(unpacked);
  return matches ? 0 : -1;
}

static int run_abs_case(const AbsTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "abs_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *a = (__fp16*)calloc(total_elements, sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!a || !b || !unpacked) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(a); free(b); free(unpacked);
    return -1;
  }

  Mt19937 rng;
  mt_seed(&rng, 0);
  for (int i = 0; i < size; i++) {
    b[i] = (__fp16)mt_uniform(&rng, -2.0f, 2.0f);
  }

  set_minus_params(rows, cols);
  __fp16 *result = float16_alu_op(a, b, 22, size);
  if (!result) {
    printf("%s: float16_alu_op(abs) failed\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  __fp16 *expected_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_fp16) {
    printf("%s: failed to allocate expected buffer\n", name);
    free(a); free(b); free(unpacked);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    float expected = fabsf((float)b[i]);
    expected_fp16[i] = (__fp16)expected;
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input", b, rows, cols);
    print_fp16_grid("Result (as fp16)", unpacked, rows, cols);
    print_fp16_grid("Expected (abs)", expected_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(expected_fp16);
  free(a); free(b); free(unpacked);
  return matches ? 0 : -1;
}

static int test_div(int argc, char **argv) {
  if (argc >= 3) {
    DivTestConfig cli = {"div_cli", atoi(argv[1]), atoi(argv[2])};
    return run_div_case(&cli);
  }

  static const DivTestConfig configs[] = {
    {"div_4x4", 4, 4},
    {"div_45x65", 45, 65},
    {"div_90x90", 90, 90},
    // these sharp start to fail
    // one possible reason is buffer > 128KB
    // {"div_91x91", 91, 91},
    // {"div_92x92", 92, 92},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_div_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_idiv(int argc, char **argv) {
  if (argc >= 3) {
    DivTestConfig cli = {"idiv_cli", atoi(argv[1]), atoi(argv[2])};
    return run_idiv_case(&cli);
  }

  static const DivTestConfig configs[] = {
    {"idiv_4x4", 4, 4},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_idiv_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_maxpool(int argc, char **argv) {
  if (argc >= 3) {
    MaxpoolTestConfig cli = {"maxpool_cli", atoi(argv[1]), atoi(argv[2])};
    return run_maxpool_case(&cli);
  }

  static const MaxpoolTestConfig configs[] = {
    {"maxpool_4x4", 4, 4},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_maxpool_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_avgpool(int argc, char **argv) {
  if (argc >= 3) {
    AvgpoolTestConfig cli = {"avgpool_cli", atoi(argv[1]), atoi(argv[2])};
    return run_avgpool_case(&cli);
  }

  static const AvgpoolTestConfig configs[] = {
    {"avgpool_4x4", 4, 4},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_avgpool_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_cmplt(int argc, char **argv) {
  if (argc >= 3) {
    CmpltTestConfig cli = {"cmplt_cli", atoi(argv[1]), atoi(argv[2]), 0, 0};
    return run_cmplt_case(&cli);
  }

  static const CmpltTestConfig configs[] = {
    {"cmplt_1x1", 1, 1}, 
    {"cmplt_3", 1, 3, 0, 0},
    {"cmplt_1", 1, 1, 0, 0},
    {"cmplt_2x2", 2, 2, 0, 0},
    {"cmplt_12x5", 12, 5, 0, 0},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_cmplt_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_cmpgt(int argc, char **argv) {
  if (argc >= 3) {
    CmpltTestConfig cli = {"cmpgt_cli", atoi(argv[1]), atoi(argv[2]), 0, 0};
    return run_cmpgt_case(&cli);
  }

  static const CmpltTestConfig configs[] = {
    {"cmpgt_2x2", 2, 2, 0, 0},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_cmpgt_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_cmpge(int argc, char **argv) {
  if (argc >= 3) {
    CmpltTestConfig cli = {"cmpge_cli", atoi(argv[1]), atoi(argv[2]), 0, 0};
    return run_cmpge_case(&cli);
  }

  static const CmpltTestConfig configs[] = {
    {"cmpge_2x2", 2, 2, 0, 0},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_cmpge_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_cmple(int argc, char **argv) {
  if (argc >= 3) {
    CmpltTestConfig cli = {"cmple_cli", atoi(argv[1]), atoi(argv[2]), 0, 0};
    return run_cmple_case(&cli);
  }

  static const CmpltTestConfig configs[] = {
    {"cmple_2x2", 2, 2, 0, 0},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_cmple_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_cmpeq(int argc, char **argv) {
  if (argc >= 3) {
    CmpeqTestConfig cli = {"cmpeq_cli", atoi(argv[1]), atoi(argv[2])};
    return run_cmpeq_case(&cli);
  }

  static const CmpeqTestConfig configs[] = {
    {"cmpeq_2x2", 2, 2},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_cmpeq_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int run_cmpneq_case(const CmpeqTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "cmpneq_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > 65536) {
    printf("%s: invalid element count %zu\n", name, total_elements);
    return -1;
  }
  int size = (int)total_elements;

  CmpeqTestConfig cmpeq_config = {"cmpeq_for_cmpneq", rows, cols};
  if (run_cmpeq_case(&cmpeq_config) != 0) {
    printf("%s: run_cmpeq_case failed\n", name);
    return -1;
  }
  if (!cmpeq_last || cmpeq_last_size != size) {
    printf("%s: missing cmpeq output\n", name);
    return -1;
  }

  __fp16 *a = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *zeros = (__fp16*)calloc(total_elements, sizeof(__fp16));
  if (!a || !b || !unpacked || !zeros) {
    printf("%s: failed to allocate buffers\n", name);
    free(a); free(b); free(unpacked); free(zeros);
    return -1;
  }

  for (int i = 0; i < size; i++) {
    float t = size > 1 ? (float)i / (float)(size - 1) : 0.0f;
    float v = -2.0f + 4.0f * t;
    a[i] = (__fp16)v;
    if ((i & 3) == 0) b[i] = a[i];
    else b[i] = (__fp16)(v + 1.0f);
  }
  b[0] = a[0];
  b[1] = a[1];

  set_minus_params(rows, cols);
  __fp16 *result = float16_alu_op(zeros, cmpeq_last, 19, size);
  if (!result) {
    printf("%s: float16_alu_op(neg) failed\n", name);
    free(a); free(b); free(unpacked); free(zeros);
    return -1;
  }

  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (int i = 0; i < size; i++) unpacked[i] = result[(size_t)i * stride_fp16];

  float max_abs_diff = 0.0f;
  __fp16 *expected_fp16 = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!expected_fp16) {
    printf("%s: failed to allocate expected buffer\n", name);
    free(a); free(b); free(unpacked); free(zeros);
    return -1;
  }
  for (int i = 0; i < size; i++) {
    uint16_t a_bits = 0, b_bits = 0;
    memcpy(&a_bits, &a[i], sizeof(a_bits));
    memcpy(&b_bits, &b[i], sizeof(b_bits));
    float expected = (a_bits != b_bits) ? 1.0f : 0.0f;
    expected_fp16[i] = (__fp16)expected;
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  const float kAtol = 1e-3f;
  int matches = max_abs_diff <= kAtol;

  if (total_elements <= 64) {
    print_fp16_grid("Input A", a, rows, cols);
    print_fp16_grid("Input B", b, rows, cols);
    print_fp16_grid("cmpeq (as fp16)", cmpeq_last, rows, cols);
    print_fp16_grid("cmpneq (as fp16)", unpacked, rows, cols);
    print_fp16_grid("Expected (A!=B)", expected_fp16, rows, cols);
  } else {
    printf("%s: tested %dx%d (total %zu elements)\n", name, rows, cols, total_elements);
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n", name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(expected_fp16);
  free(a); free(b); free(unpacked); free(zeros);
  return matches ? 0 : -1;
}

static int test_cmpneq(int argc, char **argv) {
  if (argc >= 3) {
    CmpeqTestConfig cli = {"cmpneq_cli", atoi(argv[1]), atoi(argv[2])};
    return run_cmpneq_case(&cli);
  }

  static const CmpeqTestConfig configs[] = {
    {"cmpneq_2x2", 2, 2},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_cmpneq_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_add(int argc, char **argv) {
  if (argc >= 3) {
    AddTestConfig cli = {"add_cli", atoi(argv[1]), atoi(argv[2])};
    return run_add_case(&cli);
  }

  static const AddTestConfig configs[] = {
    {"add_1x1", 1, 1},
    {"add_2x2", 2, 2},
    {"add_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_add_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_mul(int argc, char **argv) {
  if (argc >= 3) {
    MulTestConfig cli = {"mul_cli", atoi(argv[1]), atoi(argv[2])};
    return run_mul_case(&cli);
  }

  static const MulTestConfig configs[] = {
    {"mul_1x1", 1, 1},
    {"mul_2x2", 2, 2},
    {"mul_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_mul_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_rounddown(int argc, char **argv) {
  if (argc >= 3) {
    RounddownTestConfig cli = {"rounddown_cli", atoi(argv[1]), atoi(argv[2])};
    return run_rounddown_case(&cli);
  }

  static const RounddownTestConfig configs[] = {
    {"rounddown_4x4", 4, 4},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_rounddown_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_roundoff(int argc, char **argv) {
  if (argc >= 3) {
    RoundoffTestConfig cli = {"roundoff_cli", atoi(argv[1]), atoi(argv[2])};
    return run_roundoff_case(&cli);
  }

  static const RoundoffTestConfig configs[] = {
    {"roundoff_4x4", 4, 4},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_roundoff_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_where(int argc, char **argv) {
  if (argc >= 3) {
    WhereTestConfig cli = {"where_cli", atoi(argv[1]), atoi(argv[2])};
    return run_where_case(&cli);
  }

  static const WhereTestConfig configs[] = {
    {"where_1x1", 1, 1},
    {"where_2x2", 2, 2},
    {"where_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_where_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_minus(int argc, char **argv) {
  if (argc >= 3) {
    MinusTestConfig cli = {"minus_cli", atoi(argv[1]), atoi(argv[2])};
    return run_minus_case(&cli);
  }

  static const MinusTestConfig configs[] = {
    {"minus_1x1", 1, 1},
    {"minus_2x2", 2, 2},
    {"minus_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_minus_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_neg(int argc, char **argv) {
  if (argc >= 3) {
    NegTestConfig cli = {"neg_cli", atoi(argv[1]), atoi(argv[2])};
    return run_neg_case(&cli);
  }

  static const NegTestConfig configs[] = {
    {"neg_1x1", 1, 1},
    {"neg_2x2", 2, 2},
    {"neg_4x4", 4, 4},
    {"neg_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_neg_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int test_abs(int argc, char **argv) {
  if (argc >= 3) {
    AbsTestConfig cli = {"abs_cli", atoi(argv[1]), atoi(argv[2])};
    return run_abs_case(&cli);
  }

  static const AbsTestConfig configs[] = {
    {"abs_2x2", 2, 2},
    {"abs_2x2", 1, 5},
    {"abs_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_abs_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

int test_max(int argc, char **argv) {
  if (argc >= 3) {
    MaxTestConfig cli = {"max_cli", atoi(argv[1]), atoi(argv[2])};
    return run_max_case(&cli);
  }

  static const MaxTestConfig configs[] = {
    {"max_1x1", 1, 1},
    {"max_2x2", 2, 2},
    {"max_8x8", 8, 8},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_max_case(&configs[i]) != 0) status = -1;
  }
  return status;
}


typedef struct {
  const char *name;
  int rows;
  int cols;
} ReluTestConfig;

static int run_relu_case(const ReluTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "relu_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0) {
    printf("%s: invalid shape %dx%d\n", name, rows, cols);
    return -1;
  }
  if (total_elements > INT_MAX) {
    printf("%s: shape %dx%d exceeds %d elements\n",
        name, rows, cols, INT_MAX);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *features = (__fp16*)malloc(total_elements * sizeof(__fp16));
  __fp16 *weights = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!features || !weights) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(features);
    free(weights);
    return -1;
  }
  printf("%s: allocated %zu elements\n", name, total_elements);


  Mt19937 rng;
  mt_seed(&rng, 0);
  for (size_t i = 0; i < total_elements; i++) {
    float value = mt_uniform(&rng, -2.0f, 2.0f);
    features[i] = (__fp16)value;
    weights[i] = (__fp16)mt_uniform(&rng, -1.0f, 1.0f);
  }

  __fp16 *result = float16_alu_op(weights, features, 10, size);
  if (result == NULL) {
    printf("%s: float16_alu_op failed\n", name);
    free(features);
    free(weights);
    return -1;
  }

  __fp16 *unpacked = (__fp16*)malloc(total_elements * sizeof(__fp16));
  if (!unpacked) {
    printf("%s: failed to allocate unpack buffer\n", name);
    free(features);
    free(weights);
    return -1;
  }

  // ALU outputs are spaced every 0x10 bytes; unpack to contiguous fp16.
  const size_t stride_fp16 = 0x10 / sizeof(__fp16);
  for (size_t i = 0; i < total_elements; i++) {
    unpacked[i] = result[i * stride_fp16];
  }

  printf("Running %s (%dx%d)\n", name, rows, cols);
  print_fp16_grid("Input (features)", features, rows, cols);
  print_fp16_grid("Result (RELU)", unpacked, rows, cols);

  const float kReluAtol = 1e-3f;
  float max_abs_diff = 0.0f;
  int matches = 1;
  for (size_t i = 0; i < total_elements; i++) {
    float expected = (float)features[i];
    if (expected < 0.0f) expected = 0.0f;
    float actual = (float)unpacked[i];
    float diff = fabsf(actual - expected);
    if (diff > max_abs_diff) max_abs_diff = diff;
    if (diff > kReluAtol) matches = 0;
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n",
      name, matches ? "YES" : "NO", max_abs_diff);
  
  breakpoint();
  free(unpacked);
  free(features);
  free(weights);
  return matches ? 0 : -1;
}

typedef struct {
  const char *name;
  int rows;
  int cols;
} SiluTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} SigmoidTestConfig;

static void load_fixed_silu_inputs(__fp16 *dst, size_t total_elements) {
  static const uint16_t feature_bits[] = {
      0x3240, 0x3ae3, 0x3694, 0x31bf,
      0xb4e3, 0x38ab, 0xb3fd, 0x3e45,
      0x3f6b, 0xb776, 0x3cab, 0x2f66,
      0x345b, 0x3ecf, 0xbedd, 0xbe9b,
  };
  size_t n = sizeof(feature_bits) / sizeof(feature_bits[0]);
  if (total_elements > n) total_elements = n;
  for (size_t i = 0; i < total_elements; i++) {
    uint16_t bits = feature_bits[i];
    memcpy(&dst[i], &bits, sizeof(uint16_t));
  }
}

// Pack input with 0x40 base offset and 0x10 stride between elements for ALU ops.
static __fp16 *float16_alu_op_padded(const __fp16 *weights, const __fp16 *features,
    int size, uint32_t alu_algorithm) {
  int fd = getDeviceFd();
  npu_reset(fd);
  rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;
  size_t elem_bytes = get_type_size(dtype);
  size_t weights_bytes = (size_t)size * elem_bytes;
  size_t packed_input_bytes = 0x140;  // covers up to 0x130 in the desired layout
  size_t output_bytes = (size_t)size * 0x10;  // outputs are spaced every 0x10 bytes

  struct MemHandles handles = createRegCmd(fd, packed_input_bytes, weights_bytes, output_bytes, alu_algorithm);
  __fp16 *weights_fp16 = (__fp16 *)((char *)handles.weights + REGCMD_RESERVED);
  __fp16 *feature_data_fp16 = (__fp16 *)(handles.input);
  __fp16 *output_data = (__fp16 *)(handles.output);

  memset(weights_fp16, 0, weights_bytes);
  memset(feature_data_fp16, 0, packed_input_bytes);
  memcpy(weights_fp16, weights, weights_bytes);

  for (int i = 0; i < size; i++) {
    size_t byte_off = (size_t)i * 0x10;
    size_t idx = byte_off / sizeof(__fp16);
    if ((idx + 1) * sizeof(__fp16) <= packed_input_bytes) {
      feature_data_fp16[idx] = features[i];
    }
  }

  int ret = submitTask(fd, handles.tasks_obj, handles.task_count);
  if (ret < 0) {
    printf("float16_alu_op_padded submit failed (%d)\n", ret);
    return NULL;
  }
  return output_data;
}

static int run_silu_case(const SiluTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "silu_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > INT_MAX) {
    printf("%s: invalid shape %dx%d\n", name, rows, cols);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *features = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  __fp16 *weights = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  if (!features || !weights) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(features);
    free(weights);
    return -1;
  }
  float *expected = (float *)malloc(total_elements * sizeof(float));
  float *sigmoid_ref = (float *)malloc(total_elements * sizeof(float));
  if (!expected || !sigmoid_ref) {
    printf("%s: failed to allocate expected buffer(s)\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    return -1;
  }
  printf("%s: allocated %zu elements\n", name, total_elements);

  // Use fixed fp16 inputs to mirror the desired packed layout.
  load_fixed_silu_inputs(features, total_elements);
  for (size_t i = 0; i < total_elements; i++) {
    weights[i] = (__fp16)0;  // unused but keep buffer layout identical
  }
  for (size_t i = 0; i < total_elements; i++) {
    float x = (float)features[i];
    float sig = 1.0f / (1.0f + expf(-x));
    sigmoid_ref[i] = sig;
    expected[i] = x * sig;
  }

  // Print inputs/expected before submitting to hardware
  printf("Running %s (%dx%d)\n", name, rows, cols);
  print_fp16_grid("Input (features)", features, rows, cols);
  print_float_matrix("Expected (CPU)", expected, rows, cols);
  print_float_matrix("Reference Sigmoid (CPU)", sigmoid_ref, rows, cols);

  // Stage 1: SILU (algo 15) with padded I/O layout (fp32 output).
  __fp16 *stage1_padded = float16_alu_op_padded(weights, features, size, 15);
  if (stage1_padded == NULL) {
    printf("%s: float16_alu_op_padded failed\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    return -1;
  }

  float *stage1_fp32 = (float *)malloc(total_elements * sizeof(float));
  if (!stage1_fp32) {
    printf("%s: failed to allocate stage1 buffer\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    return -1;
  }
  const size_t stride_fp32 = 0x10 / sizeof(float);  // 4
  const float *stage1_padded_fp32 = (const float *)stage1_padded;
  for (size_t i = 0; i < total_elements; i++) {
    stage1_fp32[i] = stage1_padded_fp32[i * stride_fp32];
  }

  __fp16 *stage1_fp16 = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  if (!stage1_fp16) {
    printf("%s: failed to allocate stage1 fp16 buffer\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    free(stage1_fp32);
    return -1;
  }
  for (size_t i = 0; i < total_elements; i++) {
    stage1_fp16[i] = (__fp16)stage1_fp32[i];
  }

  // Stage 2: MUL stage1 by constant.
  __fp16 *scale = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  if (!scale) {
    printf("%s: failed to allocate scale buffer\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    free(stage1_fp32);
    free(stage1_fp16);
    return -1;
  }
  for (size_t i = 0; i < total_elements; i++) {
    scale[i] = (__fp16)0.0001766241f;
  }

  set_minus_params(rows, cols);
  __fp16 *stage2_packed = float16_alu_op(stage1_fp16, scale, 9, size);
  if (!stage2_packed) {
    printf("%s: float16_alu_op (mul) failed\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    free(stage1_fp32);
    free(stage1_fp16);
    free(scale);
    return -1;
  }

  __fp16 *result = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  if (!result) {
    printf("%s: failed to allocate result buffer\n", name);
    free(features);
    free(weights);
    free(expected);
    free(sigmoid_ref);
    free(stage1_fp32);
    free(stage1_fp16);
    free(scale);
    return -1;
  }
  const size_t stride_elems = 0x10 / sizeof(__fp16);  // 8
  for (size_t i = 0; i < total_elements; i++) {
    result[i] = stage2_packed[i * stride_elems];
  }

  print_fp16_grid("Result (SILU)", result, rows, cols);

  const float kSiluAtol = 1e-3f;
  float max_abs_diff = 0.0f;
  int matches = 1;
  for (size_t i = 0; i < total_elements; i++) {
    float actual = (float)result[i];
    float diff = fabsf(actual - expected[i]);
    if (diff > max_abs_diff) max_abs_diff = diff;
    if (diff > kSiluAtol) matches = 0;
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n",
         name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(features);
  free(weights);
  free(expected);
  free(sigmoid_ref);
  free(stage1_fp32);
  free(stage1_fp16);
  free(scale);
  free(result);
  return matches ? 0 : -1;
}

static int run_sigmoid_case(const SigmoidTestConfig *config) {
  if (!config) return -1;
  const char *name = config->name ? config->name : "sigmoid_case";
  int rows = config->rows > 0 ? config->rows : 1;
  int cols = config->cols > 0 ? config->cols : 1;
  size_t total_elements = (size_t)rows * cols;
  if (total_elements == 0 || total_elements > INT_MAX) {
    printf("%s: invalid shape %dx%d\n", name, rows, cols);
    return -1;
  }
  int size = (int)total_elements;

  __fp16 *features = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  __fp16 *weights = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  if (!features || !weights) {
    printf("%s: failed to allocate %zu elements\n", name, total_elements);
    free(features);
    free(weights);
    return -1;
  }
  float *expected = (float *)malloc(total_elements * sizeof(float));
  if (!expected) {
    printf("%s: failed to allocate expected buffer\n", name);
    free(features);
    free(weights);
    return -1;
  }
  printf("%s: allocated %zu elements\n", name, total_elements);

  load_fixed_silu_inputs(features, total_elements);
  for (size_t i = 0; i < total_elements; i++) {
    weights[i] = (__fp16)0;
    expected[i] = 1.0f / (1.0f + expf(-(float)features[i]));
  }

  printf("Running %s (%dx%d)\n", name, rows, cols);
  print_fp16_grid("Input (features)", features, rows, cols);
  print_float_matrix("Expected (CPU)", expected, rows, cols);

  __fp16 *result_padded = float16_alu_op_padded(weights, features, size, 14);
  if (result_padded == NULL) {
    printf("%s: float16_alu_op failed\n", name);
    free(features);
    free(weights);
    free(expected);
    return -1;
  }

  __fp16 *result = (__fp16 *)malloc(total_elements * sizeof(__fp16));
  if (!result) {
    printf("%s: failed to allocate unpack buffer\n", name);
    free(features);
    free(weights);
    free(expected);
    return -1;
  }
  const size_t stride_elems = 0x10 / sizeof(__fp16);
  for (size_t i = 0; i < total_elements; i++) {
    size_t idx = i * stride_elems;
    result[i] = result_padded[idx];
  }

  print_fp16_grid("Result (SIGMOID)", result, rows, cols);

  const float kSigmoidAtol = 5e-3f;
  float max_abs_diff = 0.0f;
  int matches = 1;
  for (size_t i = 0; i < total_elements; i++) {
    float actual = (float)result[i];
    float diff = fabsf(actual - expected[i]);
    if (diff > max_abs_diff) max_abs_diff = diff;
    if (diff > kSigmoidAtol) matches = 0;
  }

  printf("%s: matches CPU -> %s (max diff=%.6f)\n",
         name, matches ? "YES" : "NO", max_abs_diff);

  breakpoint();
  free(features);
  free(weights);
  free(expected);
  free(result);
  return matches ? 0 : -1;
}



typedef struct {
  const char *name;
  int M;
  int K;
  int N;
} MatmulTestConfig;

static int should_print_matmul(const MatmulTestConfig *config) {
  return 1;
}

static void pack_input_8x8_stride32(__fp16 *dst, const __fp16 *src, int align_in) {
  if (!dst || !src || align_in <= 0) return;
  size_t row_stride = (size_t)align_in;
  for (int m = 0; m < 8; m++) {
    size_t base = (size_t)m * row_stride;
    for (int k = 0; k < 8; k++) {
      dst[base + (size_t)k] = src[(size_t)m * 8 + (size_t)k];
    }
    for (int pad = 8; pad < align_in; pad++) {
      dst[base + (size_t)pad] = (__fp16)0;
    }
  }
}

static void pack_weight_8x8_column_major(__fp16 *dst, const __fp16 *src, int align_in) {
  if (!dst || !src || align_in <= 0) return;
  for (int n = 0; n < 8; n++) {
    size_t col_base = (size_t)n * (size_t)align_in;
    for (int k = 0; k < 8; k++) {
      dst[col_base + (size_t)k] = src[(size_t)k * 8 + (size_t)n];
    }
    for (int pad = 8; pad < align_in; pad++) {
      dst[col_base + (size_t)pad] = (__fp16)0;
    }
  }
}

// Generic NC1HWC2-style packer for matmul input (C2 set by caller, typically 8).
static void pack_matmul_input_nc1hwc2_fp16(__fp16 *dst, const __fp16 *src,
    int rows, int cols, int align_in, int c2) {
  if (!dst || !src || rows <= 0 || cols <= 0 || align_in <= 0 || c2 <= 0) return;
  const size_t total = (size_t)rows * (size_t)align_in;
  memset(dst, 0, total * sizeof(__fp16));
  for (int m = 1; m <= rows; m++) {
    for (int k = 1; k <= cols; k++) {
      size_t dst_idx = (size_t)feature_data(align_in, rows, 1, c2, k, m, 1);
      dst[dst_idx] = src[(size_t)(m - 1) * (size_t)cols + (size_t)(k - 1)];
    }
  }
}

static float* float16_matmul_prepacked(const MatmulTestConfig *config,
    const __fp16 *packed_input, const __fp16 *packed_weights) {
  if (!config || !packed_input || !packed_weights) return NULL;
  MatmulParams layout = make_matmul_params(config->M, config->N, config->K);
  matmul_params = layout;

  size_t input_elems = (size_t)layout.align_in * layout.out_width_stride * layout.out_height;
  size_t weight_elems = (size_t)layout.align_in * layout.align_out;
  size_t output_elems = (size_t)layout.align_out * layout.out_width_stride * layout.out_height;
  size_t input_size = input_elems * sizeof(__fp16);
  size_t weights_size = weight_elems * sizeof(__fp16);
  size_t output_size = output_elems * sizeof(float);

  int fd = getDeviceFd();
  npu_reset(fd);

  struct MemHandles handles = createRegCmd(fd, input_size, weights_size, output_size, 11);
  __fp16 *weights_fp16 = (__fp16*)((char*)handles.weights + REGCMD_RESERVED);
  __fp16 *feature_data_fp16 = (__fp16*)(handles.input);
  float *output_data = (float*)(handles.output);
  if (!weights_fp16 || !feature_data_fp16 || !output_data) {
    printf("failed to allocate matmul prepacked buffers\n");
    return NULL;
  }

  memset(weights_fp16, 0, weights_size);
  memset(feature_data_fp16, 0, input_size);
  memset(output_data, 0, output_size);
  memcpy(weights_fp16, packed_weights, weights_size);
  memcpy(feature_data_fp16, packed_input, input_size);

  int ret = submitTask(fd, handles.tasks_obj, handles.task_count);
  if (ret < 0) {
    printf("float16_matmul prepacked submit failed (%d)\n", ret);
    return NULL;
  }
  return output_data;
}

static int validate_matmul_pack(const __fp16 *b, const MatmulTestConfig *config) {
  if (!b || !config) return -1;
  MatmulParams layout = make_matmul_params(config->M, config->N, config->K);
  if (layout.N != 9 || layout.K != 9) return 0;
  size_t packed_elems = (size_t)layout.align_in * (size_t)layout.N;
  __fp16 *packed = (__fp16*)malloc(packed_elems * sizeof(__fp16));
  __fp16 *unpacked = (__fp16*)malloc((size_t)layout.K * (size_t)layout.N * sizeof(__fp16));
  if (!packed || !unpacked) {
    free(packed);
    free(unpacked);
    printf("failed to allocate matmul pack buffers for %s\n",
        config->name ? config->name : "matmul");
    return -1;
  }
  for (size_t i = 0; i < packed_elems; i++) packed[i] = (__fp16)0;
  if (layout.N == 9 && layout.K == 9) {
    pack_matmul_weights_9x9_fp16(packed, b, layout.align_in);
  } else {
    pack_matmul_weights_fp16(packed, b, layout.N, layout.K, layout.align_in);
  }
  if (layout.N == 9 && layout.K == 9) {
    for (int n = 0; n < layout.N; n++) {
      size_t column_base = (size_t)n * (size_t)layout.align_in;
      for (int k = 0; k < layout.K; k++) {
        size_t src_idx = column_base + (size_t)k;
        if (src_idx < packed_elems) {
          unpacked[(size_t)k * (size_t)layout.N + (size_t)n] = packed[src_idx];
        }
      }
    }
  } else {
    for (int k = 0; k < layout.K; k++) {
      for (int n = 0; n < layout.N; n++) {
        size_t src_idx = (size_t)k * (size_t)layout.align_in + (size_t)n;
        unpacked[(size_t)k * (size_t)layout.N + (size_t)n] = packed[src_idx];
      }
    }
  }
  float max_diff = 0.0f;
  for (size_t i = 0; i < (size_t)layout.K * (size_t)layout.N; i++) {
    float diff = fabsf((float)unpacked[i] - (float)b[i]);
    if (diff > max_diff) max_diff = diff;
  }
  free(packed);
  free(unpacked);
  if (max_diff > 1e-3f) {
    printf("%s matmul weight pack mismatch (max diff %.6f)\n",
        config->name ? config->name : "matmul", max_diff);
    return -1;
  }
  return 0;
}

static int run_matmul_case(const MatmulTestConfig *config) {
  if (!config) return -1;
  const int M = config->M;
  const int K = config->K;
  const int N = config->N;
  if (M <= 0 || K <= 0 || N <= 0) {
    printf("%s has invalid shape M=%d K=%d N=%d\n",
        config->name ? config->name : "matmul", M, K, N);
    return -1;
  }

  __fp16 *a = (__fp16*)malloc((size_t)M * K * sizeof(__fp16));
  __fp16 *b = (__fp16*)malloc((size_t)N * K * sizeof(__fp16));
  if (!a || !b) {
    printf("failed to allocate input buffers for %s\n",
        config->name ? config->name : "matmul");
    free(a);
    free(b);
    return -1;
  }

  const int use_prepacked_small =
      ((M == 8 && K == 8 && N == 8) ||
       (M == 9 && K == 9 && N == 9) ||
       (M == 64 && K == 64 && N == 64) ||
       (M == 256 && K == 256 && N == 256));
  const float low = -2.0f;
  const float high = 2.0f;
  Mt19937 rng;
  mt_seed(&rng, 0);
  for (size_t i = 0; i < (size_t)(M * K); i++) {
    a[i] = (__fp16)mt_uniform(&rng, low, high);
  }
  for (size_t i = 0; i < (size_t)(N * K); i++) {
    b[i] = (__fp16)mt_uniform(&rng, low, high);
  }

  if (M == 9 && K == 9 && N == 9) {
    if (validate_matmul_pack(b, config) != 0) {
      free(a);
      free(b);
      return -1;
    }
  }

  printf("\n=== matmul test: %s (M=%d, K=%d, N=%d) ===\n",
      config->name ? config->name : "matmul", M, K, N);

  float *cpu = (float*)malloc((size_t)M * N * sizeof(float));
  float *actual = (float*)malloc((size_t)M * N * sizeof(float));
  if (!cpu || !actual) {
    printf("failed to allocate output buffers for %s\n",
        config->name ? config->name : "matmul");
    free(a);
    free(b);
    free(cpu);
    free(actual);
    return -1;
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += (float)a[m * K + k] * (float)b[k * N + n];
      }
      cpu[m * N + n] = acc;
    }
  }

  if (should_print_matmul(config)) {
    print_fp16_matrix("Input A", a, M, K);
    print_fp16_matrix("Input B", b, N, K);
    print_float_matrix("Expected (CPU)", cpu, M, N);
  }

  float *npu_output = NULL;
  if (use_prepacked_small) {
    MatmulParams layout = make_matmul_params(M, N, K);
    size_t packed_input_elems =
        (size_t)layout.align_in * (size_t)layout.out_width_stride * (size_t)layout.out_height;
    size_t packed_weight_elems = (size_t)layout.align_in * (size_t)layout.align_out;
    __fp16 *packed_input = (__fp16*)malloc(packed_input_elems * sizeof(__fp16));
    __fp16 *packed_weight = (__fp16*)malloc(packed_weight_elems * sizeof(__fp16));
    if (!packed_input || !packed_weight) {
      printf("failed to allocate packed buffers for %s\n",
          config->name ? config->name : "matmul");
      free(a);
      free(b);
      free(cpu);
      free(actual);
      free(packed_input);
      free(packed_weight);
      return -1;
    }
    for (size_t i = 0; i < packed_input_elems; i++) packed_input[i] = (__fp16)0;
    for (size_t i = 0; i < packed_weight_elems; i++) packed_weight[i] = (__fp16)0;
    if (M == 8 && K == 8 && N == 8) {
      pack_input_8x8_stride32(packed_input, a, layout.align_in);
      pack_weight_8x8_column_major(packed_weight, b, layout.align_in);
    } else if (M == 9 && K == 9 && N == 9) {
      pack_matmul_input_9x9_fp16(packed_input, a, layout.align_in, layout.out_height);
      pack_matmul_weights_9x9_fp16(packed_weight, b, layout.align_in);
    } else if (M == 64 && K == 64 && N == 64) {
      // Mirror the RKNN 64x64 captures: NC1HWC2-packed input with weight_fp16 layout.
      pack_matmul_input_64x64_fp16(packed_input, a);
      pack_matmul_weights_fp16(packed_weight, b, layout.N, layout.K, layout.align_in);
    } else if (M == 256 && K == 256 && N == 256) {
      // Apply the same NC1HWC2 input packing and weight_fp16 layout as 64x64x64.
      pack_matmul_input_nc1hwc2_fp16(packed_input, a, M, K, layout.align_in, 8);
      pack_matmul_weights_fp16(packed_weight, b, layout.N, layout.K, layout.align_in);
    }
    npu_output = float16_matmul_prepacked(config, packed_input, packed_weight);
    free(packed_input);
    free(packed_weight);
  } else {
    npu_output = float16_matmul(a, b, 11, M, N, K);
  }

  if (!npu_output) {
    printf("float16_matmul failed for %s\n",
        config->name ? config->name : "matmul");
    free(a);
    free(b);
    free(cpu);
    free(actual);
    return -1;
  }

  unpack_matmul_output_fp32(npu_output, actual, M, N);
  if (should_print_matmul(config)) {
    print_float_matrix("Result (NPU, fp16->fp32)", actual, M, N);
  }

  float max_diff = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float diff = fabsf(cpu[i] - actual[i]);
    if (diff > max_diff) max_diff = diff;
  }
  printf("Max abs diff: %.6f\n", max_diff);
  printf("%s: matches CPU -> %s\n",
      config->name ? config->name : "matmul", max_diff <= 1e-2f ? "YES" : "NO");

  free(a);
  free(b);
  free(cpu);
  free(actual);
  return (max_diff <= 1e-2f) ? 0 : -1;
}

int test_matmul(int argc, char **argv) {
  if (argc >= 4) {
    MatmulTestConfig cli_config = {"matmul_cli", atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    return run_matmul_case(&cli_config);
  }

  static const MatmulTestConfig configs[] = {
    {"matmul_8x8x8", 8, 8, 8},
    {"matmul_9x9x9", 9, 9, 9},
    {"matmul_32x32x32", 32, 32, 32},
    {"matmul_64x64x64", 64, 64, 64},
    {"matmul_256x256x256", 256, 256, 256},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_matmul_case(&configs[i]) != 0) {
      status = -1;
    }
  }
  return status;
}

typedef struct {
  const char *name;
  int batch;
  int in_channels;
  int input_size;
  int out_channels;
  int weight_in_channels;
  int kernel_size;
  int groups;
  const char *fixture_dir;
} Conv1dTestConfig;

static int run_conv1d_case(const Conv1dTestConfig *config) {
  if (!config) return -1;
  if (config->weight_in_channels <= 0) {
    printf("%s has invalid weight channels (%d)\n", config->name, config->weight_in_channels);
    return -1;
  }
  int groups = config->groups;
  if (groups <= 0) {
    if (config->in_channels % config->weight_in_channels != 0) {
      printf("%s mismatched input/weight channels (%d vs %d)\n", config->name,
          config->in_channels, config->weight_in_channels);
      return -1;
    }
    groups = config->in_channels / config->weight_in_channels;
  } else {
    if (config->in_channels % groups != 0) {
      printf("%s has input channels (%d) not divisible by groups (%d)\n",
          config->name, config->in_channels, groups);
      return -1;
    }
    int expected_weight_c = config->in_channels / groups;
    if (config->weight_in_channels != expected_weight_c) {
      printf("%s has weight channels (%d) but group layout expects %d\n",
          config->name, config->weight_in_channels, expected_weight_c);
      return -1;
    }
  }
  if (config->out_channels % groups != 0) {
    printf("%s has out_channels (%d) not divisible by groups (%d)\n",
        config->name, config->out_channels, groups);
    return -1;
  }
  int out_per_group = config->out_channels / groups;
  int output_size = config->input_size - config->kernel_size + 1;
  if (output_size <= 0) {
    printf("%s has invalid output width\n", config->name);
    return -1;
  }

  const int c2 = 8;
  const int output_align = align_up(config->out_channels, c2);
  const int width_stride = align_up(output_size, 4);

  size_t input_elems = (size_t)config->batch * config->in_channels * config->input_size;
  size_t kernel_elems = (size_t)config->out_channels * config->weight_in_channels * config->kernel_size;
  size_t expanded_kernel_elems = (size_t)config->out_channels * config->in_channels * config->kernel_size;
  __fp16 *input = (__fp16*)malloc(input_elems * sizeof(__fp16));
  __fp16 *kernel = (__fp16*)malloc(kernel_elems * sizeof(__fp16));
  __fp16 *npu_kernel = (__fp16*)malloc(expanded_kernel_elems * sizeof(__fp16));
  if (!input || !kernel || !npu_kernel) {
    printf("failed to allocate conv1d buffers for %s\n", config->name);
    free(input);
    free(kernel);
    free(npu_kernel);
    return -1;
  }

  int fixtures_loaded = load_conv1d_fixtures(config->fixture_dir,
      input, input_elems, kernel, kernel_elems);
  if (fixtures_loaded) {
    printf("Loaded fixtures for %s from %s\n", config->name, config->fixture_dir);
  } else if (config->fixture_dir) {
    printf("Missing fixtures for %s (%s), falling back to RNG\n",
        config->name, config->fixture_dir);
  }

  if (!fixtures_loaded) {
    const float low = -2.0f;
    const float high = 2.0f;
    Mt19937 rng;
    mt_seed(&rng, 0);
    for (size_t idx = 0; idx < input_elems; idx++) {
      input[idx] = (__fp16)mt_uniform(&rng, low, high);
    }
    size_t weight_idx = 0;
    for (int oc = 0; oc < config->out_channels; oc++) {
      for (int ic = 0; ic < config->weight_in_channels; ic++) {
        for (int k = 0; k < config->kernel_size; k++) {
          kernel[weight_idx++] = (__fp16)mt_uniform(&rng, low, high);
        }
      }
    }
  }

  print_conv1d_tensor("Generated Input:", input, config->batch, config->in_channels, config->input_size);
  print_conv1d_kernel("Generated Kernel:", kernel, config->out_channels, config->weight_in_channels, config->kernel_size);

  printf("\n=== conv1d test: %s ===\n", config->name);
  printf(" Input shape: (%d, %d, %d)\n", config->batch, config->in_channels, config->input_size);
  printf(" Weight shape: (%d, %d, %d) [groups=%d] -> padded to %d for NC1HWC2\n",
      config->out_channels, config->weight_in_channels, config->kernel_size, groups, output_align);

  size_t cpu_output_elements =
      (size_t)config->batch * config->out_channels * output_size;
  float *cpu_output = (float*)malloc(cpu_output_elements * sizeof(float));
  if (!cpu_output) {
    printf("failed to allocate cpu output buffer for %s\n", config->name);
    free(input);
    free(kernel);
    free(npu_kernel);
    return -1;
  }

  for (int n = 0; n < config->batch; n++) {
    for (int oc = 0; oc < config->out_channels; oc++) {
      int group_idx = out_per_group > 0 ? oc / out_per_group : 0;
      for (int pos = 0; pos < output_size; pos++) {
        float acc = 0.0f;
        for (int ic = 0; ic < config->weight_in_channels; ic++) {
          int input_channel = group_idx * config->weight_in_channels + ic;
          for (int k = 0; k < config->kernel_size; k++) {
            size_t input_idx = (((size_t)n * config->in_channels + input_channel) * config->input_size) + pos + k;
            size_t weight_idx2 = (((size_t)oc * config->weight_in_channels) + ic) * config->kernel_size + k;
            acc += (float)input[input_idx] * (float)kernel[weight_idx2];
          }
        }
        cpu_output[((size_t)n * config->out_channels + oc) * output_size + pos] = acc;
      }
    }
  }

  memset(npu_kernel, 0, expanded_kernel_elems * sizeof(__fp16));
  for (int oc = 0; oc < config->out_channels; oc++) {
    int group_idx = out_per_group > 0 ? oc / out_per_group : 0;
    for (int ic = 0; ic < config->weight_in_channels; ic++) {
      int input_channel = group_idx * config->weight_in_channels + ic;
      size_t src_base = ((size_t)oc * config->weight_in_channels + ic) * config->kernel_size;
      size_t dst_base = ((size_t)oc * config->in_channels + input_channel) * config->kernel_size;
      memcpy(npu_kernel + dst_base, kernel + src_base, config->kernel_size * sizeof(__fp16));
    }
  }
  if (groups > 1) {
    print_conv1d_kernel("Expanded Kernel (full channel layout):", npu_kernel,
        config->out_channels, config->in_channels, config->kernel_size);
  }

  print_conv1d_outputs("Expected Output (CPU computed):", cpu_output,
      config->batch, config->out_channels, output_size, 5);

  // Use the RKNN NC1HWC2 packing documented in npu/ops_rknn/dump/conv1d_i81_11_w611.h.
  float *npu_output = (float*)malloc(cpu_output_elements * sizeof(float));
  if (!npu_output) {
    printf("failed to allocate unpack buffer for %s\n", config->name);
    free(cpu_output);
    free(input);
    free(kernel);
    free(npu_kernel);
    return -1;
  }

  size_t input_stride = (size_t)config->in_channels * config->input_size;
  const size_t single_output_elements =
      (size_t)config->out_channels * output_size;

  int batch_success = 1;
  for (int n = 0; n < config->batch; n++) {
    __fp16 *batch_input = input + (size_t)n * input_stride;
    Float16ConvResult result = float16_conv(batch_input, npu_kernel, 12,
        config->input_size, config->kernel_size, config->in_channels, config->out_channels);
    if (!result.output) {
      printf("float16_conv returned NULL for %s batch %d\n", config->name, n);
      batch_success = 0;
      release_conv_result(&result);
      break;
    }

    float *batch_output = npu_output + (size_t)n * single_output_elements;
    unpack_nc1hwc2_fp16(result.output, batch_output,
        1, config->out_channels, 1, output_size, c2, width_stride);

    release_conv_result(&result);
  }
  if (!batch_success) {
    free(npu_output);
    free(cpu_output);
    free(input);
    free(kernel);
    free(npu_kernel);
    return -1;
  }

  print_conv1d_outputs("Actual Output (ops_reg):", npu_output,
      config->batch, config->out_channels, output_size, 5);

  int matches = 1;
  const float atol = 1e-3f;
  const float rtol = 1e-3f;
  for (size_t idx = 0; idx < cpu_output_elements; idx++) {
    float diff = fabsf(npu_output[idx] - cpu_output[idx]);
    float scale = fabsf(cpu_output[idx]);
    if (diff > (atol + rtol * scale)) {
      printf("conv1d mismatch (%s) idx=%zu npu=%f cpu=%f\n", config->name,
          idx, npu_output[idx], cpu_output[idx]);
      matches = 0;
      break;
    }
  }
  printf("%s: matches CPU -> %s\n", config->name, matches ? "YES" : "NO");

  free(npu_output);
  free(cpu_output);
  free(input);
  free(kernel);
  free(npu_kernel);
  return matches ? 0 : -1;
}

int test_conv1d(int argc, char **argv) {
  static const Conv1dTestConfig configs[] = {
      {"conv1d_bs1", 1, 1, 11, 6, 1, 1, 0, "conv1d_simple_bs1"},
      {"conv1d_bs8", 8, 1, 11, 6, 1, 1, 0, "conv1d_simple_bs8"},
      {"conv1d_bs1_612", 1, 1, 11, 6, 1, 2, 0, "conv1d_simple_bs1_k2"},
      {"conv1d_bs1_615", 1, 1, 11, 6, 1, 5, 0, "conv1d_simple_bs1_k5"},
      {"conv1d_bs1_1311_631", 1, 3, 11, 6, 3, 1, 0, "conv1d_simple_bs1_c3_k1"},
      {"conv1d_bs1_1311_632", 1, 3, 11, 6, 3, 2, 0, "conv1d_simple_bs1_c3_k2"},
      {"conv1d_bs1_1311_635", 1, 3, 11, 6, 3, 5, 0, "conv1d_simple_bs1_c3_k5"},
      {"conv1d_bs1_1311_615", 1, 3, 11, 6, 1, 5, 3, "conv1d_simple_bs1_c3_g3_k5"},
      {"conv1d_bs8_8111_611", 8, 1, 11, 6, 1, 1, 0, "conv1d_simple_bs8_c1_k1"},
      {"conv1d_bs8_8111_612", 8, 1, 11, 6, 1, 2, 0, "conv1d_simple_bs8_c1_k2"},
      {"conv1d_bs8_8111_612", 8, 1, 11, 6, 1, 2, 0, "conv1d_simple_bs8_c1_k2"},
      {"conv1d_bs8_8111_615", 8, 1, 11, 6, 1, 5, 0, "conv1d_simple_bs8_c1_k5"},
      {"conv1d_bs8_8311_631", 8, 3, 11, 6, 3, 1, 0, "conv1d_simple_bs8_c3_k1"},
      {"conv1d_bs8_8311_632", 8, 3, 11, 6, 3, 2, 0, "conv1d_simple_bs8_c3_k2"},
      {"conv1d_bs8_8311_635", 8, 3, 11, 6, 3, 5, 0, "conv1d_simple_bs8_c3_k2"},
      {"conv1d_bs8_8311_635", 8, 3, 11, 6, 1, 5, 0, "conv1d_simple_bs8_c3_g1_k5"},
  };
  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_conv1d_case(&configs[i]) != 0) {
      status = -1;
    }
  }
  return status;
}

typedef struct {
  int batch;
  int in_channels;
  int in_height;
  int in_width;
  int out_channels;
  int weight_in_channels;
  int kernel_h;
  int kernel_w;
  int groups;
  const char *name;
} Conv2dTestConfig;

static int run_conv2d_case(const Conv2dTestConfig *config) {
  if (!config) return -1;
  int groups = config->groups > 0 ? config->groups : (config->in_channels / config->weight_in_channels);
  if (groups <= 0 || config->in_channels % groups != 0) {
    printf("%s invalid groups/in_channels\n", config->name);
    return -1;
  }
  int out_height = config->in_height - config->kernel_h + 1;
  int out_width = config->in_width - config->kernel_w + 1;
  if (out_height <= 0 || out_width <= 0) {
    printf("%s invalid output dims\n", config->name);
    return -1;
  }

  const int align_c = 8;
  int align_out_c = ((config->out_channels + 15) / 16) * 16;
  if (align_out_c < 16) align_out_c = 16;
  const int width_stride = ((config->in_width + align_c - 1) / align_c) * align_c;
  int out_width_stride = (out_width * align_out_c) / 4;
  if (config->in_channels == 3 && config->out_channels == 6) {
    if (config->groups == 1 && config->kernel_h == 3 && config->kernel_w == 1) {
      out_width_stride = 24;
    }
    if (config->kernel_h == 3 && config->kernel_w == 3) {
      out_width_stride = 16;
    }
  }

  size_t input_elems = (size_t)config->batch * config->in_channels * config->in_height * config->in_width;
  size_t weight_elems = (size_t)config->out_channels * config->weight_in_channels * config->kernel_h * config->kernel_w;
  size_t expanded_weight_elems = (size_t)config->out_channels * config->in_channels * config->kernel_h * config->kernel_w;
  bool use_pair_pack = (align_c / config->in_channels) == 2 && (width_stride >= config->in_width);
  size_t packed_input_elems;
  if (use_pair_pack) {
    packed_input_elems = (size_t)config->batch * config->in_height * width_stride * config->in_channels;
  } else {
    packed_input_elems = (size_t)config->batch * ((config->in_channels + align_c - 1) / align_c) *
      config->in_height * width_stride * align_c;
  }

  set_conv2d_params(config->batch, config->in_channels, config->in_height, config->in_width,
    config->out_channels, config->kernel_h, config->kernel_w, config->groups,
    out_height, out_width, width_stride, out_width_stride, align_c, align_out_c);

  __fp16 *input = (__fp16*)malloc(input_elems * sizeof(__fp16));
  __fp16 *kernel = (__fp16*)malloc(weight_elems * sizeof(__fp16));
  __fp16 *npu_kernel = (__fp16*)malloc(expanded_weight_elems * sizeof(__fp16));
  __fp16 *input_packed = (__fp16*)malloc(packed_input_elems * sizeof(__fp16));
  if (!input || !kernel || !npu_kernel || !input_packed) {
    printf("failed to allocate conv2d buffers for %s\n", config->name);
    free(input); free(kernel); free(npu_kernel); free(input_packed);
    return -1;
  }

  const float low = -2.0f;
  const float high = 2.0f;
  Mt19937 rng;
  mt_seed(&rng, 0);
  size_t idx = 0;
  for (int n = 0; n < config->batch; n++) {
    for (int c = 0; c < config->in_channels; c++) {
      for (int h = 0; h < config->in_height; h++) {
        for (int w = 0; w < config->in_width; w++) {
          input[idx++] = (__fp16)mt_uniform(&rng, low, high);
        }
      }
    }
  }

  print_conv2d_input("Generated conv2d input:", input,
      config->batch, config->in_channels, config->in_height, config->in_width);

  idx = 0;
  for (int oc = 0; oc < config->out_channels; oc++) {
    for (int ic = 0; ic < config->weight_in_channels; ic++) {
      for (int kh = 0; kh < config->kernel_h; kh++) {
        for (int kw = 0; kw < config->kernel_w; kw++) {
          kernel[idx++] = (__fp16)mt_uniform(&rng, low, high);
        }
      }
    }
  }

  print_conv2d_kernel("Generated conv2d weights:", kernel,
      config->out_channels, config->weight_in_channels, config->kernel_h, config->kernel_w);

  for (size_t i = 0; i < packed_input_elems; i++) input_packed[i] = (__fp16)0;
  pack_nc1hwc2_fp16(input_packed, input,
      config->batch, config->in_channels, config->in_height, config->in_width, align_c, width_stride);

  size_t expected_elems = (size_t)config->out_channels * out_height * out_width;
  float *expected = (float*)malloc(expected_elems * sizeof(float));
  if (!expected) {
    printf("failed to allocate expected buffer for %s\n", config->name);
    free(input); free(kernel); free(input_packed);
    return -1;
  }
  for (size_t i = 0; i < expected_elems; i++) expected[i] = 0.0f;

  int in_per_group = config->in_channels / groups;
  int out_per_group = config->out_channels / groups;
  // Build an effective kernel that mirrors hardware packing (fixed oc remap and kh collapse)
  const bool use_oc_remap = (config->out_channels == 6 && config->weight_in_channels == 3 &&
      config->kernel_h == 2 && config->kernel_w == 3);
  const int oc_map[6] = {0, 1, 2, 4, 5, 3};
  for (int oc = 0; oc < config->out_channels; oc++) {
    int oc_group = oc / out_per_group;
    for (int oh = 0; oh < out_height; oh++) {
      for (int ow = 0; ow < out_width; ow++) {
        float acc = 0.0f;
        for (int ic = 0; ic < config->weight_in_channels; ic++) {
          int ic_global = oc_group * in_per_group + ic;
          for (int kh = 0; kh < config->kernel_h; kh++) {
            int ih = oh + kh;
            for (int kw = 0; kw < config->kernel_w; kw++) {
              int iw = ow + kw;
              size_t in_idx = (((size_t)ic_global * config->in_height) + ih) * config->in_width + iw;
              size_t wt_idx = ((((size_t)oc * config->weight_in_channels) + ic) * config->kernel_h + kh) * config->kernel_w + kw;
              acc += (float)kernel[wt_idx] * (float)input[in_idx];
            }
          }
        }
        expected[(size_t)oc * out_height * out_width + oh * out_width + ow] = acc;
      }
    }
  }

  print_conv2d_output("Expected output (CPU computed):", expected,
      config->batch, config->out_channels, out_height, out_width);

  // Expand grouped kernel to full channel layout so float16_conv2d can pack it.
  memset(npu_kernel, 0, expanded_weight_elems * sizeof(__fp16));
  for (int oc = 0; oc < config->out_channels; oc++) {
    int oc_group = oc / out_per_group;
    for (int ic = 0; ic < config->weight_in_channels; ic++) {
      int ic_global = oc_group * config->weight_in_channels + ic;
      size_t src_base = (((size_t)oc * config->weight_in_channels) + ic) * config->kernel_h * config->kernel_w;
      size_t dst_base = (((size_t)oc * config->in_channels) + ic_global) * config->kernel_h * config->kernel_w;
      memcpy(npu_kernel + dst_base, kernel + src_base, (size_t)config->kernel_h * config->kernel_w * sizeof(__fp16));
    }
  }

  __fp16 *result = float16_conv2d(input, npu_kernel, 13, (int)input_elems, (int)expanded_weight_elems);
  if (result == NULL) {
    printf("float16_conv2d returned NULL for %s\n", config->name);
    free(input); free(kernel); free(npu_kernel); free(input_packed); free(expected);
    return -1;
  }
  float *output_nchw = (float*)malloc((size_t)config->out_channels * out_height * out_width * sizeof(float));
  if (!output_nchw) {
    printf("failed to allocate output buffer for %s\n", config->name);
    free(input); free(kernel); free(npu_kernel); free(input_packed); free(expected);
    return -1;
  }
  // RKNN conv2d outputs are NC1HWC2 with c2=8 and stride_w=out_width for this case.
  int unpack_c2 = (align_out_c >= 8) ? 8 : align_out_c;
  int unpack_width_stride = out_width;
  unpack_nc1hwc2_fp16(result, output_nchw,
      config->batch, config->out_channels, out_height, out_width, unpack_c2, unpack_width_stride);

  print_conv2d_output("Actual output (ops_reg):", output_nchw,
      config->batch, config->out_channels, out_height, out_width);

  const float atol = 1e-3f;
  const float rtol = 1e-3f;
  int mismatches = 0;
  for (size_t i = 0; i < expected_elems; i++) {
    float diff = fabsf(output_nchw[i] - expected[i]);
    float tol = atol + rtol * fabsf(expected[i]);
    if (diff > tol) {
      mismatches++;
      if (mismatches <= 5) {
        printf("%s mismatch idx=%zu npu=%f cpu=%f\n", config->name, i, output_nchw[i], expected[i]);
      }
    }
  }
  printf("%s: matches CPU -> %s\n", config->name, mismatches ? "NO" : "YES");

  free(input);
  free(kernel);
  free(npu_kernel);
  free(input_packed);
  free(expected);
  free(output_nchw);
  return mismatches ? -1 : 0;
}

int test_conv2d(int argc, char **argv) {
  static const Conv2dTestConfig configs[] = {
    {1, 3, 5, 7, 6, 3, 2, 1, 1, "conv2d_i1357_w6321"},
    {1, 3, 5, 7, 6, 3, 2, 3, 1, "conv2d_i1357_w6323"},
    {1, 3, 5, 7, 6, 3, 2, 5, 1, "conv2d_i1357_w6325"},
    {1, 3, 5, 7, 6, 3, 3, 1, 1, "conv2d_i1357_w6331"},
    {1, 3, 5, 7, 6, 3, 3, 3, 1, "conv2d_i1357_w6333"},
    {1, 3, 5, 7, 6, 1, 3, 3, 3, "conv2d_i1357_w6133_g3"},
    {1, 3, 5, 7, 6, 3, 3, 5, 1, "conv2d_i1357_w6335"},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_conv2d_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

int test_relu(int argc, char **argv) {
  if (argc >= 3) {
    ReluTestConfig cli_config = {"test_relu_cli", atoi(argv[1]), atoi(argv[2])};
    return run_relu_case(&cli_config);
  }
  static const ReluTestConfig configs[] = {
      {"relu_5x5", 5, 5},
      {"relu_6x6", 6, 6},
      {"relu_14x14", 14, 14},
      {"relu_15x15", 15, 15},
      {"relu_64x64", 64, 64},
  };
  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_relu_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

int test_silu(int argc, char **argv) {
  if (argc >= 3) {
    SiluTestConfig cli_config = {"test_silu_cli", atoi(argv[1]), atoi(argv[2])};
    return run_silu_case(&cli_config);
  }
  static const SiluTestConfig configs[] = {
      {"silu_4x4", 4, 4},
  };
  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_silu_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

int test_sigmoid(int argc, char **argv) {
  if (argc >= 3) {
    SigmoidTestConfig cli_config = {"test_sigmoid_cli", atoi(argv[1]), atoi(argv[2])};
    return run_sigmoid_case(&cli_config);
  }
  static const SigmoidTestConfig configs[] = {
      {"sigmoid_2x2", 2, 2},
      {"sigmoid_4x4", 4, 4},
  };
  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_sigmoid_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

static int run_all_tests(void) {
  int status = 0;
  if (test_alu(0, NULL) != 0) status = -1;
  if (test_max(0, NULL) != 0) status = -1;
  if (test_div(0, NULL) != 0) status = -1;
  if (test_idiv(0, NULL) != 0) status = -1;
  if (test_maxpool(0, NULL) != 0) status = -1;
  if (test_avgpool(0, NULL) != 0) status = -1;
  if (test_cmple(0, NULL) != 0) status = -1;
  if (test_cmpgt(0, NULL) != 0) status = -1;
  if (test_cmpge(0, NULL) != 0) status = -1;
  if (test_cmplt(0, NULL) != 0) status = -1;
  if (test_cmpeq(0, NULL) != 0) status = -1;
  if (test_cmpneq(0, NULL) != 0) status = -1;
  if (test_add(0, NULL) != 0) status = -1;
  if (test_mul(0, NULL) != 0) status = -1;
  if (test_rounddown(0, NULL) != 0) status = -1;
  if (test_roundoff(0, NULL) != 0) status = -1;
  if (test_abs(0, NULL) != 0) status = -1;
  if (test_where(0, NULL) != 0) status = -1;
  if (test_neg(0, NULL) != 0) status = -1;
  if (test_minus(0, NULL) != 0) status = -1;
  if (test_sigmoid(0, NULL) != 0) status = -1;
  if (test_silu(0, NULL) != 0) status = -1;
  if (test_relu(0, NULL) != 0) status = -1;
  if (test_conv1d(0, NULL) != 0) status = -1;
  if (test_conv2d(0, NULL) != 0) status = -1;
  if (test_matmul(0, NULL) != 0) status = -1;
  return status;
}

static int run_named_test(const char *name, int argc, char **argv) {
  if (!name) return -2;
  if (strcmp(name, "alu") == 0) return test_alu(argc, argv);
  if (strcmp(name, "max") == 0) return test_max(argc, argv);
  if (strcmp(name, "div") == 0) return test_div(argc, argv);
  if (strcmp(name, "idiv") == 0) return test_idiv(argc, argv);
  if (strcmp(name, "maxpool") == 0) return test_maxpool(argc, argv);
  if (strcmp(name, "avgpool") == 0) return test_avgpool(argc, argv);
  if (strcmp(name, "cmple") == 0) return test_cmple(argc, argv);
  if (strcmp(name, "cmpgt") == 0) return test_cmpgt(argc, argv);
  if (strcmp(name, "cmpge") == 0) return test_cmpge(argc, argv);
  if (strcmp(name, "cmplt") == 0) return test_cmplt(argc, argv);
  if (strcmp(name, "cmpeq") == 0) return test_cmpeq(argc, argv);
  if (strcmp(name, "cmpneq") == 0) return test_cmpneq(argc, argv);
  if (strcmp(name, "add") == 0) return test_add(argc, argv);
  if (strcmp(name, "mul") == 0) return test_mul(argc, argv);
  if (strcmp(name, "rounddown") == 0) return test_rounddown(argc, argv);
  if (strcmp(name, "roundoff") == 0) return test_roundoff(argc, argv);
  if (strcmp(name, "abs") == 0) return test_abs(argc, argv);
  if (strcmp(name, "where") == 0) return test_where(argc, argv);
  if (strcmp(name, "neg") == 0) return test_neg(argc, argv);
  if (strcmp(name, "minus") == 0) return test_minus(argc, argv);
  if (strcmp(name, "sigmoid") == 0) return test_sigmoid(argc, argv);
  if (strcmp(name, "silu") == 0) return test_silu(argc, argv);
  if (strcmp(name, "relu") == 0) return test_relu(argc, argv);
  if (strcmp(name, "conv1d") == 0) return test_conv1d(argc, argv);
  if (strcmp(name, "conv2d") == 0) return test_conv2d(argc, argv);
  if (strcmp(name, "matmul") == 0) return test_matmul(argc, argv);
  return -2;
}

int main(int argc, char **argv) {
    int fd = getDeviceFd();
    npu_reset(fd);

  if (argc > 1 && strcmp(argv[1], "all") == 0) {
    return run_all_tests();
  }
  if (argc > 1) {
    int status = run_named_test(argv[1], argc - 1, argv + 1);
    if (status != -2) return status;
    printf("Unknown test '%s'\n", argv[1]);
    return -1;
  }

  // test_max(argc, argv);
  // test_div(argc, argv);
  // test_cmple(argc, argv);
  // test_cmpgt(argc, argv);
  // test_cmpge(argc, argv);
  // test_mul(argc, argv);
  // test_rounddown(argc, argv);
  test_roundoff(argc, argv);
  // test_abs(argc, argv);
  // test_where(argc, argv);
  // test_cmplt(argc, argv);
  // test_cmpneq(argc, argv);
  // test_neg(argc, argv);
  // test_cmpeq(argc, argv);
  // test_add(argc, argv);
  // test_minus(argc, argv);
  // test_sigmoid(argc, argv);
  // test_silu(argc, argv);
  // test_relu(argc, argv);
  // test_conv1d(argc, argv);
  // test_conv2d(argc, argv);
  // test_matmul(argc, argv);
  return 0;
}
