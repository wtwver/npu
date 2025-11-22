#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
    __fp16* result = float16_matmul(a, b, 11, 32, 32, 32);
    printf("Input0: "); for (size_t i = 0; i < M*K ; i++) printf("%f ", a[i]); printf("\n");
    printf("Input1: "); for (size_t i = 0; i < N*K ; i++) printf("%f ", b[i]); printf("\n");
    printf("Result/Input0: "); for (size_t i = 0; i < M*N ; i++) printf("fp16: %f fp32: %f \n", (__fp16)result[i], result[i]); printf("\n");
    return 0;
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
  const int out_width_stride = (out_width * align_out_c) / 4;

  size_t input_elems = (size_t)config->batch * config->in_channels * config->in_height * config->in_width;
  size_t weight_elems = (size_t)config->out_channels * config->weight_in_channels * config->kernel_h * config->kernel_w;
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
  __fp16 *input_packed = (__fp16*)malloc(packed_input_elems * sizeof(__fp16));
  if (!input || !kernel || !input_packed) {
    printf("failed to allocate conv2d buffers for %s\n", config->name);
    free(input); free(kernel); free(input_packed);
    return -1;
  }

  size_t idx = 0;
  for (int n = 0; n < config->batch; n++) {
    for (int c = 0; c < config->in_channels; c++) {
      for (int h = 0; h < config->in_height; h++) {
        float hv = (float)((h + 1) * (h + 1));
        for (int w = 0; w < config->in_width; w++) {
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
  for (int oc = 0; oc < config->out_channels; oc++) {
    for (int ic = 0; ic < config->weight_in_channels; ic++) {
      for (int kh = 0; kh < config->kernel_h; kh++) {
        for (int kw = 0; kw < config->kernel_w; kw++) {
          int pattern = (oc + kh + kw) % 3;
          float val = (pattern == 0) ? 1.0f : ((pattern == 1) ? -1.0f : 0.0f);
          kernel[idx++] = (__fp16)val;
        }
      }
    }
  }

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

  __fp16 *result = float16_conv2d(input, kernel, 13, (int)input_elems, (int)weight_elems);
  if (result == NULL) {
    printf("float16_conv2d returned NULL for %s\n", config->name);
    free(input); free(kernel); free(input_packed); free(expected);
    return -1;
  }
  float *output_nchw = (float*)malloc((size_t)config->out_channels * out_height * out_width * sizeof(float));
  if (!output_nchw) {
    printf("failed to allocate output buffer for %s\n", config->name);
    free(input); free(kernel); free(input_packed); free(expected);
    return -1;
  }
  // RKNN conv2d outputs are NC1HWC2 with c2=8 and stride_w=out_width for this case.
  int unpack_c2 = (align_out_c >= 8) ? 8 : align_out_c;
  int unpack_width_stride = out_width;
  unpack_nc1hwc2_fp16(result, output_nchw,
      config->batch, config->out_channels, out_height, out_width, unpack_c2, unpack_width_stride);

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
  free(input_packed);
  free(expected);
  free(output_nchw);
  return mismatches ? -1 : 0;
}

int test_conv2d(int argc, char **argv) {
  static const Conv2dTestConfig configs[] = {
    // {1, 3, 5, 7, 6, 3, 2, 3, 1, "conv2d_i1357_w6323"},
    // {1, 3, 5, 7, 6, 3, 2, 5, 1, "conv2d_i1357_w6325"},
    // {1, 3, 5, 7, 6, 3, 3, 1, 1, "conv2d_i1357_w6331"},
    {1, 3, 5, 7, 6, 3, 3, 3, 1, "conv2d_i1357_w6333"},
    // {1, 3, 5, 7, 6, 1, 3, 3, 3, "conv2d_i1357_w6133_g3"},
    // {1, 3, 5, 7, 6, 3, 3, 5, 1, "conv2d_i1357_w6335"},
  };

  int status = 0;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
    if (run_conv2d_case(&configs[i]) != 0) status = -1;
  }
  return status;
}

int main(int argc, char **argv) {
    int fd = getDeviceFd();
    npu_reset(fd);

    // test_alu(argc, argv);
    // test_matmul(argc, argv);
    // test_conv1d(argc, argv);
    test_conv2d(argc, argv);
    return 0;
}
