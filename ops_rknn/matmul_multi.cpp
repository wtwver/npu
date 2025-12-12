#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

static const char* get_format_string(int fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
    case RKNN_TENSOR_UNDEFINED: return "UNDEFINED";
    default: return "UNKNOWN";
  }
}

static bool load_model(const std::string& path, std::vector<uint8_t>& data) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) {
    std::cerr << "Failed to open model file: " << path << std::endl;
    return false;
  }

  struct stat st;
  if (stat(path.c_str(), &st) != 0 || st.st_size <= 0) {
    std::cerr << "Failed to stat model file: " << path << std::endl;
    fclose(fp);
    return false;
  }

  data.resize(static_cast<size_t>(st.st_size));
  if (fread(data.data(), 1, data.size(), fp) != data.size()) {
    std::cerr << "Failed to read model file: " << path << std::endl;
    fclose(fp);
    return false;
  }

  fclose(fp);
  return true;
}

// MT19937 for deterministic inputs (matches other RKNN tests)
struct Mt19937 {
  uint32_t mt[624];
  int index;
};

static void mt_seed(Mt19937* rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; ++i) {
    rng->mt[i] = 1812433253U * (rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
  }
  rng->index = 624;
}

static uint32_t mt_extract(Mt19937* rng) {
  const uint32_t mag01[2] = {0U, 0x9908b0dfU};
  if (rng->index >= 624) {
    for (int kk = 0; kk < 624 - 397; ++kk) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk + 1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (int kk = 624 - 397; kk < 623; ++kk) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk + 1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk - (624 - 397)] ^ (y >> 1) ^ mag01[y & 1U];
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

static float mt_uniform(Mt19937* rng, float low, float high) {
  const double a = static_cast<double>(mt_extract(rng) >> 5);
  const double b = static_cast<double>(mt_extract(rng) >> 6);
  const double random = (a * 67108864.0 + b) / 9007199254740992.0;  // 2^53
  return static_cast<float>(low + (high - low) * random);
}

struct MatmulConfig {
  std::string model_name;
  int M;
  int N;
  int K;
};

template <typename T>
static void print_matrix(const std::vector<T>& data, int rows, int cols, const std::string& label,
                         const std::string& precision_info = "") {
  std::cout << "  " << label;
  if (!precision_info.empty()) {
    std::cout << " [" << precision_info << "]";
  }
  std::cout << " (" << rows << "x" << cols << "):" << std::endl;
  for (int r = 0; r < rows; ++r) {
    std::cout << "    ";
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r) * cols + c;
      if (idx < data.size()) {
        std::cout << static_cast<float>(data[idx]) << " ";
      }
    }
    std::cout << std::endl;
  }
}

// Generate a deterministic matrix using the MT RNG in the requested precision.
template <typename T>
static std::vector<T> make_random_matrix(int rows, int cols, Mt19937* rng, float low, float high) {
  std::vector<T> data(static_cast<size_t>(rows) * cols);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<T>(mt_uniform(rng, low, high));
  }
  return data;
}

// Generate an MxN input matrix using the MT RNG in float16 or float32 form.
template <typename T>
static std::vector<T> make_input(int M, int N, Mt19937* rng, float low, float high) {
  return make_random_matrix<T>(M, N, rng, low, high);
}

static bool run_matmul_test(const MatmulConfig& cfg) {
  std::cout << "\n" << std::string(72, '=') << std::endl;
  std::cout << "TEST: " << cfg.model_name << "  (" << cfg.M << "x" << cfg.N << ") * (" << cfg.N << "x" << cfg.K << ")" << std::endl;
  std::cout << std::string(72, '=') << std::endl;

  std::vector<uint8_t> model_data;
  std::string model_path = "models/" + cfg.model_name + ".rknn";
  if (!load_model(model_path, model_data)) {
    return false;
  }

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_init failed: " << ret << std::endl;
    return false;
  }

  rknn_tensor_attr input_attr;
  std::memset(&input_attr, 0, sizeof(input_attr));
  input_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
  if (ret < 0) {
    std::cerr << "rknn_query input attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  rknn_tensor_attr native_input_attr;
  std::memset(&native_input_attr, 0, sizeof(native_input_attr));
  native_input_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &native_input_attr, sizeof(native_input_attr));
  if (ret < 0) {
    std::cerr << "rknn_query native input attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  rknn_tensor_attr output_attr;
  std::memset(&output_attr, 0, sizeof(output_attr));
  output_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
  if (ret < 0) {
    std::cerr << "rknn_query output attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  std::cout << "  User input type=" << get_type_string(static_cast<rknn_tensor_type>(input_attr.type))
            << ", fmt=" << get_format_string(input_attr.fmt) << ", dims:";
  for (uint32_t i = 0; i < input_attr.n_dims; ++i) {
    std::cout << " " << input_attr.dims[i];
  }
  std::cout << " (size=" << input_attr.size << ")\n";

  std::cout << "  Input type=" << get_type_string(static_cast<rknn_tensor_type>(native_input_attr.type))
            << ", fmt=" << get_format_string(native_input_attr.fmt) << ", dims:";
  for (uint32_t i = 0; i < native_input_attr.n_dims; ++i) {
    std::cout << " " << native_input_attr.dims[i];
  }
  std::cout << " (size=" << native_input_attr.size << ", size_with_stride=" << native_input_attr.size_with_stride << ")\n";

  std::cout << "  Output type=" << get_type_string(static_cast<rknn_tensor_type>(output_attr.type))
            << ", fmt=" << get_format_string(output_attr.fmt) << ", dims:";
  for (uint32_t i = 0; i < output_attr.n_dims; ++i) {
    std::cout << " " << output_attr.dims[i];
  }
  std::cout << std::endl;

  const float low = -2.0f, high = 2.0f;
  Mt19937 rng{};
  mt_seed(&rng, 0);

  std::vector<uint8_t> input_bytes;
  size_t input_count = static_cast<size_t>(cfg.M) * cfg.N;
  std::vector<float> input_for_print;

  if (input_attr.type == RKNN_TENSOR_FLOAT16) {
    std::vector<__fp16> input_fp16 = make_input<__fp16>(cfg.M, cfg.N, &rng, low, high);
    input_for_print.resize(input_fp16.size());
    for (size_t i = 0; i < input_fp16.size(); ++i) input_for_print[i] = static_cast<float>(input_fp16[i]);
    size_t buf_size = input_attr.size > 0 ? input_attr.size : input_fp16.size() * sizeof(__fp16);
    input_bytes.resize(buf_size, 0);
    std::memcpy(input_bytes.data(), input_fp16.data(), std::min(buf_size, input_fp16.size() * sizeof(__fp16)));
  } else if (input_attr.type == RKNN_TENSOR_FLOAT32) {
    std::vector<float> input_fp32 = make_input<float>(cfg.M, cfg.N, &rng, low, high);
    input_for_print = input_fp32;
    size_t buf_size = input_attr.size > 0 ? input_attr.size : input_fp32.size() * sizeof(float);
    input_bytes.resize(buf_size, 0);
    std::memcpy(input_bytes.data(), input_fp32.data(), std::min(buf_size, input_fp32.size() * sizeof(float)));
  } else {
    std::cerr << "Unsupported input type: " << input_attr.type << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  rknn_input input_desc;
  std::memset(&input_desc, 0, sizeof(input_desc));
  input_desc.index = 0;
  input_desc.type = static_cast<rknn_tensor_type>(input_attr.type);
  input_desc.fmt = static_cast<rknn_tensor_format>(input_attr.fmt);
  input_desc.size = static_cast<uint32_t>(input_bytes.size());
  input_desc.buf = input_bytes.data();
  input_desc.pass_through = 0;  // let RKNN handle layout/stride packing

  ret = rknn_inputs_set(ctx, 1, &input_desc);
  if (ret < 0) {
    std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  auto start = std::chrono::high_resolution_clock::now();
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_run failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  std::cout << "  rknn_run time: " << elapsed_ms << " ms" << std::endl;

  rknn_output output_desc;
  std::memset(&output_desc, 0, sizeof(output_desc));
  output_desc.want_float = 1;   // let RKNN convert to float32
  output_desc.index = 0;
  output_desc.is_prealloc = 0;  // runtime allocs

  ret = rknn_outputs_get(ctx, 1, &output_desc, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  size_t out_count = 1;
  for (uint32_t i = 0; i < output_attr.n_dims; ++i) {
    out_count *= static_cast<size_t>(output_attr.dims[i]);
  }
  if (out_count == 0 && output_attr.size > 0) {
    // Fallback: derive from size and assume float32 since want_float=1
    out_count = output_attr.size / sizeof(float);
  }

  std::vector<float> output_host(out_count, 0.f);
  if (output_desc.buf && out_count > 0) {
    std::memcpy(output_host.data(), output_desc.buf, out_count * sizeof(float));
  } else {
    std::cerr << "Output buffer is empty" << std::endl;
  }

  // Compute expected result on CPU
  std::vector<__fp16> B_full = make_random_matrix<__fp16>(cfg.N, cfg.K, &rng, low, high);
  const std::vector<float>& input_for_cpu = input_for_print;
  std::vector<float> expected(static_cast<size_t>(cfg.M) * cfg.K, 0.f);
  std::vector<float> expected_fullsum(expected.size(), 0.f);
  for (int m = 0; m < cfg.M; ++m) {
    for (int k = 0; k < cfg.K; ++k) {
      float acc32 = 0.f;
      for (int n = 0; n < cfg.N; ++n) {
        size_t a_idx = static_cast<size_t>(m) * cfg.N + n;
        size_t b_idx = static_cast<size_t>(n) * cfg.K + k;
        float a_f = input_for_cpu[a_idx];
        float b_f = static_cast<float>(B_full[b_idx]);
        float prod = a_f * b_f;
        acc32 += prod;
      }
      expected[static_cast<size_t>(m) * cfg.K + k] = acc32;
      expected_fullsum[static_cast<size_t>(m) * cfg.K + k] = acc32;
    }
  }

  // Compare
  bool passed = true;
  float max_error = 0.f;
  const float abs_tol = 0.15f;
  const float rel_tol = 1e-2f;
  const float max_possible =
      2.0f * (static_cast<float>(cfg.N * cfg.K - 1) / 10.0f) * static_cast<float>(cfg.N);  // rough overflow bound
  size_t compare_count = std::min(expected.size(), output_host.size());
  for (size_t i = 0; i < compare_count; ++i) {
    if (std::isinf(output_host[i]) && (std::fabs(expected_fullsum[i]) > 6.5e4f || max_possible > 6.5e4f)) {
      // Treat expected overflow as acceptable if sign matches
      if ((output_host[i] > 0 && expected_fullsum[i] > 0) ||
          (output_host[i] < 0 && expected_fullsum[i] < 0)) {
        continue;
      }
    }
    if (std::isnan(output_host[i]) && std::isnan(expected[i])) continue;
    float diff = std::fabs(output_host[i] - expected[i]);
    max_error = std::max(max_error, diff);
    float tol = std::max(abs_tol, rel_tol * std::max(1.0f, std::fabs(expected[i])));
    if (diff > tol) {
      if (passed) {
        std::cerr << "  First mismatch at idx " << i << ": expected=" << expected[i]
                  << ", got=" << output_host[i] << ", diff=" << diff << std::endl;
      }
      passed = false;
    }
  }

  if (!passed) {
    std::cerr << "  Output count expected " << expected.size() << ", got " << output_host.size() << std::endl;
  }

  rknn_outputs_release(ctx, 1, &output_desc);
  rknn_destroy(ctx);

  // Print data for inspection
  print_matrix(input_for_print, cfg.M, cfg.N, "Input A",
               get_type_string(static_cast<rknn_tensor_type>(native_input_attr.type)));
  print_matrix(B_full, cfg.N, cfg.K, "Weight B", get_type_string(RKNN_TENSOR_FLOAT16));
  print_matrix(expected, cfg.M, cfg.K, "Expected (CPU FP32 accum)");
  print_matrix(output_host, cfg.M, cfg.K, "NPU Output",
               get_type_string(static_cast<rknn_tensor_type>(output_attr.type)));

  if (passed) {
    std::cout << "  ✓ PASSED (max error " << max_error << ")" << std::endl;
  } else {
    std::cout << "  ✗ FAILED (max error " << max_error << ")" << std::endl;
  }
  return passed;
}

int main() {
  std::vector<MatmulConfig> tests = {
    // {"matmul_a8x8_b8x8", 8, 8, 8},
    {"matmul_a9x9_b9x9", 9, 9, 9},
    // {"matmul_a64x64_b64x64", 64, 64, 64},
    // {"matmul_a256x256_b256x256", 256, 256, 256},
  };

  int passed = 0;
  int failed = 0;

  std::cout << "\n" << std::string(72, '#') << std::endl;
  std::cout << "MatMul Multi-Test Suite (" << tests.size() << " cases)" << std::endl;
  std::cout << std::string(72, '#') << std::endl;

  for (const auto& t : tests) {
    if (run_matmul_test(t)) {
      ++passed;
    } else {
      ++failed;
    }
  }

  std::cout << "\n" << std::string(72, '#') << std::endl;
  std::cout << "SUMMARY: passed=" << passed << ", failed=" << failed << std::endl;
  std::cout << std::string(72, '#') << std::endl;
  return failed == 0 ? 0 : 1;
}
