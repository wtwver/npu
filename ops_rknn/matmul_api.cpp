#include "rknn_matmul_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

static int align_up(int value, int align) {
  if (align <= 0) {
    return value;
  }
  return ((value + align - 1) / align) * align;
}

template <typename Scalar>
static std::vector<Scalar> pad_matrix(const std::vector<Scalar>& src, int src_rows,
                                      int src_cols, int dst_rows, int dst_cols) {
  std::vector<Scalar> padded(static_cast<size_t>(dst_rows) * dst_cols, Scalar(0));
  int copy_rows = std::min(src_rows, dst_rows);
  int copy_cols = std::min(src_cols, dst_cols);
  for (int r = 0; r < copy_rows; ++r) {
    for (int c = 0; c < copy_cols; ++c) {
      padded[static_cast<size_t>(r) * dst_cols + c] =
          src[static_cast<size_t>(r) * src_cols + c];
    }
  }
  return padded;
}

static void log_tensor_attr(const char* name,
                            const rknn_matmul_tensor_attr& attr) {
  std::cout << "  " << name << ": dims=[";
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << attr.dims[i];
  }
  std::cout << "], size=" << attr.size << std::endl;
}

static bool attr_matches_dims(const rknn_matmul_tensor_attr& attr,
                              const int32_t* expected, size_t expected_len) {
  if (attr.n_dims != expected_len) {
    return false;
  }
  for (size_t i = 0; i < expected_len; ++i) {
    if (attr.dims[i] != static_cast<uint32_t>(expected[i])) {
      return false;
    }
  }
  return true;
}

template <typename Getter>
static void print_tensor_as_list(const char* label, int rows, int cols,
                                 Getter getter) {
  std::cout << "  " << label << " (shape=" << rows << "x" << cols << "): [";
  bool first = true;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << getter(r, c);
    }
  }
  std::cout << "]\n";
}

static void print_fp16_matrix(const char* label, const __fp16* data, int rows,
                              int cols, int stride) {
  print_tensor_as_list(label, rows, cols,
                       [data, stride](int r, int c) -> float {
                         size_t idx = static_cast<size_t>(r) * stride + c;
                         return static_cast<float>(data[idx]);
                       });
}

static void print_float_matrix(const char* label, const float* data, int rows,
                               int cols, int stride) {
  print_tensor_as_list(label, rows, cols,
                       [data, rows, stride](int r, int c) -> float {
                         size_t idx = static_cast<size_t>(r) * stride + c;
                         if (idx >= static_cast<size_t>(rows) * stride) {
                           return 0.0f;
                         }
                         return data[idx];
                       });
}

static void print_int8_matrix(const char* label, const int8_t* data, int rows,
                              int cols, int stride) {
  print_tensor_as_list(label, rows, cols,
                       [data, rows, stride](int r, int c) -> int32_t {
                         size_t idx = static_cast<size_t>(r) * stride + c;
                         if (idx >= static_cast<size_t>(rows) * stride) {
                           return 0;
                         }
                         return static_cast<int32_t>(data[idx]);
                       });
}

static void print_int32_matrix(const char* label, const int32_t* data, int rows,
                               int cols, int stride) {
  print_tensor_as_list(label, rows, cols,
                       [data, rows, stride](int r, int c) -> int32_t {
                         size_t idx = static_cast<size_t>(r) * stride + c;
                         if (idx >= static_cast<size_t>(rows) * stride) {
                           return 0;
                         }
                         return data[idx];
                       });
}

struct Mt19937 {
  uint32_t mt[624];
  int index;
};

static void mt_seed(Mt19937* rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; ++i) {
    uint32_t prev = rng->mt[i - 1];
    rng->mt[i] =
        1812433253U * (prev ^ (prev >> 30)) + static_cast<uint32_t>(i);
  }
  rng->index = 624;
}

static uint32_t mt_extract(Mt19937* rng) {
  static const uint32_t mag01[2] = {0U, 0x9908b0dfU};
  if (rng->index >= 624) {
    for (int i = 0; i < 624 - 397; ++i) {
      uint32_t y = (rng->mt[i] & 0x80000000U) |
                   (rng->mt[i + 1] & 0x7fffffffU);
      rng->mt[i] = rng->mt[i + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (int i = 624 - 397; i < 623; ++i) {
      uint32_t y = (rng->mt[i] & 0x80000000U) |
                   (rng->mt[i + 1] & 0x7fffffffU);
      rng->mt[i] = rng->mt[i - (624 - 397)] ^ (y >> 1) ^ mag01[y & 1U];
    }
    uint32_t y =
        (rng->mt[623] & 0x80000000U) | (rng->mt[0] & 0x7fffffffU);
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
  const double random = (a * 67108864.0 + b) / 9007199254740992.0;
  return static_cast<float>(low + (high - low) * random);
}

static int32_t mt_uniform_int(Mt19937* rng, int32_t low, int32_t high) {
  if (low > high) {
    std::swap(low, high);
  }
  double sample = mt_uniform(rng, 0.0f, 1.0f);
  int64_t range = static_cast<int64_t>(high) - low + 1;
  if (range <= 0) {
    return low;
  }
  int64_t scaled = static_cast<int64_t>(sample * range);
  if (scaled >= range) {
    scaled = range - 1;
  }
  return static_cast<int32_t>(low + scaled);
}

template <typename Scalar>
static std::vector<Scalar> make_random_matrix(int rows, int cols, Mt19937* rng,
                                             float low, float high) {
  std::vector<Scalar> matrix(static_cast<size_t>(rows) * cols);
  for (size_t i = 0; i < matrix.size(); ++i) {
    matrix[i] = static_cast<Scalar>(mt_uniform(rng, low, high));
  }
  return matrix;
}

static std::vector<int8_t> make_random_int8_matrix(int rows, int cols,
                                                   Mt19937* rng, int8_t low,
                                                   int8_t high) {
  std::vector<int8_t> matrix(static_cast<size_t>(rows) * cols);
  for (size_t i = 0; i < matrix.size(); ++i) {
    matrix[i] = static_cast<int8_t>(mt_uniform_int(rng, low, high));
  }
  return matrix;
}

static void compute_expected_fp32(const std::vector<float>& A,
                                  const std::vector<float>& B, int M, int N,
                                  int K, std::vector<float>& output) {
  output.assign(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(m) * K + k] *
               B[static_cast<size_t>(k) * N + n];
      }
      output[static_cast<size_t>(m) * N + n] = acc;
    }
  }
}

static void compute_expected_int32(const std::vector<int8_t>& A,
                                   const std::vector<int8_t>& B, int M, int N,
                                   int K, std::vector<int32_t>& output) {
  output.assign(static_cast<size_t>(M) * N, 0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
               static_cast<int32_t>(B[static_cast<size_t>(k) * N + n]);
      }
      output[static_cast<size_t>(m) * N + n] = acc;
    }
  }
}

enum class MatmulPrecision {
  kFloat16,
  kInt8,
};

struct MatmulCase {
  std::string name;
  int M;
  int K;
  int N;
  rknn_matmul_type type;
  MatmulPrecision precision;
};

static const std::vector<MatmulCase> kMatmulCases = {
    // {"matmul_fp16_8x8x8", 8, 8, 8, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_9x9", 9, 9, 9, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_32x32", 32, 32, 32, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_64x64", 64, 64, 64, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_256x256", 256, 256, 256, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_1x8192x8192", 1, 8192, 8192, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_int8_64x64", 64, 64, 64, RKNN_INT8_MM_INT8_TO_INT32, MatmulPrecision::kInt8},
    
    // {"matmul_fp16_1x768x768", 1, 768, 768, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_1x768x2048", 1, 768, 2048, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_1x2048x2048", 1, 2048, 2048, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_1x8192x7744", 1, 8192, 7744, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_33x33x33", 33, 33, 33, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    // {"matmul_fp16_34x34x34", 34, 34, 34, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
    {"matmul_fp16_65x65x65", 65, 65, 65, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, MatmulPrecision::kFloat16},
};

struct MatmulCtxGuard {
  rknn_matmul_ctx* ctx;
  explicit MatmulCtxGuard(rknn_matmul_ctx* ctx_ptr) : ctx(ctx_ptr) {}
  ~MatmulCtxGuard() {
    if (ctx && *ctx) {
      rknn_matmul_destroy(*ctx);
      *ctx = 0;
    }
  }
};

struct TensorMemDeleter {
  rknn_matmul_ctx ctx;
  explicit TensorMemDeleter(rknn_matmul_ctx context) : ctx(context) {}
  void operator()(rknn_tensor_mem* mem) const {
    if (mem && ctx) {
      rknn_destroy_mem(ctx, mem);
    }
  }
};

static void initialize_tensor_mem(rknn_tensor_mem* mem, const void* src,
                                  size_t src_bytes) {
  if (mem == nullptr || mem->virt_addr == nullptr) {
    return;
  }
  std::memset(mem->virt_addr, 0, mem->size);
  if (src == nullptr || src_bytes == 0) {
    return;
  }
  size_t copy_bytes = std::min<size_t>(mem->size, src_bytes);
  std::memcpy(mem->virt_addr, src, copy_bytes);
}

static void dump_case_header(const MatmulCase& cfg) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << cfg.name << " (M=" << cfg.M << ", N=" << cfg.N << ", K=" << cfg.K
            << ", type=" << get_matmul_type_string(cfg.type) << ")\n";
  std::cout << std::string(60, '=') << "\n";
}

static bool run_float16_case(const MatmulCase& cfg) {
  Mt19937 rng{};
  mt_seed(&rng, 0);
  const float low = -2.0f;
  const float high = 2.0f;
  const int align_k = 32;  // value alignment for RK3588 fp16 K
  const int align_n = 32;  // value alignment for RK3588 fp16 N
  const int aligned_M = cfg.M;
  const int aligned_K = align_up(cfg.K, align_k);
  const int aligned_N = align_up(cfg.N, align_n);
  auto input_fp16 = make_random_matrix<__fp16>(cfg.M, cfg.K, &rng, low, high);
  auto weight_fp16 = make_random_matrix<__fp16>(cfg.K, cfg.N, &rng, low, high);
  bool special_9x9 = (cfg.M == 32 && cfg.N == 32 && cfg.K == 32);
  const int special_size = 9;
  // if (special_9x9) {
  //   for (int m = 0; m < cfg.M; ++m) {
  //     for (int k = 0; k < cfg.K; ++k) {
  //       if (m >= special_size || k >= special_size) {
  //         input_fp16[static_cast<size_t>(m) * cfg.K + k] = __fp16(0);
  //       }
  //     }
  //   }
  //   for (int k = 0; k < cfg.K; ++k) {
  //     for (int n = 0; n < cfg.N; ++n) {
  //       if (k >= special_size || n >= special_size) {
  //         weight_fp16[static_cast<size_t>(k) * cfg.N + n] = __fp16(0);
  //       }
  //     }
  //   }
  // }

  std::vector<float> input_host(input_fp16.size());
  std::vector<float> weight_host(weight_fp16.size());
  for (size_t i = 0; i < input_fp16.size(); ++i) {
    input_host[i] = static_cast<float>(input_fp16[i]);
  }
  for (size_t i = 0; i < weight_fp16.size(); ++i) {
    weight_host[i] = static_cast<float>(weight_fp16[i]);
  }
  auto padded_input =
      pad_matrix(input_fp16, cfg.M, cfg.K, aligned_M, aligned_K);
  auto padded_weight =
      pad_matrix(weight_fp16, cfg.K, cfg.N, aligned_K, aligned_N);
  // print_fp16_matrix("Input A padded", padded_input.data(), aligned_M, aligned_K, aligned_K);
  // print_fp16_matrix("Input B padded", padded_weight.data(), aligned_K, aligned_N, aligned_N);

  rknn_matmul_ctx ctx = 0;
  MatmulCtxGuard ctx_guard(&ctx);

  const int B_layout = 0;
  const int AC_layout = 0;
  const bool convert_B_to_native = (B_layout == 1);

  rknn_matmul_info info{};
  info.M = aligned_M;
  info.K = aligned_K;
  info.N = aligned_N;
  info.type = cfg.type;
  info.B_layout = static_cast<int16_t>(B_layout);
  info.AC_layout = static_cast<int16_t>(AC_layout);
  info.B_quant_type = 0;
  info.AC_quant_type = 0;
  info.iommu_domain_id = 0;
  info.group_size = 0;
  std::memset(info.reserved, 0, sizeof(info.reserved));

  rknn_matmul_io_attr attr{};
  int ret = rknn_matmul_create(&ctx, &info, &attr);
  if (ret != 0) {
    std::cerr << "rknn_matmul_create failed: " << ret << std::endl;
    return false;
  }

  std::cout << "  attr after create:" << std::endl;
  log_tensor_attr("A", attr.A);
  log_tensor_attr("B", attr.B);
  log_tensor_attr("C", attr.C);

  const int32_t expected_A_dims[2] = {aligned_M, aligned_K};
  const int32_t expected_B_dims[2] = {aligned_K, aligned_N};
  const int32_t expected_C_dims[2] = {aligned_M, aligned_N};
  bool attr_matches = true;
  if (AC_layout == 0) {
    attr_matches = attr_matches &&
                   attr_matches_dims(attr.A, expected_A_dims, 2) &&
                   attr_matches_dims(attr.C, expected_C_dims, 2);
  }
  if (B_layout == 0) {
    attr_matches = attr_matches &&
                   attr_matches_dims(attr.B, expected_B_dims, 2);
  }
  if (!attr_matches) {
    std::cout
        << "  attr dims do not match padded shapes, recreating with dynamic "
           "shape"
        << std::endl;
    rknn_matmul_destroy(ctx);
    ctx = 0;
    rknn_matmul_shape shapes[1] = {{aligned_M, aligned_K, aligned_N}};
    rknn_matmul_io_attr attr_array[1]{};
    ret = rknn_matmul_create_dynamic_shape(&ctx, &info, 1, shapes, attr_array);
    if (ret != 0) {
      std::cerr << "rknn_matmul_create_dynamic_shape failed: " << ret
                << std::endl;
      return false;
    }
    attr = attr_array[0];
    std::cout << "  attr after dynamic shape:" << std::endl;
    log_tensor_attr("A", attr.A);
    log_tensor_attr("B", attr.B);
    log_tensor_attr("C", attr.C);
  }

  const __fp16* weight_data = padded_weight.data();
  std::vector<__fp16> native_weight;
  if (convert_B_to_native) {
    native_weight.resize(padded_weight.size());
    int conv_ret = rknn_B_normal_layout_to_native_layout(
        padded_weight.data(), native_weight.data(), aligned_K, aligned_N,
        &info);
    if (conv_ret != 0) {
      std::cerr << "rknn_B_normal_layout_to_native_layout failed: "
                << conv_ret << std::endl;
      return false;
    }
    weight_data = native_weight.data();
  }

  using TensorMemPtr = std::unique_ptr<rknn_tensor_mem, TensorMemDeleter>;
  TensorMemPtr memA(rknn_create_mem(ctx, attr.A.size), TensorMemDeleter(ctx));
  if (!memA) {
    std::cerr << "rknn_create_mem(A) failed" << std::endl;
    return false;
  }
  TensorMemPtr memB(rknn_create_mem(ctx, attr.B.size), TensorMemDeleter(ctx));
  if (!memB) {
    std::cerr << "rknn_create_mem(B) failed" << std::endl;
    return false;
  }
  TensorMemPtr memC(rknn_create_mem(ctx, attr.C.size), TensorMemDeleter(ctx));
  if (!memC) {
    std::cerr << "rknn_create_mem(C) failed" << std::endl;
    return false;
  }

  const size_t input_bytes = padded_input.size() * sizeof(__fp16);
  if (input_bytes > memA->size) {
    std::cerr << "padded A bytes exceed allocated size: " << input_bytes
              << " > " << memA->size << std::endl;
    return false;
  }
  initialize_tensor_mem(memA.get(), padded_input.data(), input_bytes);

  const size_t weight_bytes = padded_weight.size() * sizeof(__fp16);
  if (weight_bytes > memB->size) {
    std::cerr << "padded B bytes exceed allocated size: " << weight_bytes
              << " > " << memB->size << std::endl;
    return false;
  }
  initialize_tensor_mem(memB.get(), weight_data, weight_bytes);
  initialize_tensor_mem(memC.get(), nullptr, 0);

  ret = rknn_mem_sync(ctx, memA.get(), RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync A failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_mem_sync(ctx, memB.get(), RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync B failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_mem_sync(ctx, memC.get(), RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync C failed: " << ret << std::endl;
    return false;
  }

  ret = rknn_matmul_set_io_mem(ctx, memA.get(), &attr.A);
  if (ret != 0) {
    std::cerr << "rknn_matmul_set_io_mem A failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_matmul_set_io_mem(ctx, memB.get(), &attr.B);
  if (ret != 0) {
    std::cerr << "rknn_matmul_set_io_mem B failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_matmul_set_io_mem(ctx, memC.get(), &attr.C);
  if (ret != 0) {
    std::cerr << "rknn_matmul_set_io_mem C failed: " << ret << std::endl;
    return false;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  auto start = std::chrono::high_resolution_clock::now();
  ret = rknn_matmul_run(ctx);
  auto end = std::chrono::high_resolution_clock::now();
  if (ret != 0) {
    std::cerr << "rknn_matmul_run failed: " << ret << std::endl;
    return false;
  }

  ret = rknn_mem_sync(ctx, memC.get(), RKNN_MEMORY_SYNC_FROM_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync failed: " << ret << std::endl;
    return false;
  }

  double elapsed_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;

  const size_t output_count = static_cast<size_t>(cfg.M) * cfg.N;
  size_t output_elems = attr.C.size / sizeof(float);
  if (output_elems == 0) {
    std::cerr << "  Output buffer is empty" << std::endl;
    return false;
  }
  std::vector<float> output(output_elems, 0.0f);
  size_t copy_bytes = std::min<size_t>(memC->size, output.size() * sizeof(float));
  std::memcpy(output.data(), memC->virt_addr, copy_bytes);
  int output_stride = aligned_N;
  if (AC_layout == 0 && attr.C.n_dims >= 2 && attr.C.dims[1] > 0) {
    output_stride = static_cast<int>(attr.C.dims[1]);
  }
  if (output_stride <= 0) {
    output_stride = aligned_N;
  }
  print_float_matrix("Actual output", output.data(), cfg.M, cfg.N, output_stride);

  std::vector<float> expected(output_count);
  compute_expected_fp32(input_host, weight_host, cfg.M, cfg.N, cfg.K,
                        expected);
  std::vector<float> expected_fullsum = expected;
  print_float_matrix("Expected output", expected.data(), cfg.M, cfg.N, cfg.N);

  bool passed = true;
  float max_error = 0.0f;
  const float abs_tol = 0.15f;
  const float rel_tol = 1e-2f;
  const float max_possible =
      2.0f * ((static_cast<float>(cfg.N) * cfg.K - 1.0f) / 10.0f) *
      static_cast<float>(cfg.N);
  size_t compare_count = expected.size();
  if (compare_count == 0) {
    std::cerr << "  Output buffer is empty" << std::endl;
    return false;
  }
  for (size_t i = 0; i < compare_count; ++i) {
    int row = static_cast<int>(i / cfg.N);
    int col = static_cast<int>(i % cfg.N);
    size_t data_idx = static_cast<size_t>(row) * output_stride + col;
    float actual = data_idx < output.size() ? output[data_idx] : 0.0f;
    if (std::isinf(actual) &&
        (std::fabs(expected_fullsum[i]) > 6.5e4f || max_possible > 6.5e4f)) {
      if ((actual > 0 && expected_fullsum[i] > 0) ||
          (actual < 0 && expected_fullsum[i] < 0)) {
        continue;
      }
    }
    if (std::isnan(actual) && std::isnan(expected[i])) {
      continue;
    }
    float diff = std::fabs(actual - expected[i]);
    max_error = std::max(max_error, diff);
    float tol = std::max(abs_tol, rel_tol * std::max(1.0f, std::fabs(expected[i])));
    if (diff > tol) {
      // if (passed) {
      //   std::cerr << "  First mismatch at idx " << i << ": expected=" << expected[i]
      //             << ", got=" << actual << ", diff=" << diff << std::endl;
      // }
      passed = false;
    }
  }

  std::cout << "  run time: " << elapsed_ms << " ms, max error: " << max_error
            << ", " << (passed ? "PASSED" : "FAILED") << std::endl;
  return passed;
}

static bool run_int8_case(const MatmulCase& cfg) {
  Mt19937 rng{};
  mt_seed(&rng, 0);
  const int8_t low = -8;
  const int8_t high = 7;
  const int align_k = 32;
  const int align_n = 32;
  const int aligned_K = align_up(cfg.K, align_k);
  const int aligned_N = align_up(cfg.N, align_n);
  auto input_int8 = make_random_int8_matrix(cfg.M, cfg.K, &rng, low, high);
  auto weight_int8 = make_random_int8_matrix(cfg.K, cfg.N, &rng, low, high);
  auto padded_input =
      pad_matrix(input_int8, cfg.M, cfg.K, cfg.M, aligned_K);
  auto padded_weight =
      pad_matrix(weight_int8, cfg.K, cfg.N, aligned_K, aligned_N);

  print_int8_matrix("Input A padded", padded_input.data(), cfg.M, aligned_K,
                    aligned_K);
  print_int8_matrix("Input B padded", padded_weight.data(), aligned_K,
                    aligned_N, aligned_N);

  rknn_matmul_ctx ctx = 0;
  MatmulCtxGuard ctx_guard(&ctx);

  rknn_matmul_info info{};
  info.M = cfg.M;
  info.K = aligned_K;
  info.N = aligned_N;
  info.type = cfg.type;
  info.B_layout = 0;
  info.AC_layout = 0;
  info.B_quant_type = 0;
  info.AC_quant_type = 0;
  info.iommu_domain_id = 0;
  info.group_size = 0;
  std::memset(info.reserved, 0, sizeof(info.reserved));

  rknn_matmul_io_attr attr{};
  int ret = rknn_matmul_create(&ctx, &info, &attr);
  if (ret != 0) {
    std::cerr << "rknn_matmul_create failed: " << ret << std::endl;
    return false;
  }

  using TensorMemPtr = std::unique_ptr<rknn_tensor_mem, TensorMemDeleter>;
  TensorMemPtr memA(rknn_create_mem(ctx, attr.A.size), TensorMemDeleter(ctx));
  if (!memA) {
    std::cerr << "rknn_create_mem(A) failed" << std::endl;
    return false;
  }
  TensorMemPtr memB(rknn_create_mem(ctx, attr.B.size), TensorMemDeleter(ctx));
  if (!memB) {
    std::cerr << "rknn_create_mem(B) failed" << std::endl;
    return false;
  }
  TensorMemPtr memC(rknn_create_mem(ctx, attr.C.size), TensorMemDeleter(ctx));
  if (!memC) {
    std::cerr << "rknn_create_mem(C) failed" << std::endl;
    return false;
  }

  initialize_tensor_mem(memA.get(), padded_input.data(),
                        padded_input.size() * sizeof(int8_t));
  initialize_tensor_mem(memB.get(), padded_weight.data(),
                        padded_weight.size() * sizeof(int8_t));
  initialize_tensor_mem(memC.get(), nullptr, 0);

  ret = rknn_mem_sync(ctx, memA.get(), RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync A failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_mem_sync(ctx, memB.get(), RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync B failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_mem_sync(ctx, memC.get(), RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync C failed: " << ret << std::endl;
    return false;
  }

  ret = rknn_matmul_set_io_mem(ctx, memA.get(), &attr.A);
  if (ret != 0) {
    std::cerr << "rknn_matmul_set_io_mem A failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_matmul_set_io_mem(ctx, memB.get(), &attr.B);
  if (ret != 0) {
    std::cerr << "rknn_matmul_set_io_mem B failed: " << ret << std::endl;
    return false;
  }
  ret = rknn_matmul_set_io_mem(ctx, memC.get(), &attr.C);
  if (ret != 0) {
    std::cerr << "rknn_matmul_set_io_mem C failed: " << ret << std::endl;
    return false;
  }

  auto start = std::chrono::high_resolution_clock::now();
  ret = rknn_matmul_run(ctx);
  auto end = std::chrono::high_resolution_clock::now();
  if (ret != 0) {
    std::cerr << "rknn_matmul_run failed: " << ret << std::endl;
    return false;
  }

  ret = rknn_mem_sync(ctx, memC.get(), RKNN_MEMORY_SYNC_FROM_DEVICE);
  if (ret != 0) {
    std::cerr << "rknn_mem_sync failed: " << ret << std::endl;
    return false;
  }

  double elapsed_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;

  const size_t output_count = static_cast<size_t>(cfg.M) * cfg.N;
  std::vector<int32_t> output(static_cast<size_t>(cfg.M) * aligned_N, 0);
  size_t copy_bytes =
      std::min<size_t>(memC->size, output.size() * sizeof(int32_t));
  std::memcpy(output.data(), memC->virt_addr, copy_bytes);

  print_int32_matrix("Actual output", output.data(), cfg.M, cfg.N, aligned_N);

  std::vector<int32_t> expected(output_count);
  compute_expected_int32(input_int8, weight_int8, cfg.M, cfg.N, cfg.K,
                         expected);

  print_int32_matrix("Expected output", expected.data(), cfg.M, cfg.N, cfg.N);

  bool passed = true;
  size_t compare_count = expected.size();
  if (compare_count == 0) {
    std::cerr << "  Output buffer is empty" << std::endl;
    return false;
  }
  for (size_t i = 0; i < compare_count; ++i) {
    size_t row = i / cfg.N;
    size_t col = i % cfg.N;
    size_t data_idx = row * aligned_N + col;
    int32_t actual = data_idx < output.size() ? output[data_idx] : 0;
    if (actual != expected[i]) {
      std::cerr << "  First mismatch at idx " << i << ": expected=" << expected[i]
                << ", got=" << actual << std::endl;
      passed = false;
      break;
    }
  }

  std::cout << "  run time: " << elapsed_ms << " ms, result="
            << (passed ? "PASSED" : "FAILED") << std::endl;
  return passed;
}

static bool run_case(const MatmulCase& cfg) {
  dump_case_header(cfg);
  switch (cfg.precision) {
    case MatmulPrecision::kFloat16:
      return run_float16_case(cfg);
    case MatmulPrecision::kInt8:
      return run_int8_case(cfg);
  }
  return false;
}

}  // namespace

int main() {
  int passed = 0;
  int failed = 0;
  for (const auto& cfg : kMatmulCases) {
    if (run_case(cfg)) {
      ++passed;
    } else {
      ++failed;
    }
  }

  std::cout << "\n" << std::string(60, '#') << "\n";
  std::cout << "SUMMARY: passed=" << passed << ", failed=" << failed << "\n";
  std::cout << std::string(60, '#') << "\n";
  return failed == 0 ? 0 : 1;
}
