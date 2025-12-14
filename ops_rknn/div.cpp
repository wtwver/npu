#include "rknn_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

static bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

static std::string pick_model(int width) {
  std::string exact = "models/div_float16_1x" + std::to_string(width) + ".rknn";
  if (file_exists(exact)) return exact;
  if (width == 1) return exact;
  std::cerr << "Missing model for width " << width << " (" << exact << ")" << std::endl;
  return "";
}

static bool load_model(const std::string &path, std::vector<unsigned char> &data) {
  FILE *fp = std::fopen(path.c_str(), "rb");
  if (!fp) {
    std::cerr << "failed to open model: " << path << std::endl;
    return false;
  }
  struct stat st {};
  if (stat(path.c_str(), &st) != 0 || st.st_size <= 0) {
    std::cerr << "failed to stat model: " << path << std::endl;
    std::fclose(fp);
    return false;
  }
  data.resize(static_cast<size_t>(st.st_size));
  if (std::fread(data.data(), 1, data.size(), fp) != data.size()) {
    std::cerr << "failed to read model" << std::endl;
    std::fclose(fp);
    return false;
  }
  std::fclose(fp);
  return true;
}

static void print_tensor(const char *label, const std::vector<float> &vals) {
  std::cout << label;
  size_t limit = std::min<size_t>(vals.size(), 16);
  for (size_t i = 0; i < limit; i++) std::cout << " " << vals[i];
  if (vals.size() > limit) std::cout << " ...";
  std::cout << std::endl;
}

struct Mt19937 {
  uint32_t mt[624];
  int index;
};

static void mt_seed(Mt19937 *rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; i++) {
    rng->mt[i] = 1812433253U * (rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
  }
  rng->index = 624;
}

static uint32_t mt_extract(Mt19937 *rng) {
  const uint32_t mag01[2] = {0U, 0x9908b0dfU};
  if (rng->index >= 624) {
    int kk;
    for (kk = 0; kk < 624 - 397; kk++) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk + 1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (; kk < 623; kk++) {
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

static float mt_uniform(Mt19937 *rng, float low, float high) {
  const double a = static_cast<double>(mt_extract(rng) >> 5);
  const double b = static_cast<double>(mt_extract(rng) >> 6);
  const double random = (a * 67108864.0 + b) / 9007199254740992.0;
  return static_cast<float>(low + (high - low) * random);
}

static int run_case(int size) {
  if (size <= 0) return -1;
  int total = size * size;
  std::string model_path = pick_model(size);
  if (model_path.empty()) return 2;
  std::vector<unsigned char> model;
  if (!load_model(model_path, model)) return -1;

  rknn_context ctx;
  int ret = rknn_init(&ctx, model.data(), model.size(), 0, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_init failed: " << ret << std::endl;
    return -1;
  }

  std::vector<__fp16> input_x(total);
  std::vector<__fp16> input_y(total);
  std::vector<float> input_x_f(total);
  std::vector<float> input_y_f(total);
  Mt19937 rng{};
  mt_seed(&rng, 0);
  for (int i = 0; i < total; i++) {
    float vx = mt_uniform(&rng, -2.0f, 2.0f);
    float vy = mt_uniform(&rng, -2.0f, 2.0f);
    input_x[i] = static_cast<__fp16>(vx);
    input_y[i] = static_cast<__fp16>(vy);
    if (std::abs((float)input_y[i]) < 1e-3f) input_y[i] = static_cast<__fp16>(1.0f);
    input_x_f[i] = (float)input_x[i];
    input_y_f[i] = (float)input_y[i];
  }

  rknn_input in[2] {};
  in[0].index = 0;
  in[0].type = RKNN_TENSOR_FLOAT16;
  in[0].fmt = RKNN_TENSOR_NHWC;
  in[0].buf = input_x.data();
  in[0].size = input_x.size() * sizeof(__fp16);

  in[1].index = 1;
  in[1].type = RKNN_TENSOR_FLOAT16;
  in[1].fmt = RKNN_TENSOR_NHWC;
  in[1].buf = input_y.data();
  in[1].size = input_y.size() * sizeof(__fp16);

  ret = rknn_inputs_set(ctx, 2, in);
  if (ret < 0) {
    std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_run failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  rknn_output out {};
  out.want_float = 1;
  out.index = 0;
  ret = rknn_outputs_get(ctx, 1, &out, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  float *result = static_cast<float *>(out.buf);
  std::vector<float> output(result, result + total);
  std::vector<float> expected_fp16(total);
  std::vector<float> expected_fp32(total);
  for (int i = 0; i < total; i++) {
    float fp32 = input_x_f[i] / input_y_f[i];
    __fp16 fp16 = static_cast<__fp16>(fp32);
    expected_fp16[i] = static_cast<float>(fp16);
    expected_fp32[i] = fp32;
  }

  std::cout << "=== div " << size << "x" << size << " ===" << std::endl;
  print_tensor("input_x:", input_x_f);
  print_tensor("input_y:", input_y_f);
  print_tensor("output :", output);
  print_tensor("exp16  :", expected_fp16);
  print_tensor("exp32  :", expected_fp32);

  bool ok_fp16 = true;
  bool ok_fp32 = true;
  float max_diff_fp16 = 0.f;
  float max_diff_fp32 = 0.f;
  for (int i = 0; i < total; i++) {
    if (!std::isfinite(output[i])) {
      ok_fp16 = false;
      ok_fp32 = false;
      continue;
    }
    float diff_fp16 = std::abs(output[i] - expected_fp16[i]);
    float diff_fp32 = std::abs(output[i] - expected_fp32[i]);
    if (diff_fp16 > max_diff_fp16) max_diff_fp16 = diff_fp16;
    if (diff_fp32 > max_diff_fp32) max_diff_fp32 = diff_fp32;
    if (diff_fp16 > 3.2e-2f) ok_fp16 = false;
    if (diff_fp32 > 2e-1f) ok_fp32 = false;
  }
  std::cout << "NPU match fp16-CPU: " << (ok_fp16 ? "YES" : "NO") << std::endl;
  std::cout << "NPU match fp32-CPU: " << (ok_fp32 ? "YES" : "NO") << std::endl;
  std::cout << "Max abs diff fp16: " << max_diff_fp16 << std::endl;
  std::cout << "Max abs diff fp32: " << max_diff_fp32 << std::endl;

  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return (ok_fp16 || ok_fp32) ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc > 1) {
    int size = std::atoi(argv[1]);
    if (size <= 0) {
      std::cerr << "invalid size" << std::endl;
      return -1;
    }
    return run_case(size);
  }

  const int sizes[] = {64, 72, 80, 88, 90, 91, 92, 96, 112, 120, 124, 126, 127, 128};
  int status = 0;
  for (int size : sizes) {
    int ret = run_case(size);
    if (ret == 2) {
      std::cout << "=== div " << size << "x" << size << " ===" << std::endl;
      std::cout << "SKIP: missing model div_float16_1x" << size << ".rknn" << std::endl;
      continue;
    }
    if (ret != 0) status = ret;
  }
  return status;
}
