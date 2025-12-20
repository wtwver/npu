#include "rknn_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
  std::string exact = "models/abs_float16_1x" + std::to_string(width) + ".rknn";
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

static bool query_io(rknn_context ctx, std::vector<rknn_tensor_attr> &inputs, std::vector<rknn_tensor_attr> &outputs) {
  rknn_input_output_num io_num {};
  int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    std::cerr << "rknn_query(io_num) failed: " << ret << std::endl;
    return false;
  }
  inputs.resize(io_num.n_input);
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    std::memset(&inputs[i], 0, sizeof(rknn_tensor_attr));
    inputs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &inputs[i], sizeof(rknn_tensor_attr));
    if (ret < 0) {
      std::cerr << "rknn_query(input_attr) failed: " << ret << std::endl;
      return false;
    }
  }
  outputs.resize(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    std::memset(&outputs[i], 0, sizeof(rknn_tensor_attr));
    outputs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &outputs[i], sizeof(rknn_tensor_attr));
    if (ret < 0) {
      std::cerr << "rknn_query(output_attr) failed: " << ret << std::endl;
      return false;
    }
  }
  return true;
}

struct Mt19937 {
  uint32_t mt[624];
  int index;
};

static void mt_seed(Mt19937 *rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; i++) rng->mt[i] = 1812433253U * (rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
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

void breakpoint(){};

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

  std::vector<rknn_tensor_attr> in_attrs, out_attrs;
  if (!query_io(ctx, in_attrs, out_attrs) || in_attrs.size() < 1 || out_attrs.size() < 1) {
    rknn_destroy(ctx);
    return -1;
  }

  std::vector<__fp16> input(total);
  std::vector<float> input_f(total);
  Mt19937 rng{};
  mt_seed(&rng, 0);
  for (int i = 0; i < total; i++) {
    __fp16 hv = static_cast<__fp16>(mt_uniform(&rng, -2.0f, 2.0f));
    input[i] = hv;
    input_f[i] = (float)hv;
  }

  rknn_input in {};
  in.index = 0;
  in.type = RKNN_TENSOR_FLOAT16;
  in.fmt = in_attrs[0].fmt;
  in.buf = input.data();
  in.size = input.size() * sizeof(__fp16);
  ret = rknn_inputs_set(ctx, 1, &in);
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

  breakpoint();

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
  for (int i = 0; i < total; i++) {
    __fp16 fp16 = static_cast<__fp16>(std::abs(input_f[i]));
    expected_fp16[i] = static_cast<float>(fp16);
  }

  std::cout << "=== abs " << size << "x" << size << " ===" << std::endl;
  print_tensor("input :", input_f);
  print_tensor("output:", output);
  print_tensor("exp16 :", expected_fp16);

  bool ok = true;
  float max_diff = 0.f;
  for (int i = 0; i < total; i++) {
    if (!std::isfinite(output[i])) {
      ok = false;
      continue;
    }
    float diff = std::abs(output[i] - expected_fp16[i]);
    if (diff > max_diff) max_diff = diff;
    if (diff > 3.2e-2f) ok = false;
  }
  std::cout << "NPU result match CPU(fp16): " << (ok ? "YES" : "NO") << std::endl;
  std::cout << "Max abs diff: " << max_diff << std::endl;

  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return ok ? 0 : 1;
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

  const int sizes[] = {1, 4, 8, 16, 32, 64};
  int status = 0;
  for (int size : sizes) {
    int ret = run_case(size);
    if (ret == 2) {
      std::cout << "=== abs " << size << "x" << size << " ===" << std::endl;
      std::cout << "SKIP: missing model abs_float16_1x" << size << ".rknn" << std::endl;
      continue;
    }
    if (ret != 0) status = ret;
  }
  return status;
}

