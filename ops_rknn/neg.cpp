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
  std::string exact = "models/neg_bool_1x" + std::to_string(width) + ".rknn";
  if (file_exists(exact)) return exact;
  std::string generic = "models/neg_bool_1x1.rknn";
  if (file_exists(generic)) return generic;
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

static bool run_scalar(rknn_context ctx, const rknn_tensor_attr &in_attr, const rknn_tensor_attr &out_attr, uint8_t x, float &out) {
  rknn_input in {};
  in.index = 0;
  in.type = in_attr.type;
  in.fmt = in_attr.fmt;
  in.buf = &x;
  in.size = sizeof(uint8_t);

  int ret = rknn_inputs_set(ctx, 1, &in);
  if (ret < 0) {
    std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
    return false;
  }

  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_run failed: " << ret << std::endl;
    return false;
  }

  rknn_output rk_out {};
  rk_out.index = 0;
  rk_out.want_float = (out_attr.type == RKNN_TENSOR_FLOAT16 || out_attr.type == RKNN_TENSOR_FLOAT32);
  ret = rknn_outputs_get(ctx, 1, &rk_out, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    return false;
  }

  if (rk_out.want_float) {
    float v = *static_cast<float *>(rk_out.buf);
    if (!std::isfinite(v)) {
      rknn_outputs_release(ctx, 1, &rk_out);
      return false;
    }
    out = v;
  } else {
    uint8_t v = 0;
    if (rk_out.size >= 1) v = *static_cast<uint8_t *>(rk_out.buf);
    out = v ? 1.0f : 0.0f;
  }
  rknn_outputs_release(ctx, 1, &rk_out);
  return true;
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

  std::vector<rknn_tensor_attr> in_attrs, out_attrs;
  if (!query_io(ctx, in_attrs, out_attrs) || in_attrs.size() < 1 || out_attrs.size() < 1) {
    rknn_destroy(ctx);
    return -1;
  }
  const rknn_tensor_attr &in_attr = in_attrs[0];
  const rknn_tensor_attr &out_attr = out_attrs[0];

  std::vector<uint8_t> input(static_cast<size_t>(total));
  std::vector<float> input_f(static_cast<size_t>(total));
  for (int i = 0; i < total; i++) {
    input[i] = static_cast<uint8_t>(i & 1);
    input_f[i] = input[i] ? 1.0f : 0.0f;
  }

  std::vector<float> output(static_cast<size_t>(total));
  if (model_path.find("_1x1.rknn") != std::string::npos && size != 1) {
    for (int i = 0; i < total; i++) {
      float v {};
      if (!run_scalar(ctx, in_attr, out_attr, input[i], v)) {
        rknn_destroy(ctx);
        return -1;
      }
      output[i] = v;
    }
  } else {
    rknn_input in {};
    in.index = 0;
    in.type = in_attr.type;
    in.fmt = in_attr.fmt;
    in.buf = input.data();
    in.size = input.size() * sizeof(uint8_t);

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

    rknn_output out {};
    out.index = 0;
    out.want_float = (out_attr.type == RKNN_TENSOR_FLOAT16 || out_attr.type == RKNN_TENSOR_FLOAT32);
    ret = rknn_outputs_get(ctx, 1, &out, nullptr);
    if (ret < 0) {
      std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
      rknn_destroy(ctx);
      return -1;
    }

    if (out.want_float) {
      float *result = static_cast<float *>(out.buf);
      std::copy(result, result + total, output.begin());
    } else {
      uint8_t *result = static_cast<uint8_t *>(out.buf);
      for (int i = 0; i < total; i++) output[i] = result[i] ? 1.0f : 0.0f;
    }
    rknn_outputs_release(ctx, 1, &out);
  }

  std::vector<float> expected(static_cast<size_t>(total));
  for (int i = 0; i < total; i++) expected[i] = input_f[i] ? 0.0f : 1.0f;

  std::cout << "=== neg_bool " << size << "x" << size << " (" << model_path << ") ===" << std::endl;
  print_tensor("input :", input_f);
  print_tensor("output:", output);
  print_tensor("expect:", expected);

  bool ok = true;
  for (int i = 0; i < total; i++) {
    if (output[i] != expected[i]) ok = false;
  }
  std::cout << "Result match CPU: " << (ok ? "YES" : "NO") << std::endl;

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

  const int sizes[] = {4};
  int status = 0;
  for (int size : sizes) {
    int ret = run_case(size);
    if (ret == 2) {
      std::cout << "=== neg_bool " << size << "x" << size << " ===" << std::endl;
      std::cout << "SKIP: missing model neg_bool_1x" << size << ".rknn" << std::endl;
      continue;
    }
    if (ret != 0) status = ret;
  }
  return status;
}

