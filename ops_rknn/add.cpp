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
  std::string exact = "models/add_float16_1x" + std::to_string(width) + ".rknn";
  if (file_exists(exact)) return exact;
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

static bool run_scalar(rknn_context ctx, const std::vector<rknn_tensor_attr> &in_attrs, rknn_tensor_type out_type,
                       __fp16 x, __fp16 y, float &out) {
  rknn_input in[2] {};
  in[0].index = 0;
  in[0].type = RKNN_TENSOR_FLOAT16;
  in[0].fmt = in_attrs.size() > 0 ? in_attrs[0].fmt : RKNN_TENSOR_NCHW;
  in[0].buf = &x;
  in[0].size = sizeof(__fp16);

  in[1].index = 1;
  in[1].type = RKNN_TENSOR_FLOAT16;
  in[1].fmt = in_attrs.size() > 1 ? in_attrs[1].fmt : RKNN_TENSOR_NCHW;
  in[1].buf = &y;
  in[1].size = sizeof(__fp16);

  int ret = rknn_inputs_set(ctx, 2, in);
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
  rk_out.want_float = (out_type == RKNN_TENSOR_FLOAT16 || out_type == RKNN_TENSOR_FLOAT32);
  rk_out.index = 0;
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
    rknn_outputs_release(ctx, 1, &rk_out);
    return false;
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
  if (!query_io(ctx, in_attrs, out_attrs) || out_attrs.size() < 1) {
    rknn_destroy(ctx);
    return -1;
  }
  rknn_tensor_type out_type = out_attrs[0].type;

  std::vector<__fp16> input_x(total);
  std::vector<__fp16> input_y(total);
  std::vector<float> input_x_f(total);
  std::vector<float> input_y_f(total);
  for (int i = 0; i < total; i++) {
    float vx = std::sin(i * 0.17f) * 2.0f;
    float vy = std::cos(i * 0.31f) * 2.0f;
    input_x[i] = static_cast<__fp16>(vx);
    input_y[i] = static_cast<__fp16>(vy);
    input_x_f[i] = static_cast<float>(input_x[i]);
    input_y_f[i] = static_cast<float>(input_y[i]);
  }

  std::vector<float> output(total);
  if (model_path.find("_1x1.rknn") != std::string::npos && size != 1) {
    for (int i = 0; i < total; i++) {
      float v {};
      if (!run_scalar(ctx, in_attrs, out_type, input_x[i], input_y[i], v)) {
        rknn_destroy(ctx);
        return -1;
      }
      output[i] = v;
    }
  } else {
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
    out.want_float = (out_type == RKNN_TENSOR_FLOAT16 || out_type == RKNN_TENSOR_FLOAT32);
    out.index = 0;
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
      std::cerr << "unsupported output type: " << out_type << std::endl;
      rknn_outputs_release(ctx, 1, &out);
      rknn_destroy(ctx);
      return -1;
    }
    rknn_outputs_release(ctx, 1, &out);
  }

  std::vector<float> expected(total);
  for (int i = 0; i < total; i++) expected[i] = input_x_f[i] + input_y_f[i];

  std::cout << "=== add " << size << "x" << size << " (" << model_path << ") ===" << std::endl;
  print_tensor("input_x:", input_x_f);
  print_tensor("input_y:", input_y_f);
  print_tensor("output :", output);
  print_tensor("expect :", expected);

  bool ok = true;
  float max_abs = 0.f;
  for (int i = 0; i < total; i++) {
    float diff = std::abs(output[i] - expected[i]);
    if (diff > max_abs) max_abs = diff;
    if (diff > 3.2e-2f) ok = false;
  }
  std::cout << "NPU result match CPU: " << (ok ? "YES" : "NO") << std::endl;
  std::cout << "Max abs diff: " << max_abs << std::endl;

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

  const int sizes[] = {1, 2, 4, 8, 16, 64};
  int status = 0;
  for (int size : sizes) {
    int ret = run_case(size);
    if (ret == 2) {
      std::cout << "=== add " << size << "x" << size << " ===" << std::endl;
      std::cout << "SKIP: missing model add_float16_1x" << size << ".rknn" << std::endl;
      continue;
    }
    if (ret != 0) status = ret;
  }
  return status;
}
