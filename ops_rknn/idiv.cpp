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
  std::string exact = "models/div_int32_1x" + std::to_string(width) + ".rknn";
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

static void print_tensor(const char *label, const std::vector<int32_t> &vals) {
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
                       int32_t x, int32_t y, int32_t &out) {
  if (y == 0) return false;

  rknn_input in[2] {};
  in[0].index = 0;
  in[0].type = RKNN_TENSOR_INT32;
  in[0].fmt = in_attrs.size() > 0 ? in_attrs[0].fmt : RKNN_TENSOR_NCHW;
  in[0].buf = &x;
  in[0].size = sizeof(int32_t);

  in[1].index = 1;
  in[1].type = RKNN_TENSOR_INT32;
  in[1].fmt = in_attrs.size() > 1 ? in_attrs[1].fmt : RKNN_TENSOR_NCHW;
  in[1].buf = &y;
  in[1].size = sizeof(int32_t);

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

  if (out_type == RKNN_TENSOR_INT32) {
    out = *static_cast<int32_t *>(rk_out.buf);
  } else if (rk_out.want_float) {
    float v = *static_cast<float *>(rk_out.buf);
    if (!std::isfinite(v)) {
      rknn_outputs_release(ctx, 1, &rk_out);
      return false;
    }
    out = static_cast<int32_t>(std::trunc(v));
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

  std::vector<int32_t> input_x(total);
  std::vector<int32_t> input_y(total);
  for (int i = 0; i < total; i++) {
    int32_t x = static_cast<int32_t>(i * 3 - total);
    int32_t y = static_cast<int32_t>((i % 7) - 3);
    if (y == 0) y = 1;
    input_x[i] = x;
    input_y[i] = y;
  }

  std::vector<int32_t> output(total);
  if (model_path.find("_1x1.rknn") != std::string::npos && size != 1) {
    for (int i = 0; i < total; i++) {
      int32_t v {};
      if (!run_scalar(ctx, in_attrs, out_type, input_x[i], input_y[i], v)) {
        rknn_destroy(ctx);
        return -1;
      }
      output[i] = v;
    }
  } else {
    rknn_input in[2] {};
    in[0].index = 0;
    in[0].type = RKNN_TENSOR_INT32;
    in[0].fmt = in_attrs.size() > 0 ? in_attrs[0].fmt : RKNN_TENSOR_NCHW;
    in[0].buf = input_x.data();
    in[0].size = input_x.size() * sizeof(int32_t);

    in[1].index = 1;
    in[1].type = RKNN_TENSOR_INT32;
    in[1].fmt = in_attrs.size() > 1 ? in_attrs[1].fmt : RKNN_TENSOR_NCHW;
    in[1].buf = input_y.data();
    in[1].size = input_y.size() * sizeof(int32_t);

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

    if (out_type == RKNN_TENSOR_INT32) {
      if (out.size < static_cast<uint32_t>(total * sizeof(int32_t))) {
        std::cerr << "unexpected output size: " << out.size << std::endl;
        rknn_outputs_release(ctx, 1, &out);
        rknn_destroy(ctx);
        return -1;
      }
      int32_t *result = static_cast<int32_t *>(out.buf);
      std::copy(result, result + total, output.begin());
    } else if (out.want_float) {
      if (out.size < static_cast<uint32_t>(total * sizeof(float))) {
        std::cerr << "unexpected output size: " << out.size << std::endl;
        rknn_outputs_release(ctx, 1, &out);
        rknn_destroy(ctx);
        return -1;
      }
      float *result = static_cast<float *>(out.buf);
      for (int i = 0; i < total; i++) {
        if (!std::isfinite(result[i])) {
          rknn_outputs_release(ctx, 1, &out);
          rknn_destroy(ctx);
          return -1;
        }
        output[i] = static_cast<int32_t>(std::trunc(result[i]));
      }
    } else {
      std::cerr << "unsupported output type: " << out_type << std::endl;
      rknn_outputs_release(ctx, 1, &out);
      rknn_destroy(ctx);
      return -1;
    }
    rknn_outputs_release(ctx, 1, &out);
  }

  std::vector<int32_t> expected(total);
  for (int i = 0; i < total; i++) expected[i] = input_x[i] / input_y[i];

  std::cout << "=== idiv " << size << "x" << size << " ===" << std::endl;
  print_tensor("input_x:", input_x);
  print_tensor("input_y:", input_y);
  print_tensor("output :", output);
  print_tensor("expect :", expected);

  bool ok = true;
  int32_t max_abs_diff = 0;
  for (int i = 0; i < total; i++) {
    int32_t diff = std::abs(output[i] - expected[i]);
    if (diff > max_abs_diff) max_abs_diff = diff;
    if (diff != 0) ok = false;
  }
  std::cout << "NPU result match CPU: " << (ok ? "YES" : "NO") << std::endl;
  std::cout << "Max abs diff: " << max_abs_diff << std::endl;

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
      std::cout << "=== idiv " << size << "x" << size << " ===" << std::endl;
      std::cout << "SKIP: missing model div_int32_1x" << size << ".rknn" << std::endl;
      continue;
    }
    if (ret != 0) status = ret;
  }
  return status;
}
