#include "rknn_api.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

static bool load_model(const std::string &path, std::vector<uint8_t> &data) {
  FILE *fp = fopen(path.c_str(), "rb");
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

int main() {
  const std::string model_path = "models/conv2d_simple.rknn";

  std::vector<uint8_t> model_data;
  if (!load_model(model_path, model_data)) {
    return -1;
  }

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_init failed: " << ret << std::endl;
    return -1;
  }

  rknn_input_output_num io_num;
  std::memset(&io_num, 0, sizeof(io_num));
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    std::cerr << "rknn_query IO num failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  std::cout << "Model IO count - inputs: " << io_num.n_input
            << ", outputs: " << io_num.n_output << std::endl;

  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn_tensor_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret < 0) {
      std::cerr << "rknn_query input attr failed: " << ret << " index " << i << std::endl;
      rknn_destroy(ctx);
      return -1;
    }
    std::cout << "Input[" << i << "] name=" << attr.name << " dims=";
    for (uint32_t d = 0; d < attr.n_dims; ++d) {
      std::cout << attr.dims[d] << (d + 1 == attr.n_dims ? "" : "x");
    }
    std::cout << " fmt=" << get_format_string(attr.fmt)
              << " type=" << attr.type << std::endl;
  }

  std::vector<float> input = {
      1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,
      9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f};
  std::vector<float> weight = {
      1.f, 0.f, -1.f,
      1.f, 0.f, -1.f,
      1.f, 0.f, -1.f};

  rknn_input input_desc;
  std::memset(&input_desc, 0, sizeof(input_desc));
  input_desc.index = 0;
  input_desc.buf = input.data();
  input_desc.size = static_cast<uint32_t>(input.size() * sizeof(float));
  input_desc.type = RKNN_TENSOR_FLOAT32;
  input_desc.fmt = RKNN_TENSOR_NHWC;

  ret = rknn_inputs_set(ctx, 1, &input_desc);
  if (ret < 0) {
    std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_run failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  rknn_output output;
  std::memset(&output, 0, sizeof(output));
  output.want_float = 1;
  output.index = 0;

  ret = rknn_outputs_get(ctx, 1, &output, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  size_t output_elems = output.size / sizeof(float);
  std::vector<float> results(output_elems, 0.f);
  float *result_buf = reinterpret_cast<float *>(output.buf);
  for (size_t i = 0; i < results.size(); ++i) {
    results[i] = result_buf[i];
  }

  rknn_outputs_release(ctx, 1, &output);

  const int in_h = 4;
  const int in_w = 4;
  const int k_h = 3;
  const int k_w = 3;
  const int out_h = in_h - k_h + 1;
  const int out_w = in_w - k_w + 1;

  std::vector<float> expected(out_h * out_w, 0.f);
  for (int oy = 0; oy < out_h; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      float acc = 0.f;
      for (int ky = 0; ky < k_h; ++ky) {
        for (int kx = 0; kx < k_w; ++kx) {
          int in_index = (oy + ky) * in_w + (ox + kx);
          int wt_index = ky * k_w + kx;
          acc += input[in_index] * weight[wt_index];
        }
      }
      expected[oy * out_w + ox] = acc;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Input (1x1x4x4):";
  for (size_t i = 0; i < input.size(); ++i) {
    if (i % in_w == 0) std::cout << "\n  ";
    std::cout << input[i] << " ";
  }
  std::cout << "\nWeight (1x1x3x3):";
  for (size_t i = 0; i < weight.size(); ++i) {
    if (i % k_w == 0) std::cout << "\n  ";
    std::cout << weight[i] << " ";
  }

  std::cout << "\nOutput    :";
  for (size_t i = 0; i < results.size(); ++i) {
    if (i % out_w == 0) std::cout << "\n  ";
    std::cout << results[i] << " ";
  }

  std::cout << "\nExpected  :";
  for (size_t i = 0; i < expected.size(); ++i) {
    if (i % out_w == 0) std::cout << "\n  ";
    std::cout << expected[i] << " ";
  }

  std::cout << "\nElapsed us: " << duration_us << std::endl;

  rknn_destroy(ctx);
  return 0;
}
