#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

// Convert FLOAT16 (uint16_t) to FLOAT32
static float fp16_to_fp32(uint16_t fp16) {
  // FLOAT16 layout: sign(1) | exponent(5) | mantissa(10)
  uint32_t sign = (fp16 >> 15) & 0x1;
  uint32_t exponent = (fp16 >> 10) & 0x1F;
  uint32_t mantissa = fp16 & 0x3FF;

  uint32_t fp32;

  if (exponent == 0) {
    // Subnormal or zero
    if (mantissa == 0) {
      // Zero
      return sign ? -0.0f : 0.0f;
    } else {
      // Subnormal
      mantissa <<= 1;
      while ((mantissa & (1 << 10)) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3FF;
      exponent = 127 - (exponent + 1 - 15);
    }
  } else if (exponent == 31) {
    // Infinity or NaN
    fp32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    std::memcpy(&fp32, &fp32, sizeof(float));  // Keep as-is
    return *reinterpret_cast<float*>(&fp32);
  } else {
    // Normal number
    exponent = exponent - 15 + 127;
  }

  fp32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
  return *reinterpret_cast<float*>(&fp32);
}

static const char* get_format_string(int fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "UNKNOWN";
  }
}

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
              << " type=" << attr.type << " n_elems=" << attr.n_elems << " size=" << attr.size << std::endl;
  }

  std::vector<float> input = {
      // Input shape: (1,3,5,7) - flattened in NCHW order
      // Batch 0, Channel 0 (5x7):
      1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,
      8.f,  9.f, 10.f, 11.f, 12.f, 13.f, 14.f,
     15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
     22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f,
     29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f,
      // Batch 0, Channel 1 (5x7):
     36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f,
     43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 49.f,
     50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f,
     57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f,
     64.f, 65.f, 66.f, 67.f, 68.f, 69.f, 70.f,
      // Batch 0, Channel 2 (5x7):
     71.f, 72.f, 73.f, 74.f, 75.f, 76.f, 77.f,
     78.f, 79.f, 80.f, 81.f, 82.f, 83.f, 84.f,
     85.f, 86.f, 87.f, 88.f, 89.f, 90.f, 91.f,
     92.f, 93.f, 94.f, 95.f, 96.f, 97.f, 98.f,
     99.f,100.f,101.f,102.f,103.f,104.f,105.f};
  
  std::vector<float> weight = {
      // Weight shape: (6,3,2,1) - flattened in OIHW order
      // Output Channel 0, Input Channel 0 (2x1):
      1.f, 0.f,
      // Output Channel 0, Input Channel 1 (2x1):
      1.f, 0.f,
      // Output Channel 0, Input Channel 2 (2x1):
      1.f, 0.f,
      // Output Channel 1, Input Channel 0 (2x1):
      0.f, 1.f,
      // Output Channel 1, Input Channel 1 (2x1):
      0.f, 1.f,
      // Output Channel 1, Input Channel 2 (2x1):
      0.f, 1.f,
      // Output Channel 2, Input Channel 0 (2x1):
      -1.f, 0.f,
      // Output Channel 2, Input Channel 1 (2x1):
      -1.f, 0.f,
      // Output Channel 2, Input Channel 2 (2x1):
      -1.f, 0.f,
      // Output Channel 3, Input Channel 0 (2x1):
      0.f, -1.f,
      // Output Channel 3, Input Channel 1 (2x1):
      0.f, -1.f,
      // Output Channel 3, Input Channel 2 (2x1):
      0.f, -1.f,
      // Output Channel 4, Input Channel 0 (2x1):
      1.f, 1.f,
      // Output Channel 4, Input Channel 1 (2x1):
      1.f, 1.f,
      // Output Channel 4, Input Channel 2 (2x1):
      1.f, 1.f,
      // Output Channel 5, Input Channel 0 (2x1):
      -1.f, -1.f,
      // Output Channel 5, Input Channel 1 (2x1):
      -1.f, -1.f,
      // Output Channel 5, Input Channel 2 (2x1):
      -1.f, -1.f};

  // Create a properly sized input buffer
  // The model expects input shape [1, 3, 5, 7] in NCHW format (ONNX)
  // But RKNN internally uses NHWC format [1, 5, 7, 3]
  const int in_batch = 1, in_channels = 3, in_height = 5, in_width = 7;
  const int input_size = in_batch * in_channels * in_height * in_width;

  if (input.size() != input_size) {
    std::cerr << "Input size mismatch: expected " << input_size
              << " but got " << input.size() << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  // Convert from NCHW to NHWC format and to FLOAT16
  std::vector<float> input_buffer_float(input.size());
  for (int n = 0; n < in_batch; ++n) {
    for (int h = 0; h < in_height; ++h) {
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
          // NCHW index: n, c, h, w -> flat index
          int nchw_idx = ((n * in_channels) + c) * in_height * in_width + h * in_width + w;
          // NHWC index: n, h, w, c -> flat index
          int nhwc_idx = ((n * in_height + h) * in_width + w) * in_channels + c;
          input_buffer_float[nhwc_idx] = input[nchw_idx];
        }
      }
    }
  }

  // Debug: print first 10 values
  std::cout << "Input buffer (first 10 values): ";
  for (int i = 0; i < 10 && i < (int)input_buffer_float.size(); ++i) {
    std::cout << input_buffer_float[i] << " ";
  }
  std::cout << std::endl;

  // Convert to FLOAT16
  std::vector<uint16_t> input_buffer(input_buffer_float.size());
  for (size_t i = 0; i < input_buffer_float.size(); ++i) {
    // Simple FLOAT32 to FLOAT16 conversion
    float val = input_buffer_float[i];
    // Convert to FLOAT16 by rounding
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (bits & 0x7FFFFF) >> 13;

    if (exponent <= 0) {
      if (exponent < -10) {
        input_buffer[i] = sign;
      } else {
        mantissa |= 0x1000;
        mantissa >>= (1 - exponent);
        input_buffer[i] = sign | mantissa;
      }
    } else if (exponent >= 31) {
      input_buffer[i] = sign | (0x1F << 10);
    } else {
      input_buffer[i] = sign | (exponent << 10) | mantissa;
    }
  }

  std::cout << "Input buffer size: " << input_buffer.size() << " float16 values" << std::endl;
  std::cout << "First 10 input float16 values (hex): ";
  for (int i = 0; i < 10 && i < (int)input_buffer.size(); ++i) {
    printf("0x%04X ", input_buffer[i]);
  }
  std::cout << std::endl;

  rknn_input input_desc;
  std::memset(&input_desc, 0, sizeof(input_desc));
  input_desc.index = 0;
  input_desc.buf = input_buffer.data();
  input_desc.size = static_cast<uint32_t>(input_buffer.size() * sizeof(uint16_t));
  input_desc.type = RKNN_TENSOR_FLOAT16;
  input_desc.fmt = RKNN_TENSOR_NHWC;  // RKNN only supports NHWC for input!

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
  output.want_float = 1;  // Ask RKNN to convert to float for us
  output.index = 0;

  ret = rknn_outputs_get(ctx, 1, &output, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }

  // Query output attributes
  rknn_tensor_attr output_attr;
  std::memset(&output_attr, 0, sizeof(output_attr));
  output_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
  if (ret < 0) {
    std::cerr << "rknn_query output attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return -1;
  }
  std::cout << "Output dims=";
  for (uint32_t d = 0; d < output_attr.n_dims; ++d) {
    std::cout << output_attr.dims[d] << (d + 1 == output_attr.n_dims ? "" : "x");
  }
  std::cout << " fmt=" << get_format_string(output_attr.fmt) << " n_elems=" << output_attr.n_elems << " size=" << output_attr.size << std::endl;
  std::cout << "Output buffer size: " << output.size << " bytes = " << (output.size / sizeof(float)) << " float values" << std::endl;

  int out_batch = 1;
  const int in_c = 3;
  const int in_h = 5;
  const int in_w = 7;
  const int k_c = 3;
  const int k_h = 2;
  const int k_w = 1;
  int out_channels = 6;
  int out_height = in_h - k_h + 1;
  int out_width = in_w - k_w + 1;

  if (output_attr.n_dims == 4) {
    if (output_attr.fmt == RKNN_TENSOR_NHWC) {
      out_batch = output_attr.dims[0];
      out_height = output_attr.dims[1];
      out_width = output_attr.dims[2];
      out_channels = output_attr.dims[3];
    } else {
      out_batch = output_attr.dims[0];
      out_channels = output_attr.dims[1];
      out_height = output_attr.dims[2];
      out_width = output_attr.dims[3];
    }
  }

  const size_t expected_elems = static_cast<size_t>(out_batch) * out_channels * out_height * out_width;
  size_t attr_elems = output_attr.n_elems;
  if (attr_elems == 0) {
    attr_elems = output.size / sizeof(float);
  }
  const size_t copy_elems = std::min(attr_elems, expected_elems);

  std::vector<float> output_raw(expected_elems, 0.f);
  float *result_buf = reinterpret_cast<float *>(output.buf);
  for (size_t i = 0; i < copy_elems; ++i) {
    output_raw[i] = result_buf[i];
  }
  if (copy_elems < expected_elems) {
    std::cerr << "Warning: expected " << expected_elems << " output values but got " << copy_elems << std::endl;
  }

  std::cout << "Output buffer (first 30 values): ";
  for (int i = 0; i < 30 && i < (int)output_raw.size(); ++i) {
    std::cout << output_raw[i] << " ";
  }
  std::cout << std::endl;

  std::vector<float> results(out_batch * out_channels * out_height * out_width, 0.f);
  if (output_attr.fmt == RKNN_TENSOR_NHWC) {
    for (int n = 0; n < out_batch; ++n) {
      for (int h = 0; h < out_height; ++h) {
        for (int w = 0; w < out_width; ++w) {
          for (int c = 0; c < out_channels; ++c) {
            size_t nhwc_idx = ((n * out_height + h) * out_width + w) * out_channels + c;
            size_t nchw_idx = ((n * out_channels + c) * out_height * out_width) + h * out_width + w;
            if (nhwc_idx < output_raw.size() && nchw_idx < results.size()) {
              results[nchw_idx] = output_raw[nhwc_idx];
            }
          }
        }
      }
    }
  } else {
    std::copy(output_raw.begin(), output_raw.begin() + std::min(output_raw.size(), results.size()), results.begin());
  }

  rknn_outputs_release(ctx, 1, &output);

  std::vector<float> expected(out_channels * out_height * out_width, 0.f);
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int oy = 0; oy < out_height; ++oy) {
      for (int ox = 0; ox < out_width; ++ox) {
        float acc = 0.f;
        for (int ic = 0; ic < in_c; ++ic) {
          for (int ky = 0; ky < k_h; ++ky) {
            for (int kx = 0; kx < k_w; ++kx) {
              // Input indexing: (batch, channel, height, width)
              int in_index = ic * in_h * in_w + (oy + ky) * in_w + (ox + kx);
              // Weight indexing: (out_channel, in_channel, height, width)
              int wt_index = oc * k_c * k_h * k_w + ic * k_h * k_w + ky * k_w + kx;
              acc += input[in_index] * weight[wt_index];
            }
          }
        }
        expected[oc * out_height * out_width + oy * out_width + ox] = acc;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Input (1x3x5x7):" << std::endl;
  for (int c = 0; c < in_c; ++c) {
    std::cout << "  Channel " << c << ":" << std::endl;
    for (int h = 0; h < in_h; ++h) {
      std::cout << "    ";
      for (int w = 0; w < in_w; ++w) {
        int idx = c * in_h * in_w + h * in_w + w;
        std::cout << input[idx] << " ";
      }
      std::cout << std::endl;
    }
  }
  
  std::cout << "\nWeight (6x3x2x1):" << std::endl;
  for (int oc = 0; oc < out_channels; ++oc) {
    std::cout << "  Output Channel " << oc << ":" << std::endl;
    for (int ic = 0; ic < k_c; ++ic) {
      std::cout << "    Input Channel " << ic << ":" << std::endl;
      for (int h = 0; h < k_h; ++h) {
        std::cout << "      ";
        for (int w = 0; w < k_w; ++w) {
          int idx = oc * k_c * k_h * k_w + ic * k_h * k_w + h * k_w + w;
          std::cout << weight[idx] << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  std::cout << "\nOutput (" << out_batch << "x" << out_channels << "x" << out_height << "x" << out_width << "):" << std::endl;
  for (int oc = 0; oc < out_channels; ++oc) {
    std::cout << "  Output Channel " << oc << ":" << std::endl;
    for (int h = 0; h < out_height; ++h) {
      std::cout << "    ";
      for (int w = 0; w < out_width; ++w) {
        int idx = oc * out_height * out_width + h * out_width + w;
        std::cout << results[idx] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\nExpected:" << std::endl;
  for (int oc = 0; oc < out_channels; ++oc) {
    std::cout << "  Output Channel " << oc << ":" << std::endl;
    for (int h = 0; h < out_height; ++h) {
      std::cout << "    ";
      for (int w = 0; w < out_width; ++w) {
        int idx = oc * out_height * out_width + h * out_width + w;
        std::cout << expected[idx] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\nElapsed us: " << duration_us << std::endl;

  rknn_destroy(ctx);
  return 0;
}
