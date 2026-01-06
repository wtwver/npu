#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <cstdint>
#include <vector>

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

// Structure to hold test case configuration
struct ConvConfig {
  std::string model_name;
  int out_channels;
  int in_channels;
  int in_height;
  int in_width;
  int kernel_h;
  int kernel_w;
  int groups;
  std::string description;
};

// Simple MT19937 implementation (matches ops_reg/main.c)
struct Mt19937 {
  uint32_t mt[624];
  int index;
};

static void mt_seed(Mt19937* rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; i++) {
    rng->mt[i] = 1812433253U * (rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
  }
  rng->index = 624;
}

static uint32_t mt_extract(Mt19937* rng) {
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

static float mt_uniform(Mt19937* rng, float low, float high) {
  const double a = static_cast<double>(mt_extract(rng) >> 5);
  const double b = static_cast<double>(mt_extract(rng) >> 6);
  const double random = (a * 67108864.0 + b) / 9007199254740992.0;
  return static_cast<float>(low + (high - low) * random);
}

// Generate input data for a given shape using MT uniform RNG
static std::vector<__fp16> generate_input_data(int batch, int channels, int height, int width,
                                               Mt19937* rng, float low, float high) {
  std::vector<__fp16> input;
  input.reserve(batch * channels * height * width);

  for (int n = 0; n < batch; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          (void)n;  // batch dimension not used by RNG sequence
          input.push_back(static_cast<__fp16>(mt_uniform(rng, low, high)));
        }
      }
    }
  }
  return input;
}

// Generate weight data for a given configuration using MT uniform RNG
static std::vector<float> generate_weight_data(const ConvConfig& config, Mt19937* rng, float low, float high) {
  std::vector<float> weight;
  int in_ch_per_group = config.in_channels / config.groups;

  weight.reserve(config.out_channels * in_ch_per_group * config.kernel_h * config.kernel_w);

  for (int oc = 0; oc < config.out_channels; ++oc) {
    for (int ic = 0; ic < in_ch_per_group; ++ic) {
      for (int kh = 0; kh < config.kernel_h; ++kh) {
        for (int kw = 0; kw < config.kernel_w; ++kw) {
          (void)oc; (void)kh; (void)kw;  // silence unused warnings if any
          weight.push_back(mt_uniform(rng, low, high));
        }
      }
    }
  }
  return weight;
}

static std::vector<__fp16> nchw_to_nhwc(const std::vector<__fp16>& src,
                                        int batch, int channels, int height, int width) {
  std::vector<__fp16> dst(src.size());
  for (int n = 0; n < batch; ++n) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          int nchw_idx = ((n * channels + c) * height + h) * width + w;
          int nhwc_idx = ((n * height + h) * width + w) * channels + c;
          dst[nhwc_idx] = src[nchw_idx];
        }
      }
    }
  }
  return dst;
}

static std::vector<float> nc1hwc2_fp16_to_nchw(const __fp16* src,
                                               int batch, int c1, int height, int width,
                                               int stride_w, int c2, int channels) {
  std::vector<float> dst(static_cast<size_t>(batch) * channels * height * width, 0.f);
  int padded_w = stride_w > 0 ? stride_w : width;
  size_t idx = 0;
  for (int n = 0; n < batch; ++n) {
    for (int g = 0; g < c1; ++g) {
      for (int c = 0; c < c2; ++c) {
        int channel = g * c2 + c;
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < padded_w; ++x) {
            float val = static_cast<float>(src[idx++]);
            if (channel < channels && x < width) {
              size_t dst_idx = ((n * channels + channel) * height + y) * width + x;
              dst[dst_idx] = val;
            }
          }
        }
      }
    }
  }
  return dst;
}

// Run a single convolution test
static bool run_conv_test(const ConvConfig& config, bool verbose) {
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "TEST: " << config.description << std::endl;
  std::cout << "  Weight shape: (" << config.out_channels << "," << config.in_channels << ","
            << config.kernel_h << "," << config.kernel_w << ")";
  if (config.groups > 1) {
    std::cout << " [groups=" << config.groups << "]";
  }
  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // Load the model
  std::string model_path = "models/" + config.model_name + ".rknn";
  std::vector<uint8_t> model_data;
  if (!load_model(model_path, model_data)) {
    std::cerr << "Failed to load model: " << model_path << std::endl;
    return false;
  }

  // Initialize RKNN context
  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_init failed: " << ret << std::endl;
    return false;
  }

  // Get IO info
  rknn_input_output_num io_num;
  std::memset(&io_num, 0, sizeof(io_num));
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    std::cerr << "rknn_query IO num failed: " << ret << std::endl;
    rknn_destroy(ctx);
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
  std::cout << "  Native input type: " << get_type_string(static_cast<rknn_tensor_type>(native_input_attr.type))
            << ", format: " << get_format_string(native_input_attr.fmt) << std::endl;
  std::cout << "  Native input dims:";
  for (uint32_t i = 0; i < native_input_attr.n_dims; ++i) {
    std::cout << " " << native_input_attr.dims[i];
  }
  std::cout << std::endl;
  std::cout << "  Native input size: " << native_input_attr.size
            << ", size_with_stride: " << native_input_attr.size_with_stride
            << ", w_stride: " << native_input_attr.w_stride << std::endl;

  rknn_tensor_attr native_output_attr;
  std::memset(&native_output_attr, 0, sizeof(native_output_attr));
  native_output_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native_output_attr, sizeof(native_output_attr));
  if (ret < 0) {
    std::cerr << "rknn_query native output attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }
  std::cout << "  Native output type: " << get_type_string(static_cast<rknn_tensor_type>(native_output_attr.type))
            << ", format: " << get_format_string(native_output_attr.fmt) << std::endl;
  std::cout << "  Native output dims:";
  for (uint32_t i = 0; i < native_output_attr.n_dims; ++i) {
    std::cout << " " << native_output_attr.dims[i];
  }
  std::cout << std::endl;
  std::cout << "  Native output size: " << native_output_attr.size
            << ", size_with_stride: " << native_output_attr.size_with_stride
            << ", w_stride: " << native_output_attr.w_stride << std::endl;

  uint32_t output_mem_size = native_output_attr.size_with_stride > 0 ? native_output_attr.size_with_stride
                                                                     : native_output_attr.size;
  rknn_tensor_mem* output_mem = rknn_create_mem(ctx, output_mem_size);
  if (!output_mem) {
    std::cerr << "rknn_create_mem failed for output" << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  rknn_tensor_attr output_mem_attr = native_output_attr;
  output_mem_attr.pass_through = 1;
  // output_mem_attr.pass_through = 0;
  ret = rknn_set_io_mem(ctx, output_mem, &output_mem_attr);
  if (ret < 0) {
    std::cerr << "rknn_set_io_mem failed: " << ret << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  // Prepare input data
  const int in_batch = 1;
  const int in_channels = config.in_channels > 0 ? config.in_channels : 3;
  const int in_height = config.in_height > 0 ? config.in_height : 5;
  const int in_width = config.in_width > 0 ? config.in_width : 7;
  const float low = -2.0f, high = 2.0f;
  Mt19937 rng;
  mt_seed(&rng, 0);
  std::vector<__fp16> input = generate_input_data(in_batch, in_channels, in_height, in_width, &rng, low, high);
  std::vector<float> input_cpu(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    input_cpu[i] = static_cast<float>(input[i]);
  }

  if (verbose) {
    std::cout << "  Input tensor (NCHW)" << std::endl;
    for (int n = 0; n < in_batch; ++n) {
      std::cout << "    n=" << n << std::endl;
      for (int c = 0; c < in_channels; ++c) {
        std::cout << "      c=" << c << std::endl;
        for (int h = 0; h < in_height; ++h) {
          std::cout << "        h=" << h << ": ";
          for (int w = 0; w < in_width; ++w) {
            size_t idx = ((n * in_channels + c) * in_height + h) * in_width + w;
            std::cout << static_cast<float>(input[idx]) << " ";
          }
          std::cout << std::endl;
        }
      }
    }
  }

  const std::vector<__fp16>* formatted_input = &input;
  std::vector<__fp16> input_nhwc;
  std::vector<__fp16> input_padded;
  if (native_input_attr.fmt == RKNN_TENSOR_NHWC) {
    input_nhwc = nchw_to_nhwc(input, in_batch, in_channels, in_height, in_width);
    formatted_input = &input_nhwc;

    int native_batch = (native_input_attr.n_dims > 0) ? native_input_attr.dims[0] : in_batch;
    int native_height = (native_input_attr.n_dims > 1) ? native_input_attr.dims[1] : in_height;
    int native_width = (native_input_attr.n_dims > 2) ? native_input_attr.dims[2] : in_width;
    int native_channels = (native_input_attr.n_dims > 3) ? native_input_attr.dims[3] : in_channels;
    int stride_w = native_input_attr.w_stride > 0 ? native_input_attr.w_stride : native_width;

    if (stride_w != native_width) {
      input_padded.assign(static_cast<size_t>(native_batch) * native_height * stride_w * native_channels,
                          static_cast<__fp16>(0));
      for (int n = 0; n < native_batch; ++n) {
        for (int h = 0; h < native_height; ++h) {
          for (int w = 0; w < native_width; ++w) {
            for (int c = 0; c < native_channels; ++c) {
              size_t src_idx = ((n * native_height + h) * native_width + w) * native_channels + c;
              size_t dst_idx = ((n * native_height + h) * stride_w + w) * native_channels + c;
              if (src_idx < input_nhwc.size() && dst_idx < input_padded.size()) {
                input_padded[dst_idx] = input_nhwc[src_idx];
              }
            }
          }
        }
      }
      formatted_input = &input_padded;
    }
  } else if (native_input_attr.fmt != RKNN_TENSOR_NCHW) {
    std::cerr << "Unsupported native input format: " << native_input_attr.fmt << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  if (native_input_attr.type != RKNN_TENSOR_FLOAT16) {
    std::cerr << "Unsupported native input type (expected FP16): " << native_input_attr.type << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  const void* input_buf_ptr = static_cast<const void*>(formatted_input->data());
  uint32_t input_size_bytes = static_cast<uint32_t>(formatted_input->size() * sizeof(__fp16));

  // Set input
  rknn_input input_desc;
  std::memset(&input_desc, 0, sizeof(input_desc));
  input_desc.index = 0;
  input_desc.buf = const_cast<void*>(input_buf_ptr);
  input_desc.size = (native_input_attr.size_with_stride > 0) ? native_input_attr.size_with_stride : input_size_bytes;
  input_desc.pass_through = 1;
  input_desc.type = static_cast<rknn_tensor_type>(native_input_attr.type);
  input_desc.fmt = static_cast<rknn_tensor_format>(native_input_attr.fmt);

  ret = rknn_inputs_set(ctx, 1, &input_desc);
  if (ret < 0) {
    std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  // Run inference
  auto start = std::chrono::high_resolution_clock::now();
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_run failed: " << ret << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  rknn_output output_desc;
  std::memset(&output_desc, 0, sizeof(output_desc));
  output_desc.want_float = 0;
  output_desc.is_prealloc = 1;
  output_desc.index = 0;
  output_desc.buf = output_mem->virt_addr;
  output_desc.size = output_mem_size;

  ret = rknn_outputs_get(ctx, 1, &output_desc, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  // Query logical output attributes
  rknn_tensor_attr output_attr;
  std::memset(&output_attr, 0, sizeof(output_attr));
  output_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
  if (ret < 0) {
    std::cerr << "rknn_query output attr failed: " << ret << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  // Calculate expected output dimensions
  int out_height = in_height - config.kernel_h + 1;
  int out_width = in_width - config.kernel_w + 1;
  int out_channels = config.out_channels;
  int out_batch = 1;

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

  if (out_height != in_height - config.kernel_h + 1 ||
      out_width != in_width - config.kernel_w + 1) {
    std::cerr << "ERROR: Output dimensions mismatch!" << std::endl;
    std::cerr << "  Expected: " << (in_height - config.kernel_h + 1) << "x"
              << (in_width - config.kernel_w + 1) << std::endl;
    std::cerr << "  Got: " << out_height << "x" << out_width << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  std::cout << "  Output shape: " << out_batch << "x" << out_channels
            << "x" << out_height << "x" << out_width << std::endl;

  // Sync and convert raw output from NC1HWC2 FP16 to NCHW float32
  ret = rknn_mem_sync(ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
  if (ret < 0) {
    std::cerr << "rknn_mem_sync failed: " << ret << std::endl;
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    return false;
  }

  const __fp16* hw_output = reinterpret_cast<const __fp16*>(output_mem->virt_addr);
  int native_batch = native_output_attr.n_dims > 0 ? native_output_attr.dims[0] : out_batch;
  int native_c1 = native_output_attr.n_dims > 1 ? native_output_attr.dims[1] : 1;
  int native_height = native_output_attr.n_dims > 2 ? native_output_attr.dims[2] : out_height;
  int native_width = native_output_attr.n_dims > 3 ? native_output_attr.dims[3] : out_width;
  int native_c2 = native_output_attr.n_dims > 4 ? native_output_attr.dims[4] : 1;
  int stride_w = native_output_attr.w_stride > 0 ? native_output_attr.w_stride : native_width;

  // Some RKNN conv2d models flatten H*W into a single spatial dimension (e.g., NC1HWC2 with H=1, W=out_h*out_w padded).
  // If the native dims don't match the logical output, re-interpret the spatial extent using stride_w and reshape later.
  int decode_height = native_height;
  int decode_width = native_width;
  bool needs_reshape = false;
  if (decode_height * decode_width != out_height * out_width) {
    decode_height = 1;
    decode_width = stride_w;
    needs_reshape = true;
  }

  std::vector<float> output_nchw = nc1hwc2_fp16_to_nchw(
    hw_output, native_batch, native_c1, decode_height, decode_width,
    stride_w, native_c2, out_channels);

  if (needs_reshape && static_cast<int>(output_nchw.size()) == out_batch * out_channels * decode_height * decode_width) {
    std::vector<float> reshaped(out_batch * out_channels * out_height * out_width, 0.f);
    for (int n = 0; n < out_batch; ++n) {
      for (int c = 0; c < out_channels; ++c) {
        for (int h = 0; h < out_height; ++h) {
          for (int w = 0; w < out_width; ++w) {
            int flat = h * out_width + w;
            size_t src_idx = ((n * out_channels + c) * decode_height + 0) * decode_width + flat;
            size_t dst_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
            if (src_idx < output_nchw.size() && dst_idx < reshaped.size()) {
              reshaped[dst_idx] = output_nchw[src_idx];
            }
          }
        }
      }
    }
    output_nchw.swap(reshaped);
  }

  // Compute expected results on CPU (NCHW) using the same RNG sequence for weights
  std::vector<float> expected = generate_weight_data(config, &rng, low, high);
  std::vector<float> cpu_output(out_channels * out_height * out_width, 0.f);

  if (verbose) {
    std::cout << "\n  Generated Weights:" << std::endl;
  }
  int in_ch_per_group = config.in_channels / config.groups;
  int out_ch_per_group = config.out_channels / config.groups;
  for (int oc = 0; oc < config.out_channels; ++oc) {
    int group = oc / out_ch_per_group;
    if (verbose) {
      std::cout << "    oc=" << oc << " (group " << group << ")" << std::endl;
    }
    for (int ic = 0; ic < in_ch_per_group; ++ic) {
      if (verbose) {
        std::cout << "      ic=" << ic << std::endl;
      }
      for (int ky = 0; ky < config.kernel_h; ++ky) {
        if (verbose) {
          std::cout << "        ky=" << ky << ": ";
        }
        for (int kx = 0; kx < config.kernel_w; ++kx) {
          int wt_idx = oc * in_ch_per_group * config.kernel_h * config.kernel_w
                       + ic * config.kernel_h * config.kernel_w
                       + ky * config.kernel_w + kx;
          if (verbose) {
            std::cout << expected[wt_idx] << " ";
          }
        }
        if (verbose) {
          std::cout << std::endl;
        }
      }
    }
  }

  for (int oc = 0; oc < out_channels; ++oc) {
    for (int oy = 0; oy < out_height; ++oy) {
      for (int ox = 0; ox < out_width; ++ox) {
        float acc = 0.f;
        for (int ic = 0; ic < config.in_channels; ++ic) {
          int in_per_group = config.in_channels / config.groups;
          int out_per_group = config.out_channels / config.groups;
          int ic_group = ic / in_per_group;
          int oc_group = oc / out_per_group;
          if (ic_group != oc_group) continue;  // respect grouping
          int ic_in_group = ic % in_per_group;

          for (int ky = 0; ky < config.kernel_h; ++ky) {
            for (int kx = 0; kx < config.kernel_w; ++kx) {
              // Input indexing
              int in_idx = ic * in_height * in_width + (oy + ky) * in_width + (ox + kx);
              // Weight indexing
              int wt_idx = oc * (config.in_channels / config.groups) * config.kernel_h * config.kernel_w
                          + ic_in_group * config.kernel_h * config.kernel_w + ky * config.kernel_w + kx;

              acc += input_cpu[in_idx] * expected[wt_idx];
            }
          }
        }
        cpu_output[oc * out_height * out_width + oy * out_width + ox] = acc;
      }
    }
  }

  // Print detailed results
  if (verbose) {
    std::cout << "\n  Expected Output (CPU computed):" << std::endl;
    for (int oc = 0; oc < out_channels; ++oc) {
      std::cout << "    Output Channel " << oc << ":" << std::endl;
      for (int h = 0; h < out_height; ++h) {
        std::cout << "      ";
        for (int w = 0; w < out_width; ++w) {
          int idx = oc * out_height * out_width + h * out_width + w;
          std::cout << cpu_output[idx] << " ";
        }
      }
      std::cout << std::endl;
    }

    std::cout << "\n  Actual Output (RKNN):" << std::endl;
    for (int oc = 0; oc < out_channels; ++oc) {
      std::cout << "    Output Channel " << oc << ":" << std::endl;
      for (int h = 0; h < out_height; ++h) {
        std::cout << "      ";
        for (int w = 0; w < out_width; ++w) {
          int idx = oc * out_height * out_width + h * out_width + w;
          std::cout << output_nchw[idx] << " ";
        }
      }
      std::cout << std::endl;
    }
  }

  // Compare results
  bool passed = true;
  float max_error = 0.0f;
  const float abs_tol = 3.5e-2f;      // FP16 rounding + accumulation noise
  const float rel_tol = 5e-3f;        // 0.5% relative tolerance
  for (size_t i = 0; i < output_nchw.size() && i < cpu_output.size(); ++i) {
    float error = std::abs(output_nchw[i] - cpu_output[i]);
    max_error = std::max(max_error, error);

    float tol = std::max(abs_tol, rel_tol * std::max(1.0f, std::abs(cpu_output[i])));
    if (error > tol) {
      std::cerr << "ERROR at index " << i << ": expected=" << cpu_output[i]
                << ", got=" << output_nchw[i] << ", error=" << error << std::endl;
      passed = false;
    }
  }

  rknn_outputs_release(ctx, 1, &output_desc);
  rknn_destroy_mem(ctx, output_mem);
  rknn_destroy(ctx);

  std::cout << "\n";
  if (passed) {
    std::cout << "  ✓ PASSED (max error: " << max_error << ")" << std::endl;
    return true;
  } else {
    std::cout << "  ✗ FAILED" << std::endl;
    return false;
  }
}

int main(int argc, char** argv) {
  std::vector<ConvConfig> test_cases;
  bool verbose = true;
  if (argc >= 3) {
    int start = std::atoi(argv[1]);
    int end = std::atoi(argv[2]);
    if (start < 1) start = 1;
    if (end < start) end = start;
    verbose = false;
    for (int n = start; n <= end; ++n) {
      ConvConfig cfg;
      cfg.model_name = "conv2d_1x1_" + std::to_string(n);
      cfg.out_channels = 6;
      cfg.in_channels = 3;
      cfg.in_height = n;
      cfg.in_width = n;
      cfg.kernel_h = 1;
      cfg.kernel_w = 1;
      cfg.groups = 1;
      cfg.description = "conv2d input shape (1, 3, " + std::to_string(n) + ", " + std::to_string(n) +
                        "), weight shape (6, 3, 1, 1)";
      test_cases.push_back(cfg);
    }
  } else {
    test_cases = {
    // {"conv2d_2x1", 6, 3, 5, 7, 2, 1, 1, "conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 2, 1)"},
    // {"conv2d_2x3", 6, 3, 5, 7, 2, 3, 1, "conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 2, 3)"},
    // {"conv2d_2x5", 6, 3, 5, 7, 2, 5, 1, "conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 2, 5)"},
    {"conv2d_3x1", 6, 3, 5, 7, 3, 1, 1, "conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 3, 1)"},
    {"conv2d_1x1_2x2", 6, 3, 2, 2, 1, 1, 1, "conv2d input shape (1, 3, 2, 2), weight shape (6, 3, 1, 1)"},
    // {"conv2d_3x3", 6, 3, 5, 7, 3, 3, 1, "conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 3, 3)"},
    // {"conv2d_3x3_g3", 6, 3, 5, 7, 3, 3, 3, "conv2d input shape (1, 3, 5, 7), weight shape (6, 1, 3, 3)"},
    // {"conv2d_3x5", 6, 3, 5, 7, 3, 5, 1, "conv2d input shape (1, 3, 5, 7), weight shape (6, 3, 3, 5)"},
    };
  }

  std::cout << "\n" << std::string(80, '#') << std::endl;
  std::cout << "Conv2D Multi-Test Suite" << std::endl;
  std::cout << "Testing " << test_cases.size() << " different convolution shapes" << std::endl;
  std::cout << std::string(80, '#') << std::endl;

  int passed = 0;
  int failed = 0;

  // Run all tests
  for (const auto& config : test_cases) {
    if (run_conv_test(config, verbose)) {
      passed++;
    } else {
      failed++;
    }
  }

  // Print summary
  std::cout << "\n" << std::string(80, '#') << std::endl;
  std::cout << "TEST SUMMARY" << std::endl;
  std::cout << std::string(80, '#') << std::endl;
  std::cout << "Total tests: " << test_cases.size() << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << std::string(80, '#') << std::endl;

  return (failed == 0) ? 0 : 1;
}
