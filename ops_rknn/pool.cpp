#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

static const char* get_format_string(int fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
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

// Simple MT19937 implementation for deterministic test inputs.
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
    int kk;
    for (kk = 0; kk < 624 - 397; ++kk) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk + 1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (; kk < 623; ++kk) {
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

enum class PoolType { Max, Min, Avg };

static const char* pool_type_string(PoolType t) {
  switch (t) {
    case PoolType::Max: return "MaxPool";
    case PoolType::Min: return "MinPool";
    case PoolType::Avg: return "AvgPool";
    default: return "Pool";
  }
}

struct PoolConfig {
  std::string model_name;
  PoolType type;
  int batch;
  int channels;
  int height;
  int width;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int dilation_h;
  int dilation_w;
  bool ceil_mode;
  bool count_include_pad;
  bool adaptive;
  int out_h;
  int out_w;
  std::string description;
};

struct Shape {
  int n = 1;
  int c = 1;
  int h = 1;
  int w = 1;
};

static Shape shape_from_attr(const rknn_tensor_attr& attr) {
  Shape s;
  if (attr.n_dims == 4) {
    if (attr.fmt == RKNN_TENSOR_NHWC) {
      s.n = attr.dims[0];
      s.h = attr.dims[1];
      s.w = attr.dims[2];
      s.c = attr.dims[3];
    } else {
      s.n = attr.dims[0];
      s.c = attr.dims[1];
      s.h = attr.dims[2];
      s.w = attr.dims[3];
    }
  } else if (attr.n_dims == 5 && attr.fmt == RKNN_TENSOR_NC1HWC2) {
    s.n = attr.dims[0];
    s.c = attr.dims[1] * attr.dims[4];
    s.h = attr.dims[2];
    s.w = attr.dims[3];
  }
  return s;
}

static std::vector<__fp16> generate_input(const Shape& s, Mt19937* rng, float low = -1.0f, float high = 1.0f) {
  std::vector<__fp16> input(static_cast<size_t>(s.n) * s.c * s.h * s.w);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<__fp16>(mt_uniform(rng, low, high));
  }
  return input;
}

static std::vector<__fp16> nchw_to_nhwc(const std::vector<__fp16>& src, int batch, int channels, int height, int width) {
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
                                               int batch, int c1, int native_h, int logical_h, int logical_w,
                                               int stride_w, int c2, int channels) {
  std::vector<float> dst(static_cast<size_t>(batch) * channels * logical_h * logical_w, 0.f);
  size_t idx = 0;
  int padded_w = stride_w > 0 ? stride_w : logical_w;
  for (int n = 0; n < batch; ++n) {
    for (int g = 0; g < c1; ++g) {
      for (int y = 0; y < native_h; ++y) {
        for (int x = 0; x < padded_w; ++x) {
          for (int c = 0; c < c2; ++c) {
            int channel = g * c2 + c;
            int logical_y = (native_h == 1 && logical_h > 1) ? (x / logical_w) : y;
            int logical_x = (native_h == 1 && logical_h > 1) ? (x % logical_w) : x;
            float val = static_cast<float>(src[idx++]);
            if (channel < channels && logical_y < logical_h && logical_x < logical_w) {
              size_t dst_idx = ((n * channels + channel) * logical_h + logical_y) * logical_w + logical_x;
              dst[dst_idx] = val;
            }
          }
        }
      }
    }
  }
  return dst;
}

static int compute_out_dim(int in, int kernel, int stride, int pad, int dilation, bool ceil_mode) {
  if (in <= 0 || kernel <= 0 || stride <= 0) {
    return 0;
  }
  int kernel_extent = dilation * (kernel - 1) + 1;
  float numer = static_cast<float>(in + 2 * pad - kernel_extent);
  float raw = numer / static_cast<float>(stride);
  int out = ceil_mode ? static_cast<int>(std::ceil(raw)) + 1 : static_cast<int>(std::floor(raw)) + 1;
  if (ceil_mode && (out - 1) * stride >= in + pad) {
    --out;
  }
  return out;
}

static std::vector<float> pool_cpu_standard(const std::vector<float>& input, const Shape& in_shape,
                                            const PoolConfig& cfg, Shape* out_shape) {
  const int stride_h = cfg.stride_h > 0 ? cfg.stride_h : cfg.kernel_h;
  const int stride_w = cfg.stride_w > 0 ? cfg.stride_w : cfg.kernel_w;
  int out_h = compute_out_dim(in_shape.h, cfg.kernel_h, stride_h, cfg.pad_h, cfg.dilation_h, cfg.ceil_mode);
  int out_w = compute_out_dim(in_shape.w, cfg.kernel_w, stride_w, cfg.pad_w, cfg.dilation_w, cfg.ceil_mode);
  if (out_shape) {
    out_shape->n = in_shape.n;
    out_shape->c = in_shape.c;
    out_shape->h = out_h;
    out_shape->w = out_w;
  }

  std::vector<float> output(static_cast<size_t>(in_shape.n) * in_shape.c * out_h * out_w, 0.f);
  for (int n = 0; n < in_shape.n; ++n) {
    for (int c = 0; c < in_shape.c; ++c) {
      for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
          float acc = 0.f;
          if (cfg.type == PoolType::Max) acc = -std::numeric_limits<float>::infinity();
          if (cfg.type == PoolType::Min) acc = std::numeric_limits<float>::infinity();
          int valid = 0;
          int h_start = oy * stride_h - cfg.pad_h;
          int w_start = ox * stride_w - cfg.pad_w;
          for (int kh = 0; kh < cfg.kernel_h; ++kh) {
            int in_y = h_start + kh * cfg.dilation_h;
            if (in_y < 0 || in_y >= in_shape.h) continue;
            for (int kw = 0; kw < cfg.kernel_w; ++kw) {
              int in_x = w_start + kw * cfg.dilation_w;
              if (in_x < 0 || in_x >= in_shape.w) continue;
              size_t idx = ((n * in_shape.c + c) * in_shape.h + in_y) * in_shape.w + in_x;
              float val = input[idx];
              if (cfg.type == PoolType::Max) {
                acc = std::max(acc, val);
              } else if (cfg.type == PoolType::Min) {
                acc = std::min(acc, val);
              } else {
                acc += val;
              }
              ++valid;
            }
          }
          float out_val = acc;
          if (cfg.type == PoolType::Avg) {
            int denom = cfg.count_include_pad ? cfg.kernel_h * cfg.kernel_w : valid;
            if (denom <= 0) denom = 1;
            out_val = acc / static_cast<float>(denom);
          } else if (cfg.type == PoolType::Max && valid == 0) {
            out_val = -std::numeric_limits<float>::infinity();
          } else if (cfg.type == PoolType::Min && valid == 0) {
            out_val = std::numeric_limits<float>::infinity();
          }
          size_t out_idx = ((n * in_shape.c + c) * out_h + oy) * out_w + ox;
          output[out_idx] = out_val;
        }
      }
    }
  }
  return output;
}

static std::vector<float> pool_cpu_adaptive(const std::vector<float>& input, const Shape& in_shape,
                                            const PoolConfig& cfg, Shape* out_shape) {
  int out_h = cfg.out_h > 0 ? cfg.out_h : 1;
  int out_w = cfg.out_w > 0 ? cfg.out_w : 1;
  if (out_shape) {
    out_shape->n = in_shape.n;
    out_shape->c = in_shape.c;
    out_shape->h = out_h;
    out_shape->w = out_w;
  }

  std::vector<float> output(static_cast<size_t>(in_shape.n) * in_shape.c * out_h * out_w, 0.f);
  for (int n = 0; n < in_shape.n; ++n) {
    for (int c = 0; c < in_shape.c; ++c) {
      for (int oy = 0; oy < out_h; ++oy) {
        int h_start = static_cast<int>(std::floor((static_cast<float>(oy) * in_shape.h) / out_h));
        int h_end = static_cast<int>(std::ceil((static_cast<float>(oy + 1) * in_shape.h) / out_h));
        for (int ox = 0; ox < out_w; ++ox) {
          int w_start = static_cast<int>(std::floor((static_cast<float>(ox) * in_shape.w) / out_w));
          int w_end = static_cast<int>(std::ceil((static_cast<float>(ox + 1) * in_shape.w) / out_w));
          float acc = 0.f;
          if (cfg.type == PoolType::Max) acc = -std::numeric_limits<float>::infinity();
          if (cfg.type == PoolType::Min) acc = std::numeric_limits<float>::infinity();
          int valid = 0;
          for (int iy = h_start; iy < h_end; ++iy) {
            for (int ix = w_start; ix < w_end; ++ix) {
              size_t idx = ((n * in_shape.c + c) * in_shape.h + iy) * in_shape.w + ix;
              float val = input[idx];
              if (cfg.type == PoolType::Max) {
                acc = std::max(acc, val);
              } else if (cfg.type == PoolType::Min) {
                acc = std::min(acc, val);
              } else {
                acc += val;
              }
              ++valid;
            }
          }
          float out_val = acc;
          if (cfg.type == PoolType::Avg) {
            int denom = cfg.count_include_pad ? (h_end - h_start) * (w_end - w_start) : valid;
            if (denom <= 0) denom = 1;
            out_val = acc / static_cast<float>(denom);
          }
          size_t out_idx = ((n * in_shape.c + c) * out_h + oy) * out_w + ox;
          output[out_idx] = out_val;
        }
      }
    }
  }
  return output;
}

static std::vector<float> to_float(const std::vector<__fp16>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
  return dst;
}

static std::vector<float> nhwc_fp16_to_nchw(const __fp16* src, const Shape& logical, int stride_w_bytes, int channels_stride) {
  int stride_w = stride_w_bytes > 0 ? stride_w_bytes : logical.w;
  std::vector<float> dst(static_cast<size_t>(logical.n) * logical.c * logical.h * logical.w, 0.f);
  for (int n = 0; n < logical.n; ++n) {
    for (int h = 0; h < logical.h; ++h) {
      for (int w = 0; w < logical.w; ++w) {
        for (int c = 0; c < logical.c; ++c) {
          size_t src_idx = ((n * logical.h + h) * stride_w + w) * channels_stride + c;
          size_t dst_idx = ((n * logical.c + c) * logical.h + h) * logical.w + w;
          dst[dst_idx] = static_cast<float>(src[src_idx]);
        }
      }
    }
  }
  return dst;
}

static std::vector<float> nchw_fp16_to_nchw(const __fp16* src, const Shape& logical, int stride_w_bytes, int channels_stride, int native_h) {
  int stride_w = stride_w_bytes > 0 ? stride_w_bytes : logical.w;
  int h_stride = native_h > 0 ? native_h : logical.h;
  std::vector<float> dst(static_cast<size_t>(logical.n) * logical.c * logical.h * logical.w, 0.f);
  for (int n = 0; n < logical.n; ++n) {
    for (int c = 0; c < logical.c; ++c) {
      for (int h = 0; h < logical.h; ++h) {
        for (int w = 0; w < logical.w; ++w) {
          size_t src_idx = ((n * channels_stride + c) * h_stride + h) * stride_w + w;
          size_t dst_idx = ((n * logical.c + c) * logical.h + h) * logical.w + w;
          dst[dst_idx] = static_cast<float>(src[src_idx]);
        }
      }
    }
  }
  return dst;
}

static void print_tensor_nchw(const char* label, const std::vector<float>& data, const Shape& shape, size_t max_elems) {
  std::cout << "  " << label << " shape (NCHW): " << shape.n << "x" << shape.c << "x" << shape.h << "x" << shape.w << std::endl;
  size_t total = data.size();
  size_t count = std::min(total, max_elems);
  std::cout << "  " << label << " values (" << count;
  if (total > max_elems) {
    std::cout << " of " << total;
  }
  std::cout << "):";
  for (size_t i = 0; i < count; ++i) {
    if (i % shape.w == 0) std::cout << "\n    ";
    std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
  }
  if (total > max_elems) {
    std::cout << "\n    ...";
  }
  std::cout << std::endl;
}

static bool run_pool_test(const PoolConfig& cfg) {
  std::cout << "\n" << std::string(72, '=') << std::endl;
  std::cout << "TEST: " << cfg.description << " [" << pool_type_string(cfg.type) << "]" << std::endl;
  std::cout << std::string(72, '=') << std::endl;

  std::string model_path = "models/" + cfg.model_name + ".rknn";
  std::vector<uint8_t> model_data;
  if (!load_model(model_path, model_data)) {
    return false;
  }

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_init failed: " << ret << std::endl;
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

  rknn_tensor_attr native_output_attr;
  std::memset(&native_output_attr, 0, sizeof(native_output_attr));
  native_output_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native_output_attr, sizeof(native_output_attr));
  if (ret < 0) {
    std::cerr << "rknn_query native output attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  rknn_tensor_attr logical_output_attr;
  std::memset(&logical_output_attr, 0, sizeof(logical_output_attr));
  logical_output_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &logical_output_attr, sizeof(logical_output_attr));
  if (ret < 0) {
    std::cerr << "rknn_query output attr failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  Shape native_in_shape = shape_from_attr(native_input_attr);
  Shape logical_out_shape = shape_from_attr(logical_output_attr);

  std::cout << "  Native input fmt=" << get_format_string(native_input_attr.fmt)
            << " type=" << get_type_string(static_cast<rknn_tensor_type>(native_input_attr.type))
            << " dims=";
  for (uint32_t i = 0; i < native_input_attr.n_dims; ++i) std::cout << " " << native_input_attr.dims[i];
  std::cout << "\n  Native output fmt=" << get_format_string(native_output_attr.fmt)
            << " type=" << get_type_string(static_cast<rknn_tensor_type>(native_output_attr.type))
            << " dims=";
  for (uint32_t i = 0; i < native_output_attr.n_dims; ++i) std::cout << " " << native_output_attr.dims[i];
  std::cout << "\n  Output size=" << native_output_attr.size
            << " size_with_stride=" << native_output_attr.size_with_stride
            << " w_stride=" << native_output_attr.w_stride << std::endl;

  Shape input_shape = native_in_shape;
  // Allow overriding dimensions from the config if provided.
  if (cfg.batch > 0) input_shape.n = cfg.batch;
  if (cfg.channels > 0) input_shape.c = cfg.channels;
  if (cfg.height > 0) input_shape.h = cfg.height;
  if (cfg.width > 0) input_shape.w = cfg.width;

  Mt19937 rng;
  mt_seed(&rng, 0);
  std::vector<__fp16> input_nchw = generate_input(input_shape, &rng, -2.0f, 2.0f);
  std::vector<float> input_float = to_float(input_nchw);

  std::cout << "  Input shape (NCHW): " << input_shape.n << "x" << input_shape.c
            << "x" << input_shape.h << "x" << input_shape.w << std::endl;
  print_tensor_nchw("Input", input_float, input_shape, 64);

  // Prepare input buffer respecting native layout.
  const std::vector<__fp16>* formatted_input = &input_nchw;
  std::vector<__fp16> input_nhwc;
  std::vector<__fp16> input_padded;

  if (native_input_attr.fmt == RKNN_TENSOR_NHWC) {
    input_nhwc = nchw_to_nhwc(input_nchw, input_shape.n, input_shape.c, input_shape.h, input_shape.w);
    int stride_w = native_input_attr.w_stride > 0 ? native_input_attr.w_stride : input_shape.w;
    if (stride_w != input_shape.w) {
      input_padded.assign(static_cast<size_t>(input_shape.n) * input_shape.h * stride_w * input_shape.c, static_cast<__fp16>(0));
      for (int n = 0; n < input_shape.n; ++n) {
        for (int h = 0; h < input_shape.h; ++h) {
          for (int w = 0; w < input_shape.w; ++w) {
            for (int c = 0; c < input_shape.c; ++c) {
              size_t src_idx = ((n * input_shape.h + h) * input_shape.w + w) * input_shape.c + c;
              size_t dst_idx = ((n * input_shape.h + h) * stride_w + w) * input_shape.c + c;
              input_padded[dst_idx] = input_nhwc[src_idx];
            }
          }
        }
      }
      formatted_input = &input_padded;
    } else {
      formatted_input = &input_nhwc;
    }
  } else if (native_input_attr.fmt != RKNN_TENSOR_NCHW) {
    std::cerr << "Unsupported input format: " << native_input_attr.fmt << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  if (native_input_attr.type != RKNN_TENSOR_FLOAT16) {
    std::cerr << "Unsupported input type: " << native_input_attr.type << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  rknn_input input_desc;
  std::memset(&input_desc, 0, sizeof(input_desc));
  input_desc.index = 0;
  input_desc.buf = const_cast<void*>(reinterpret_cast<const void*>(formatted_input->data()));
  input_desc.size = (native_input_attr.size_with_stride > 0) ? native_input_attr.size_with_stride
                                                             : static_cast<uint32_t>(formatted_input->size() * sizeof(__fp16));
  input_desc.type = static_cast<rknn_tensor_type>(native_input_attr.type);
  input_desc.fmt = static_cast<rknn_tensor_format>(native_input_attr.fmt);
  input_desc.pass_through = 1;

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
  std::cout << "  Inference time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " us" << std::endl;

  rknn_output output_desc;
  std::memset(&output_desc, 0, sizeof(output_desc));
  output_desc.want_float = 1;
  output_desc.is_prealloc = 0;
  output_desc.index = 0;
  ret = rknn_outputs_get(ctx, 1, &output_desc, nullptr);
  if (ret < 0) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    rknn_destroy(ctx);
    return false;
  }

  // Compute CPU reference.
  Shape expected_shape;
  std::vector<float> expected;
  if (cfg.adaptive) {
    expected = pool_cpu_adaptive(input_float, input_shape, cfg, &expected_shape);
  } else {
    expected = pool_cpu_standard(input_float, input_shape, cfg, &expected_shape);
  }

  if (logical_out_shape.h != 0 && logical_out_shape.w != 0) {
    if (logical_out_shape.n != expected_shape.n || logical_out_shape.c != expected_shape.c ||
        logical_out_shape.h != expected_shape.h || logical_out_shape.w != expected_shape.w) {
      std::cerr << "Logical output shape mismatch. Model reported "
                << logical_out_shape.n << "x" << logical_out_shape.c << "x"
                << logical_out_shape.h << "x" << logical_out_shape.w
                << " but CPU expected " << expected_shape.n << "x" << expected_shape.c
                << "x" << expected_shape.h << "x" << expected_shape.w << std::endl;
      rknn_outputs_release(ctx, 1, &output_desc);
      rknn_destroy(ctx);
      return false;
    }
  }

  size_t output_count = expected.size();
  size_t returned_count = output_desc.size / sizeof(float);
  if (returned_count < output_count) {
    std::cerr << "Output size too small: got " << returned_count << " floats, expected " << output_count << std::endl;
    rknn_outputs_release(ctx, 1, &output_desc);
    rknn_destroy(ctx);
    return false;
  }
  float* out_ptr = reinterpret_cast<float*>(output_desc.buf);
  std::vector<float> output_nchw(out_ptr, out_ptr + output_count);

  rknn_outputs_release(ctx, 1, &output_desc);
  rknn_destroy(ctx);

  if (output_nchw.size() != expected.size()) {
    std::cerr << "Output size mismatch: got " << output_nchw.size() << " expected " << expected.size() << std::endl;
    return false;
  }

  float max_error = 0.f;
  const float abs_tol = 3.5e-2f;
  const float rel_tol = 5e-3f;
  bool ok = true;
  print_tensor_nchw("Output", output_nchw, expected_shape, 64);
  print_tensor_nchw("Expected", expected, expected_shape, 64);
  for (size_t i = 0; i < expected.size(); ++i) {
    float error = std::abs(output_nchw[i] - expected[i]);
    float tol = std::max(abs_tol, rel_tol * std::max(1.0f, std::abs(expected[i])));
    max_error = std::max(max_error, error);
    if (error > tol) {
      if (ok) {
        std::cerr << "  Mismatch examples:" << std::endl;
      }
      ok = false;
      std::cerr << "    idx " << i << " expected=" << expected[i]
                << " got=" << output_nchw[i] << " error=" << error << " tol=" << tol << std::endl;
      if (i > 100) break;  // avoid flooding
    }
  }

  if (ok) {
    std::cout << "  ✓ PASSED (max error " << max_error << ")" << std::endl;
  } else {
    std::cout << "  ✗ FAILED" << std::endl;
  }
  return ok;
}

int main() {
  std::vector<PoolConfig> test_cases = {
    {"max_pool2d_float16_1x4", PoolType::Max,
     1, 1, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, false, false, false, 0, 0,
     "MaxPool2d 2x2 stride1 on 1x1x4x4"},
    // {"avg_pool2d_float16_1x4", PoolType::Avg,
    //  1, 1, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, false, false, false, 0, 0,
    //  "AvgPool2d 2x2 stride1 on 1x1x4x4"},
    // {"min_pool2d_float16_1x4", PoolType::Min,
    //  1, 1, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, false, false, false, 0, 0,
    //  "MinPool2d 2x2 stride1 on 1x1x4x4"},
    // {"adaptive_avg_pool2d_float16_1x4_to_2x2", PoolType::Avg,
    //  1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, false, false, true, 2, 2,
    //  "AdaptiveAvgPool2d output 2x2 from 1x1x4x4"},
    // {"global_max_pool2d_float16_1x4", PoolType::Max,
    //  1, 1, 4, 4, 4, 4, 1, 1, 0, 0, 1, 1, false, false, false, 0, 0,
    //  "Global MaxPool2d over 1x1x4x4"}
  };

  std::cout << "\n" << std::string(80, '#') << std::endl;
  std::cout << "Pooling Multi-Test Suite" << std::endl;
  std::cout << "Running " << test_cases.size() << " pooling cases" << std::endl;
  std::cout << std::string(80, '#') << std::endl;

  int passed = 0;
  int failed = 0;
  for (const auto& cfg : test_cases) {
    if (run_pool_test(cfg)) {
      ++passed;
    } else {
      ++failed;
    }
  }

  std::cout << "\n" << std::string(80, '#') << std::endl;
  std::cout << "TEST SUMMARY" << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << std::string(80, '#') << std::endl;

  return failed == 0 ? 0 : 1;
}
