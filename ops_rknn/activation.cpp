#include "rknn_api.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

static bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

static std::string pick_model(const std::string &name, int width) {
  std::string exact = "models/" + name + "_float16_1x" + std::to_string(width) + ".rknn";
  if (file_exists(exact)) return exact;
  std::cerr << "Missing model for width " << width << " (" << exact << ")" << std::endl;
  return "";
}

static void print_tensor(const char *label, const std::vector<float> &vals) {
  std::cout << label;
  for (float v : vals) std::cout << " " << v;
  std::cout << std::endl;
}

static bool fill_input_with_numpy(int total, std::vector<float> &vals) {
  const char *tmpl = R"(python3 - <<'PY'
import numpy as np
np.random.seed(0)
vals = np.random.uniform(-2, 2, {N}).astype('float32')
import sys
sys.stdout.buffer.write(vals.tobytes())
PY
)";
  std::string cmd = tmpl;
  size_t pos = cmd.find("{N}");
  if (pos != std::string::npos) cmd.replace(pos, 3, std::to_string(total));
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) return false;
  size_t read = fread(vals.data(), sizeof(float), total, pipe);
  int rc = pclose(pipe);
  return read == static_cast<size_t>(total) && rc == 0;
}

static void compute_elementwise(const std::vector<float> &input, std::vector<float> &expected,
                                const std::function<float(float)> &fn) {
  for (size_t i = 0; i < input.size(); i++) {
    expected[i] = fn(input[i]);
  }
}

static void compute_softmax(const std::vector<float> &input, std::vector<float> &expected, int width,
                            bool log) {
  if (width <= 0) return;
  int rows = static_cast<int>(input.size()) / width;
  for (int row = 0; row < rows; row++) {
    const float *start = input.data() + row * width;
    float maxv = start[0];
    for (int i = 1; i < width; i++) maxv = std::max(maxv, start[i]);
    float sum = 0.0f;
    for (int i = 0; i < width; i++) sum += std::exp(start[i] - maxv);
    for (int i = 0; i < width; i++) {
      float soft = std::exp(start[i] - maxv) / sum;
      expected[row * width + i] = log ? std::log(soft) : soft;
    }
  }
}

static void compute_softmax(const std::vector<float> &input, std::vector<float> &expected, int width) {
  compute_softmax(input, expected, width, false);
}

static void compute_log_softmax(const std::vector<float> &input, std::vector<float> &expected, int width) {
  compute_softmax(input, expected, width, true);
}

struct ActivationDescriptor {
  const char *name;
  std::function<void(const std::vector<float> &, std::vector<float> &, int)> compute;
};

static const std::vector<ActivationDescriptor> ACTIVATIONS = {
  {"relu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) { return std::max(0.0f, x); });
  }},
  {"leaky_relu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) { return x >= 0 ? x : 0.01f * x; });
  }},
  {"celu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     constexpr float alpha = 1.0f;
     compute_elementwise(in, out, [](float x) {
       return x >= 0 ? x : alpha * (std::exp(x / alpha) - 1.0f);
     });
  }},
  {"selu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     constexpr float alpha = 1.6732632423543772848170429916717f;
     constexpr float scale = 1.0507009873554804934193349852946f;
     compute_elementwise(in, out, [](float x) {
       return x >= 0 ? scale * x : scale * (alpha * (std::exp(x) - 1.0f));
     });
  }},
  {"silu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       float sig = 1.0f / (1.0f + std::exp(-x));
       return x * sig;
     });
  }},
  {"swish", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       float sig = 1.0f / (1.0f + std::exp(-x));
       return x * sig;
     });
  }},
  {"softsign", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) { return x / (1.0f + std::abs(x)); });
  }},
  {"sigmoid", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  }},
  {"logsigmoid", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) { return -std::log1p(std::exp(-x)); });
  }},
  {"hardsigmoid", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       float v = (x + 3.0f) / 6.0f;
       return std::min(1.0f, std::max(0.0f, v));
     });
  }},
  {"softplus", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) { return std::log1p(std::exp(x)); });
  }},
  {"gelu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       constexpr float k = 0.707106781186547524f;
       return 0.5f * x * (1.0f + std::erf(x * k));
     });
  }},
  {"quick_gelu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       float sig = 1.0f / (1.0f + std::exp(-1.702f * x));
       return x * sig;
     });
  }},
  {"elu", [](const std::vector<float> &in, std::vector<float> &out, int) {
     constexpr float alpha = 1.0f;
     compute_elementwise(in, out, [](float x) {
       return x >= 0 ? x : alpha * (std::exp(x) - 1.0f);
     });
  }},
  {"relu6", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       return std::min(6.0f, std::max(0.0f, x));
     });
  }},
  {"hardswish", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       float w = std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
       return x * w;
     });
  }},
  {"mish", [](const std::vector<float> &in, std::vector<float> &out, int) {
     compute_elementwise(in, out, [](float x) {
       float e = std::exp(x);
       float soft = std::log1p(e);
       return x * std::tanh(soft);
     });
  }},
  {"softmax", [](const std::vector<float> &in, std::vector<float> &out, int width) {
     compute_softmax(in, out, width);
  }},
  {"log_softmax", [](const std::vector<float> &in, std::vector<float> &out, int width) {
     compute_log_softmax(in, out, width);
  }},
};

static const ActivationDescriptor *find_activation(const std::string &name) {
  for (const auto &desc : ACTIVATIONS) {
    if (desc.name == name) return &desc;
  }
  return nullptr;
}

static void usage(const char *prog) {
  std::cout << "Usage: " << prog << " <activation> [width]\n";
  std::cout << "Available activations:";
  for (const auto &desc : ACTIVATIONS) std::cout << " " << desc.name;
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }
  std::string activation_name = argv[1];
  int width = 64;
  if (argc >= 3) width = std::atoi(argv[2]);
  if (width <= 0) {
    std::cerr << "width must be positive" << std::endl;
    return 1;
  }
  const ActivationDescriptor *desc = find_activation(activation_name);
  if (!desc) {
    std::cerr << "unknown activation: " << activation_name << std::endl;
    usage(argv[0]);
    return 1;
  }

  std::vector<std::array<int, 4>> shapes = {{1, 1, 1, width}};

  for (auto dims : shapes) {
    int64_t total = dims[0] * dims[1] * dims[2] * dims[3];
    if (total <= 0) {
      std::cerr << "Invalid shape product" << std::endl;
      return -1;
    }
    int flat_w = dims[2] * dims[3];
    std::string model_path = pick_model(activation_name, flat_w);
    if (model_path.empty()) return -1;
    FILE *fp = std::fopen(model_path.c_str(), "rb");
    if (!fp) {
      std::cerr << "failed to open model: " << model_path << std::endl;
      return -1;
    }
    struct stat st {};
    if (stat(model_path.c_str(), &st) != 0) {
      std::cerr << "failed to stat model: " << model_path << std::endl;
      std::fclose(fp);
      return -1;
    }
    size_t model_size = st.st_size;
    std::vector<unsigned char> model_data(model_size);
    if (std::fread(model_data.data(), 1, model_size, fp) != model_size) {
      std::cerr << "failed to read model" << std::endl;
      std::fclose(fp);
      return -1;
    }
    std::fclose(fp);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data.data(), model_size, 0, nullptr);
    if (ret < 0) {
      std::cerr << "rknn_init failed: " << ret << std::endl;
      return -1;
    }

    std::vector<float> input_f(total);
    if (!fill_input_with_numpy(total, input_f)) {
      std::cerr << "failed to generate input with numpy" << std::endl;
      rknn_destroy(ctx);
      return -1;
    }
    std::vector<__fp16> input(total);
    for (int64_t i = 0; i < total; i++) input[i] = static_cast<__fp16>(input_f[i]);

    rknn_input in {};
    in.index = 0;
    in.type = RKNN_TENSOR_FLOAT16;
    in.fmt = RKNN_TENSOR_NHWC;
    in.buf = input.data();
    in.size = static_cast<uint32_t>(input.size() * sizeof(__fp16));
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
    out.want_float = 1;
    out.index = 0;
    ret = rknn_outputs_get(ctx, 1, &out, nullptr);
    if (ret < 0) {
      std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
      rknn_destroy(ctx);
      return -1;
    }

    std::vector<float> expected(total);
    desc->compute(input_f, expected, flat_w);

    float *result = static_cast<float *>(out.buf);
    std::vector<float> output(result, result + total);
    std::cout << "activation: " << activation_name << " shape "
              << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3] << std::endl;
    print_tensor("input  :", input_f);
    print_tensor("output :", output);
    print_tensor("expect :", expected);

    bool ok = true;
    float max_diff = 0.f;
    for (int64_t i = 0; i < total; i++) {
      float diff = std::abs(output[i] - expected[i]);
      max_diff = std::max(max_diff, diff);
      if (diff > 1e-3f) {
        ok = false;
        break;
      }
    }
    std::cout << "NPU result match CPU: " << (ok ? "YES" : "NO") << std::endl;
    std::cout << "Max abs diff: " << max_diff << std::endl;

    rknn_outputs_release(ctx, 1, &out);
    rknn_destroy(ctx);
  }
  return 0;
}
