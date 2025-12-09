#include "rknn_api.h"
#include <sys/stat.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

static std::string pick_model(int width) {
  std::string exact = "models/sigmoid_float16_1x" + std::to_string(width) + ".rknn";
  printf("using model %s\n", exact.c_str());
  if (file_exists(exact)) return exact;
  if (width == 1) return exact;
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

int main(int argc, char **argv) {
  std::vector<std::array<int, 4>> shapes;
  shapes.push_back({1, 1, 4, 4});

  for (auto dims : shapes) {
    int total = dims[0] * dims[1] * dims[2] * dims[3];
    if (total <= 0) {
      std::cerr << "Invalid shape product" << std::endl;
      return -1;
    }

    int flat_w = dims[2] * dims[3];
    std::string model_path = pick_model(flat_w);
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
    // Force execution to a single NPU core for deterministic behavior
    // ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0);
    // if (ret < 0) {
    //   std::cerr << "rknn_set_core_mask failed: " << ret << std::endl;
    //   rknn_destroy(ctx);
    //   return -1;
    // }

    std::vector<float> input_f(total);
    if (!fill_input_with_numpy(total, input_f)) {
      std::cerr << "failed to generate input with numpy" << std::endl;
      rknn_destroy(ctx);
      return -1;
    }
    std::vector<__fp16> input(total);
    for (int i = 0; i < total; i++) input[i] = static_cast<__fp16>(input_f[i]);

    rknn_input in {};
    in.index = 0;
    in.type = RKNN_TENSOR_FLOAT16;
    in.fmt = RKNN_TENSOR_NHWC;
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

    rknn_output out {};
    out.want_float = 1;
    out.index = 0;
    ret = rknn_outputs_get(ctx, 1, &out, nullptr);
    if (ret < 0) {
      std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
      rknn_destroy(ctx);
      return -1;
    }

    // Compute expected using FP16-rounded inputs to mirror NPU precision
    std::vector<float> expected(total);
    for (int i = 0; i < total; i++) {
      float x = static_cast<float>(static_cast<__fp16>(input_f[i]));
      expected[i] = 1.f / (1.f + std::exp(-x));
    }

    float *result = static_cast<float *>(out.buf);
    std::vector<float> output(result, result + total);
    std::cout << "Shape " << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3] << std::endl;
    if (dims[2] != 1) {
      std::cout << "Using flattened width " << flat_w << " for model input" << std::endl;
    }
    print_tensor("input  :", input_f);
    print_tensor("output :", output);
    print_tensor("expect :", expected);

    bool ok = true;
    float max_diff = 0.f;
    for (int i = 0; i < total; i++) {
      float diff = std::abs(output[i] - expected[i]);
      if (diff > max_diff) max_diff = diff;
      if (diff > 2e-3f) {
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
