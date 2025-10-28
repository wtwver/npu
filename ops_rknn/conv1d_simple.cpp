#include "rknn_api.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

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

int main() {
    const std::string model_path = "models/conv1d_simple.rknn";

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

    std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float> weight = {1.f, 0.f, -1.f};

    rknn_input inputs[2];
    std::memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].buf = input.data();
    inputs[0].size = static_cast<uint32_t>(input.size() * sizeof(float));
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NCHW;

    inputs[1].index = 1;
    inputs[1].buf = weight.data();
    inputs[1].size = static_cast<uint32_t>(weight.size() * sizeof(float));
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = RKNN_TENSOR_NCHW;

    ret = rknn_inputs_set(ctx, 2, inputs);
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
    auto* result_buf = reinterpret_cast<float*>(output.buf);
    for (size_t i = 0; i < results.size(); ++i) {
        results[i] = result_buf[i];
    }

    rknn_outputs_release(ctx, 1, &output);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Input     : ";
    for (float v : input) {
        std::cout << v << " ";
    }
    std::cout << "\nWeight    : ";
    for (float v : weight) {
        std::cout << v << " ";
    }
    std::cout << "\nOutput    : ";
    for (float v : results) {
        std::cout << v << " ";
    }

    std::cout << "\nExpected  : ";
    if (input.size() >= weight.size()) {
        for (size_t i = 0; i + weight.size() <= input.size(); ++i) {
            float acc = 0.f;
            for (size_t k = 0; k < weight.size(); ++k) {
                acc += input[i + k] * weight[k];
            }
            std::cout << acc << " ";
        }
    }
    std::cout << "\nElapsed us: " << duration_us << std::endl;

    rknn_destroy(ctx);
    return 0;
}
