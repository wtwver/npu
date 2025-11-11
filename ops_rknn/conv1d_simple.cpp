#include "rknn_api.h"
#include <chrono>
#include <cstring>
#include <iomanip>
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

struct TestCase {
    std::string name;
    std::string model_path;
    int batch;
    int in_channels;
    int input_length;
    int out_channels;
    int kernel_size;
};

static std::vector<float> build_input(int batch, int in_channels, int length) {
    std::vector<float> data(static_cast<size_t>(batch) * in_channels * length);
    for (size_t idx = 0; idx < data.size(); ++idx) {
        data[idx] = static_cast<float>((idx % length) + 1);
    }
    return data;
}

static std::vector<float> build_weight(int out_channels, int in_channels, int kernel_size) {
    std::vector<float> data(static_cast<size_t>(out_channels) * in_channels * kernel_size);
    size_t idx = 0;
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                data[idx++] = static_cast<float>(oc + 1);
            }
        }
    }
    return data;
}

static std::vector<float> compute_expected(int batch, int out_channels, int in_channels, int input_length, int kernel_size,
                                           const std::vector<float>& input, const std::vector<float>& weight) {
    int output_length = input_length - kernel_size + 1;
    std::vector<float> expected(static_cast<size_t>(batch) * out_channels * output_length);
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int pos = 0; pos < output_length; ++pos) {
                float acc = 0.f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        size_t input_idx = ((static_cast<size_t>(b) * in_channels + ic) * input_length) + pos + k;
                        size_t weight_idx = ((static_cast<size_t>(oc) * in_channels + ic) * kernel_size) + k;
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
                expected[(static_cast<size_t>(b) * out_channels + oc) * output_length + pos] = acc;
            }
        }
    }
    return expected;
}

static void print_values(const std::string& label, const std::vector<float>& values) {
    std::cout << label;
    for (float v : values) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

static void print_tensor_rows(const std::string& title, const std::vector<float>& values, int batch,
                              int channels, int length, int row_width = 5) {
    std::cout << title << std::endl;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            std::cout << "  Output Channel " << (b * channels + c) << ":" << std::endl;
            size_t base_idx = (static_cast<size_t>(b) * channels + c) * length;
            for (int offset = 0; offset < length; offset += row_width) {
                std::cout << "    ";
                for (int col = 0; col < row_width && offset + col < length; ++col) {
                    std::cout << std::setw(10) << std::fixed << std::setprecision(5)
                              << values[base_idx + offset + col];
                    if (col + 1 < row_width && offset + col + 1 < length) {
                        std::cout << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

static void print_shape(const std::string& label, int a, int b, int c) {
    std::cout << std::setw(12) << std::left << label << ": (" << a << ", " << b << ", " << c << ")" << std::endl;
}

int main() {
    const std::vector<TestCase> test_cases = {
        {"batch-1", "models/conv1d_simple_bs1.rknn", 1, 1, 11, 6, 1},
        // {"batch-8", "models/conv1d_simple_bs8.rknn", 8, 1, 11, 6, 1},
    };

    for (const auto& test : test_cases) {
        std::cout << "\nRunning test: " << test.name << std::endl;
        std::vector<float> input = build_input(test.batch, test.in_channels, test.input_length);
        std::vector<float> weight = build_weight(test.out_channels, test.in_channels, test.kernel_size);

        std::vector<uint8_t> model_data;
        if (!load_model(test.model_path, model_data)) {
            return -1;
        }

        rknn_context ctx = 0;
        int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr);
        if (ret < 0) {
            std::cerr << "rknn_init failed: " << ret << std::endl;
            return -1;
        }

        rknn_input inputs[1];
        std::memset(inputs, 0, sizeof(inputs));

        inputs[0].index = 0;
        inputs[0].buf = input.data();
        inputs[0].size = static_cast<uint32_t>(input.size() * sizeof(float));
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].fmt = RKNN_TENSOR_NCHW;

        ret = rknn_inputs_set(ctx, 1, inputs);
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
        for (size_t i = 0; i < output_elems; ++i) {
            results[i] = result_buf[i];
        }

        rknn_outputs_release(ctx, 1, &output);

        auto expected = compute_expected(test.batch, test.out_channels, test.in_channels, test.input_length,
                                         test.kernel_size, input, weight);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Model     : " << test.model_path << std::endl;
        print_shape("Input shape", test.batch, test.in_channels, test.input_length);
        print_shape("Weight shape", test.out_channels, test.in_channels, test.kernel_size);
        int output_length = test.input_length - test.kernel_size + 1;
        print_shape("Output shape", test.batch, test.out_channels, output_length);
        print_values("Input     : ", input);
        print_values("Weight    : ", weight);
        print_tensor_rows("Expected Output (CPU computed):", expected, test.batch, test.out_channels, output_length);
        print_tensor_rows("Actual Output (RKNN):", results, test.batch, test.out_channels, output_length);
        std::cout << "Elapsed us: " << duration_us << std::endl;

        rknn_destroy(ctx);
    }

    return 0;
}
