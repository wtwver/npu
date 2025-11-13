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
        std::cout << "Batch " << b << ":" << std::endl;
        for (int c = 0; c < channels; ++c) {
            std::cout << "  Output Channel " << c << ":" << std::endl;
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
        // {"batch-1", "models/conv1d_simple_bs1.rknn", 1, 1, 11, 6, 1},
        {"batch-8", "models/conv1d_simple_bs8.rknn", 8, 1, 11, 6, 1},
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

        rknn_tensor_attr native_input_attr;
        std::memset(&native_input_attr, 0, sizeof(native_input_attr));
        native_input_attr.index = 0;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &native_input_attr, sizeof(native_input_attr));
        if (ret < 0) {
            std::cerr << "rknn_query native input attr failed: " << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        rknn_tensor_attr native_output_attr;
        std::memset(&native_output_attr, 0, sizeof(native_output_attr));
        native_output_attr.index = 0;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native_output_attr, sizeof(native_output_attr));
        if (ret < 0) {
            std::cerr << "rknn_query native output attr failed: " << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
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

        std::vector<__fp16> input_fp16;
        const void* input_buffer = input.data();
        uint32_t input_size_bytes = static_cast<uint32_t>(input.size() * sizeof(float));
        if (native_input_attr.type == RKNN_TENSOR_FLOAT16) {
            input_fp16.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                input_fp16[i] = static_cast<__fp16>(input[i]);
            }
            input_buffer = input_fp16.data();
            input_size_bytes = static_cast<uint32_t>(input_fp16.size() * sizeof(__fp16));
        }

        rknn_input inputs[1];
        std::memset(inputs, 0, sizeof(inputs));

        inputs[0].index = 0;
        inputs[0].buf = const_cast<void*>(input_buffer);
        inputs[0].size = input_size_bytes;
        inputs[0].type = static_cast<rknn_tensor_type>(native_input_attr.type);
        inputs[0].fmt = static_cast<rknn_tensor_format>(native_input_attr.fmt);
        inputs[0].pass_through = 0;

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
        output.want_float = 0;
        output.index = 0;

        ret = rknn_outputs_get(ctx, 1, &output, nullptr);
        if (ret < 0) {
            std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        int output_length = test.input_length - test.kernel_size + 1;
        size_t output_elems = output.size / sizeof(__fp16);
        auto* result_buf = reinterpret_cast<const __fp16*>(output.buf);
        auto width_stride = static_cast<size_t>(native_output_attr.w_stride > 0 ? native_output_attr.w_stride
                                                                                 : output_length);
        if (width_stride < static_cast<size_t>(output_length)) {
            width_stride = output_length;
        }
        size_t total_results = static_cast<size_t>(test.batch) * test.out_channels * output_length;
        std::vector<float> results(total_results, 0.f);
        for (int n = 0; n < test.batch; ++n) {
            for (int c = 0; c < test.out_channels; ++c) {
                size_t src_base = (static_cast<size_t>(n) * test.out_channels + c) * width_stride;
                for (int w = 0; w < output_length; ++w) {
                    size_t src_idx = src_base + static_cast<size_t>(w);
                    if (src_idx >= output_elems) {
                        break;
                    }
                    size_t dst_idx = ((static_cast<size_t>(n) * test.out_channels + c) * output_length) + w;
                    results[dst_idx] = static_cast<float>(result_buf[src_idx]);
                }
            }
        }

        auto expected = compute_expected(test.batch, test.out_channels, test.in_channels, test.input_length,
                                         test.kernel_size, input, weight);

        rknn_outputs_release(ctx, 1, &output);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Model     : " << test.model_path << std::endl;
        print_shape("Input shape", test.batch, test.in_channels, test.input_length);
        print_shape("Weight shape", test.out_channels, test.in_channels, test.kernel_size);
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
