#include "rknn_api.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

struct Mt19937 {
    uint32_t mt[624];
    int index;
};

static void mt_seed(Mt19937 *rng, uint32_t seed) {
    rng->mt[0] = seed;
    for (int i = 1; i < 624; ++i) {
        rng->mt[i] = 1812433253U * (rng->mt[i-1] ^ (rng->mt[i-1] >> 30)) + static_cast<uint32_t>(i);
    }
    rng->index = 624;
}

static uint32_t mt_extract(Mt19937 *rng) {
    const uint32_t mag01[2] = {0U, 0x9908b0dfU};
    if (rng->index >= 624) {
        for (int kk = 0; kk < 624 - 397; ++kk) {
            uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk+1] & 0x7fffffffU);
            rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
        }
        for (int kk = 624 - 397; kk < 623; ++kk) {
            uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk+1] & 0x7fffffffU);
            rng->mt[kk] = rng->mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
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

static float mt_uniform(Mt19937 *rng, float low, float high) {
    const double a = static_cast<double>(mt_extract(rng) >> 5);
    const double b = static_cast<double>(mt_extract(rng) >> 6);
    const double random = (a * 67108864.0 + b) / 9007199254740992.0;
    return static_cast<float>(low + (high - low) * random);
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

struct TestCase {
    std::string name;
    std::string model_path;
    int batch;
    int in_channels;
    int input_length;
    int out_channels;
    int weight_in_channels;
    int kernel_size;
    int groups;
    std::string data_dir;

    TestCase(const std::string& n, const std::string& model, int b, int ic, int in_len,
             int oc, int wic, int k, int g = 0, const std::string& dir = "")
      : name(n), model_path(model), batch(b), in_channels(ic), input_length(in_len),
        out_channels(oc), weight_in_channels(wic), kernel_size(k),
        groups(g > 0 ? g : (wic > 0 ? ic / wic : 1)), data_dir(dir) {}

    int in_channels_per_group() const { return weight_in_channels; }
};

struct ComparisonConfig {
    float abs_tol = 5e-2f;
    float rel_tol = 5e-3f;
    std::string expected_rel_path = "expected.bin";
};

static bool read_text_file(const std::string& path, std::string& out) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return true;
}

static std::string extract_object_block(const std::string& json, const std::string& key) {
    std::string token = "\"" + key + "\"";
    size_t pos = json.find(token);
    if (pos == std::string::npos) {
        return "";
    }
    size_t brace_start = json.find('{', pos);
    if (brace_start == std::string::npos) {
        return "";
    }
    int depth = 0;
    for (size_t i = brace_start; i < json.size(); ++i) {
        if (json[i] == '{') {
            ++depth;
        } else if (json[i] == '}') {
            --depth;
            if (depth == 0) {
                return json.substr(brace_start, i - brace_start + 1);
            }
        }
    }
    return "";
}

static bool extract_float_value(const std::string& src, const std::string& key, float& value) {
    std::string token = "\"" + key + "\"";
    size_t pos = src.find(token);
    if (pos == std::string::npos) {
        return false;
    }
    pos = src.find(':', pos);
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) {
        ++pos;
    }
    size_t end = pos;
    auto is_number_char = [](char ch) {
        return std::isdigit(static_cast<unsigned char>(ch)) || ch == '+' || ch == '-' || ch == '.' || ch == 'e' || ch == 'E';
    };
    while (end < src.size() && is_number_char(src[end])) {
        ++end;
    }
    try {
        value = std::stof(src.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

static bool extract_string_value(const std::string& src, const std::string& key, std::string& value) {
    std::string token = "\"" + key + "\"";
    size_t pos = src.find(token);
    if (pos == std::string::npos) {
        return false;
    }
    pos = src.find(':', pos);
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) {
        ++pos;
    }
    if (pos >= src.size() || src[pos] != '"') {
        return false;
    }
    ++pos;
    size_t end = pos;
    while (end < src.size() && src[end] != '"') {
        ++end;
    }
    if (end >= src.size()) {
        return false;
    }
    value = src.substr(pos, end - pos);
    return true;
}

static bool load_case_metadata(const std::string& metadata_path, ComparisonConfig& config) {
    std::string contents;
    if (!read_text_file(metadata_path, contents)) {
        return false;
    }
    std::string expected_block = extract_object_block(contents, "expected");
    if (expected_block.empty()) {
        return false;
    }
    extract_float_value(expected_block, "rtol", config.rel_tol);
    extract_float_value(expected_block, "atol", config.abs_tol);
    extract_string_value(expected_block, "path", config.expected_rel_path);
    return true;
}

static bool load_fp32_tensor(const std::string& path, size_t element_count, std::vector<float>& out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    stream.seekg(0, std::ios::end);
    std::streamoff size = stream.tellg();
    const std::streamoff expected = static_cast<std::streamoff>(element_count * sizeof(float));
    if (size != expected || size <= 0) {
        return false;
    }
    stream.seekg(0, std::ios::beg);
    out.resize(element_count);
    if (!stream.read(reinterpret_cast<char*>(out.data()), expected)) {
        return false;
    }
    return true;
}

static std::vector<float> build_input(int batch, int in_channels, int length, Mt19937 &rng) {
    std::vector<float> data(static_cast<size_t>(batch) * in_channels * length);
    for (size_t idx = 0; idx < data.size(); ++idx) {
        data[idx] = mt_uniform(&rng, -2.0f, 2.0f);
    }
    return data;
}

static std::vector<float> build_default_weight(int out_channels, int in_per_group, int kernel_size) {
    std::vector<float> data(static_cast<size_t>(out_channels) * in_per_group * kernel_size);
    for (int oc = 0; oc < out_channels; ++oc) {
        float value = static_cast<float>(oc + 1);
        for (int ic = 0; ic < in_per_group; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                size_t idx = ((static_cast<size_t>(oc) * in_per_group + ic) * kernel_size) + k;
                data[idx] = value;
            }
        }
    }
    return data;
}

static float fp16_bits_to_float(uint16_t bits) {
    __fp16 half;
    std::memcpy(&half, &bits, sizeof(half));
    return static_cast<float>(half);
}

static bool load_fp16_tensor(const std::string& path, size_t element_count, std::vector<float>& out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    stream.seekg(0, std::ios::end);
    std::streamoff size = stream.tellg();
    const std::streamoff expected = static_cast<std::streamoff>(element_count * sizeof(uint16_t));
    if (size != expected || size < 0) {
        return false;
    }
    stream.seekg(0);
    std::vector<uint16_t> raw(element_count);
    if (!stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(expected))) {
        return false;
    }
    out.resize(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        out[i] = fp16_bits_to_float(raw[i]);
    }
    return true;
}

static std::vector<float> compute_expected(int batch, int out_channels, int in_channels, int in_per_group,
                                           int input_length, int kernel_size, int groups,
                                           const std::vector<float>& input, const std::vector<float>& weight) {
    int output_length = input_length - kernel_size + 1;
    std::vector<float> expected(static_cast<size_t>(batch) * out_channels * output_length);
    int out_per_group = groups > 0 ? out_channels / groups : out_channels;
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            int group_idx = out_per_group > 0 ? oc / out_per_group : 0;
            for (int pos = 0; pos < output_length; ++pos) {
                float acc = 0.f;
                for (int ic = 0; ic < in_per_group; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int input_channel = group_idx * in_per_group + ic;
                        size_t input_idx = ((static_cast<size_t>(b) * in_channels + input_channel) * input_length) + pos + k;
                        size_t weight_idx = ((static_cast<size_t>(oc) * in_per_group + ic) * kernel_size) + k;
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
            }
            std::cout << std::endl;
        }
    }
}

static void print_shape(const std::string& label, int a, int b, int c) {
    std::cout << std::setw(12) << std::left << label << ": (" << a << ", " << b << ", " << c << ")" << std::endl;
}

static std::vector<float> nc1hwc2_fp16_to_nchw(const __fp16* src, size_t src_elems,
                                               int native_batch, int c1, int height,
                                               int native_width, int stride_w,
                                               int c2, int logical_batch, int channels, int valid_width) {
    int padded_w = stride_w > 0 ? stride_w : native_width;
    if (padded_w <= 0) {
        padded_w = native_width;
    }
    std::vector<float> dst(static_cast<size_t>(logical_batch) * channels * height * valid_width, 0.f);
    const size_t total_logical_slices = static_cast<size_t>(logical_batch) * channels;
    size_t idx = 0;
    // RKNN packs NC1HWC2 lane-first: each (c1,c2) lane stores native_batch logical slices.
    for (int g = 0; g < c1; ++g) {
        for (int c_slot = 0; c_slot < c2; ++c_slot) {
            size_t lane_id = static_cast<size_t>(g) * c2 + c_slot;
            for (int lane_batch = 0; lane_batch < native_batch; ++lane_batch) {
                size_t logical_slice = lane_id * static_cast<size_t>(native_batch) + lane_batch;
                bool valid_slice = logical_slice < total_logical_slices;
                int batch_idx = 0;
                int channel_idx = 0;
                if (valid_slice) {
                    batch_idx = static_cast<int>(logical_slice / channels);
                    channel_idx = static_cast<int>(logical_slice % channels);
                }
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < padded_w; ++x) {
                        if (idx >= src_elems) {
                            return dst;
                        }
                        float val = static_cast<float>(src[idx++]);
                        if (!valid_slice || x >= valid_width) {
                            continue;
                        }
                        size_t dst_idx = (((static_cast<size_t>(batch_idx) * channels + channel_idx) * height + y)
                                          * valid_width) + x;
                        dst[dst_idx] = val;
                    }
                }
            }
        }
    }
    return dst;
}

struct DiffStats {
    bool matched = true;
    size_t worst_index = 0;
    float expected_value = 0.f;
    float actual_value = 0.f;
    float max_abs_error = 0.f;
    float max_rel_error = 0.f;
};

static DiffStats compare_tensors(const std::vector<float>& expected, const std::vector<float>& actual,
                                 float abs_tol, float rel_tol) {
    DiffStats stats;
    if (expected.size() != actual.size()) {
        stats.matched = false;
        stats.max_abs_error = std::numeric_limits<float>::infinity();
        stats.max_rel_error = std::numeric_limits<float>::infinity();
        return stats;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        float exp_val = expected[i];
        float act_val = actual[i];
        float diff = std::fabs(exp_val - act_val);
        float denom = std::max(1e-12f, std::fabs(exp_val));
        float rel_err = diff / denom;
        if (diff >= stats.max_abs_error) {
            stats.max_abs_error = diff;
            stats.expected_value = exp_val;
            stats.actual_value = act_val;
            stats.worst_index = i;
        }
        stats.max_rel_error = std::max(stats.max_rel_error, rel_err);
        float tol = std::max(abs_tol, rel_tol * std::max(1.f, std::fabs(exp_val)));
        if (diff > tol) {
            stats.matched = false;
        }
    }
    return stats;
}

static void decode_ncw_index(size_t index, int batch, int channels, int width,
                             int& n_idx, int& c_idx, int& w_idx) {
    size_t per_batch = static_cast<size_t>(channels) * width;
    n_idx = per_batch > 0 ? static_cast<int>(index / per_batch) : 0;
    size_t rem = per_batch > 0 ? index % per_batch : 0;
    c_idx = width > 0 ? static_cast<int>(rem / width) : 0;
    w_idx = width > 0 ? static_cast<int>(rem % width) : 0;
}

static void dump_channel_slice(const std::string& label, const std::vector<float>& tensor,
                               int batch, int channels, int width,
                               int batch_idx, int channel_idx) {
    if (batch_idx < 0 || batch_idx >= batch || channel_idx < 0 || channel_idx >= channels) {
        return;
    }
    size_t base = ((static_cast<size_t>(batch_idx) * channels + channel_idx) * width);
    std::cout << label;
    for (int w = 0; w < width; ++w) {
        std::cout << " " << std::fixed << std::setprecision(6) << tensor[base + w];
    }
    std::cout << std::endl;
}

static void print_conv1d_input(const std::vector<float>& data,
                               int batch, int channels, int width) {
    std::cout << "Generated Input:" << std::endl;
    for (int n = 0; n < batch; ++n) {
        std::cout << "  batch=" << n << std::endl;
        for (int c = 0; c < channels; ++c) {
            std::cout << "    channel=" << c << ": ";
            size_t base = (static_cast<size_t>(n) * channels + c) * width;
            for (int w = 0; w < width; ++w) {
                std::cout << " " << std::fixed << std::setprecision(5) << data[base + w];
            }
            std::cout << std::endl;
        }
    }
}

static void print_conv1d_kernel(const std::vector<float>& data,
                                int out_channels, int in_channels, int ksize) {
    std::cout << "Generated Kernel:" << std::endl;
    for (int oc = 0; oc < out_channels; ++oc) {
        std::cout << "  out_channel=" << oc << std::endl;
        for (int ic = 0; ic < in_channels; ++ic) {
            std::cout << "    in_channel=" << ic << ": ";
            size_t base = (static_cast<size_t>(oc) * in_channels + ic) * ksize;
            for (int k = 0; k < ksize; ++k) {
                std::cout << " " << std::fixed << std::setprecision(5) << data[base + k];
            }
            std::cout << std::endl;
        }
    }
}

int main() {
    const std::vector<TestCase> test_cases = {
        // {"conv1d-i-1-1-11-w-6-1-1", "models/conv1d-i-1-1-11-w-6-1-1.rknn", 1, 1, 11, 6, 1, 1, 0, "conv1d_simple_data/conv1d-i-1-1-11-w-6-1-1"},
        {"conv1d-i-1-1-11-w-6-1-2", "models/conv1d-i-1-1-11-w-6-1-2.rknn", 1, 1, 11, 6, 1, 2, 0, "conv1d_simple_data/conv1d-i-1-1-11-w-6-1-2"},
        // {"conv1d-i-1-1-11-w-6-1-5", "models/conv1d-i-1-1-11-w-6-1-5.rknn", 1, 1, 11, 6, 1, 5, 0, "conv1d_simple_data/conv1d-i-1-1-11-w-6-1-5"},
        // {"conv1d-i-1-3-11-w-6-3-1", "models/conv1d-i-1-3-11-w-6-3-1.rknn", 1, 3, 11, 6, 3, 1, 0, "conv1d_simple_data/conv1d-i-1-3-11-w-6-3-1"},
        // {"conv1d-i-1-3-11-w-6-3-2", "models/conv1d-i-1-3-11-w-6-3-2.rknn", 1, 3, 11, 6, 3, 2, 0, "conv1d_simple_data/conv1d-i-1-3-11-w-6-3-2"},
        // {"conv1d-i-1-3-11-w-6-3-5", "models/conv1d-i-1-3-11-w-6-3-5.rknn", 1, 3, 11, 6, 3, 5, 0, "conv1d_simple_data/conv1d-i-1-3-11-w-6-3-5"},
        // {"conv1d-i-1-3-11-w-6-1-5-g3", "models/conv1d-i-1-3-11-w-6-1-5-g3.rknn", 1, 3, 11, 6, 1, 5, 3, "conv1d_simple_data/conv1d-i-1-3-11-w-6-1-5-g3"},
        // {"conv1d-i-8-1-11-w-6-1-1", "models/conv1d-i-8-1-11-w-6-1-1.rknn", 8, 1, 11, 6, 1, 1, 0, "conv1d_simple_data/conv1d-i-8-1-11-w-6-1-1"},
        // {"conv1d-i-8-1-11-w-6-1-2", "models/conv1d-i-8-1-11-w-6-1-2.rknn", 8, 1, 11, 6, 1, 2, 0, "conv1d_simple_data/conv1d-i-8-1-11-w-6-1-2"},
        // {"conv1d-i-8-1-11-w-6-1-5", "models/conv1d-i-8-1-11-w-6-1-5.rknn", 8, 1, 11, 6, 1, 5, 0, "conv1d_simple_data/conv1d-i-8-1-11-w-6-1-5"},
        // {"conv1d-i-8-3-11-w-6-3-1", "models/conv1d-i-8-3-11-w-6-3-1.rknn", 8, 3, 11, 6, 3, 1, 0, "conv1d_simple_data/conv1d-i-8-3-11-w-6-3-1"},
        // {"conv1d-i-8-3-11-w-6-3-2", "models/conv1d-i-8-3-11-w-6-3-2.rknn", 8, 3, 11, 6, 3, 2, 0, "conv1d_simple_data/conv1d-i-8-3-11-w-6-3-2"},
        // {"conv1d-i-8-3-11-w-6-3-5", "models/conv1d-i-8-3-11-w-6-3-5.rknn", 8, 3, 11, 6, 3, 5, 0, "conv1d_simple_data/conv1d-i-8-3-11-w-6-3-5"},
        // {"conv1d-i-8-3-11-w-6-1-5-g3", "models/conv1d-i-8-3-11-w-6-1-5-g3.rknn", 8, 3, 11, 6, 1, 5, 3, "conv1d_simple_data/conv1d-i-8-3-11-w-6-1-5-g3"},
    };

    for (const auto& test : test_cases) {
        std::cout << "\nRunning test: " << test.name << std::endl;
        int output_length = test.input_length - test.kernel_size + 1;
        size_t output_count = static_cast<size_t>(test.batch) * test.out_channels * output_length;
        ComparisonConfig compare_cfg;
        bool metadata_loaded = false;
        if (!test.data_dir.empty()) {
            std::string metadata_path = test.data_dir + "/metadata.json";
            metadata_loaded = load_case_metadata(metadata_path, compare_cfg);
            std::cout << "  " << (metadata_loaded ? "Loaded metadata" : "Unable to parse metadata")
                      << " from " << metadata_path << std::endl;
        }
        std::string expected_path;
        std::vector<float> expected;
        bool expected_from_file = false;
        if (!test.data_dir.empty()) {
            expected_path = test.data_dir + "/" + compare_cfg.expected_rel_path;
            expected_from_file = load_fp32_tensor(expected_path, output_count, expected);
            std::cout << "  " << (expected_from_file ? "Loaded expected output" : "Unable to load expected output")
                      << " from " << expected_path << std::endl;
        }
        std::cout << "  Comparison tolerances -> abs_tol=" << compare_cfg.abs_tol
                  << ", rel_tol=" << compare_cfg.rel_tol << std::endl;

        size_t input_count = static_cast<size_t>(test.batch) * test.in_channels * test.input_length;
        std::vector<float> input;
        bool loaded_input = false;
        if (!test.data_dir.empty()) {
            std::string input_path = test.data_dir + "/input.bin";
            loaded_input = load_fp16_tensor(input_path, input_count, input);
            std::cout << (loaded_input ? "  Loaded deterministic input" : "  Unable to load deterministic input")
                      << " from " << input_path << std::endl;
        }
        int in_per_group = test.in_channels_per_group();
        size_t weight_count = static_cast<size_t>(test.out_channels) * static_cast<size_t>(in_per_group) * test.kernel_size;
        std::vector<float> weight;
        bool loaded_weight = false;
        if (!test.data_dir.empty()) {
            std::string weight_path = test.data_dir + "/kernel.bin";
            loaded_weight = load_fp16_tensor(weight_path, weight_count, weight);
            std::cout << (loaded_weight ? "  Loaded deterministic weight" : "  Unable to load deterministic weight")
                      << " from " << weight_path << std::endl;
        }
        Mt19937 rng;
        bool needs_random = !loaded_input || !loaded_weight;
        if (needs_random) {
            mt_seed(&rng, 0);
        }
        if (!loaded_input) {
            input = build_input(test.batch, test.in_channels, test.input_length, rng);
        }
        if (!loaded_weight) {
            weight = build_default_weight(test.out_channels, in_per_group, test.kernel_size);
        }

        print_conv1d_input(input, test.batch, test.in_channels, test.input_length);
        print_conv1d_kernel(weight, test.out_channels, in_per_group, test.kernel_size);

        if (!expected_from_file) {
            expected = compute_expected(test.batch, test.out_channels, test.in_channels, in_per_group,
                                        test.input_length, test.kernel_size, test.groups, input, weight);
            std::cout << "  Expected output generated on CPU for comparison" << std::endl;
        }

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

        rknn_tensor_attr input_attr;
        std::memset(&input_attr, 0, sizeof(input_attr));
        input_attr.index = 0;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
        if (ret < 0) {
            std::cerr << "rknn_query input attr failed: " << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }
        std::cout << "  Logical input type: " << get_type_string(static_cast<rknn_tensor_type>(input_attr.type))
                  << ", format: " << get_format_string(input_attr.fmt) << std::endl;
        std::cout << "  Logical input dims:";
        for (uint32_t i = 0; i < input_attr.n_dims; ++i) {
            std::cout << " " << input_attr.dims[i];
        }
        std::cout << std::endl;

        std::vector<float> logical_input = input;
        if (input_attr.fmt == RKNN_TENSOR_NHWC) {
            size_t elements = static_cast<size_t>(test.batch) * test.input_length * test.in_channels;
            logical_input.assign(elements, 0.f);
            for (int n = 0; n < test.batch; ++n) {
                for (int c = 0; c < test.in_channels; ++c) {
                    for (int x = 0; x < test.input_length; ++x) {
                        size_t src_idx = (static_cast<size_t>(n) * test.in_channels + c) * test.input_length + x;
                        size_t dst_idx = ((static_cast<size_t>(n) * 1) * test.input_length + x) * test.in_channels + c;
                        logical_input[dst_idx] = input[src_idx];
                    }
                }
            }
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

        uint32_t native_output_size = native_output_attr.size_with_stride > 0
                                          ? native_output_attr.size_with_stride
                                          : native_output_attr.size;
        rknn_tensor_mem* native_output_mem = nullptr;
        auto destroy_output_mem = [&](void) {
            if (native_output_mem) {
                rknn_destroy_mem(ctx, native_output_mem);
                native_output_mem = nullptr;
            }
        };
        native_output_mem = rknn_create_mem(ctx, native_output_size);
        if (!native_output_mem) {
            std::cerr << "rknn_create_mem failed for native output" << std::endl;
            rknn_destroy(ctx);
            return -1;
        }

        rknn_tensor_attr output_mem_attr = native_output_attr;
        output_mem_attr.pass_through = 1;
        ret = rknn_set_io_mem(ctx, native_output_mem, &output_mem_attr);
        if (ret < 0) {
            std::cerr << "rknn_set_io_mem for output failed: " << ret << std::endl;
            destroy_output_mem();
            rknn_destroy(ctx);
            return -1;
        }

        std::vector<__fp16> input_fp16;
        const void* input_buffer = logical_input.data();
        uint32_t input_size_bytes = static_cast<uint32_t>(logical_input.size() * sizeof(float));
        if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            input_fp16.resize(logical_input.size());
            for (size_t i = 0; i < logical_input.size(); ++i) {
                input_fp16[i] = static_cast<__fp16>(logical_input[i]);
            }
            input_buffer = input_fp16.data();
            input_size_bytes = static_cast<uint32_t>(input_fp16.size() * sizeof(__fp16));
        }

        rknn_input inputs[1];
        std::memset(inputs, 0, sizeof(inputs));

        inputs[0].index = 0;
        inputs[0].buf = const_cast<void*>(input_buffer);
        inputs[0].size = input_size_bytes;
        inputs[0].type = static_cast<rknn_tensor_type>(input_attr.type);
        inputs[0].fmt = static_cast<rknn_tensor_format>(input_attr.fmt);
        inputs[0].pass_through = 0;

        ret = rknn_inputs_set(ctx, 1, inputs);
        if (ret < 0) {
            std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
            destroy_output_mem();
            rknn_destroy(ctx);
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        ret = rknn_run(ctx, nullptr);
        if (ret < 0) {
            std::cerr << "rknn_run failed: " << ret << std::endl;
            destroy_output_mem();
            rknn_destroy(ctx);
            return -1;
        }

        rknn_output output_desc;
        std::memset(&output_desc, 0, sizeof(output_desc));
        output_desc.want_float = 0;
        output_desc.is_prealloc = 1;
        output_desc.index = 0;
        output_desc.buf = native_output_mem->virt_addr;
        output_desc.size = native_output_size;

        ret = rknn_outputs_get(ctx, 1, &output_desc, nullptr);
        if (ret < 0) {
            std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
            destroy_output_mem();
            rknn_destroy(ctx);
            return -1;
        }

        ret = rknn_mem_sync(ctx, native_output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0) {
            std::cerr << "rknn_mem_sync failed: " << ret << std::endl;
            rknn_outputs_release(ctx, 1, &output_desc);
            destroy_output_mem();
            rknn_destroy(ctx);
            return -1;
        }

        size_t output_elems = native_output_size / sizeof(__fp16);
        auto* result_buf = reinterpret_cast<const __fp16*>(native_output_mem->virt_addr);
        const char* dump_path = std::getenv("RKNN_DUMP_NATIVE");
        if (dump_path && *dump_path) {
            std::ofstream dump_stream(dump_path, std::ios::binary);
            if (dump_stream) {
                dump_stream.write(reinterpret_cast<const char*>(native_output_mem->virt_addr), native_output_size);
                std::cout << "  Dumped native output to " << dump_path << std::endl;
            } else {
                std::cerr << "  Unable to open dump path: " << dump_path << std::endl;
            }
        }

        int native_batch = native_output_attr.n_dims > 0 ? native_output_attr.dims[0] : test.batch;
        int native_c1 = native_output_attr.n_dims > 1 ? native_output_attr.dims[1]
                                                      : (test.out_channels + 1) / 2;
        int native_height = native_output_attr.n_dims > 2 ? native_output_attr.dims[2] : 1;
        int native_width = native_output_attr.n_dims > 3 ? native_output_attr.dims[3] : output_length;
        int native_c2 = native_output_attr.n_dims > 4 ? native_output_attr.dims[4] : 1;
        int padded_width = native_output_attr.w_stride > 0 ? native_output_attr.w_stride : native_width;

        std::vector<float> results;
        if (native_output_attr.fmt == RKNN_TENSOR_NC1HWC2) {
            results = nc1hwc2_fp16_to_nchw(result_buf, output_elems,
                                           native_batch, native_c1,
                                           native_height, native_width,
                                           padded_width, native_c2,
                                           test.batch, test.out_channels,
                                           output_length);
        } else {
            size_t logical_count = static_cast<size_t>(test.batch) * test.out_channels * output_length;
            size_t copy_count = std::min(logical_count, output_elems);
            results.assign(logical_count, 0.f);
            for (size_t i = 0; i < copy_count; ++i) {
                results[i] = static_cast<float>(result_buf[i]);
            }
        }

        rknn_outputs_release(ctx, 1, &output_desc);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Model     : " << test.model_path << std::endl;
        print_shape("Input shape", test.batch, test.in_channels, test.input_length);
        print_shape("Weight shape", test.out_channels, in_per_group, test.kernel_size);
        print_shape("Output shape", test.batch, test.out_channels, output_length);
        print_values("Input     : ", input);
        print_values("Weight    : ", weight);
        print_tensor_rows("Expected Output:", expected, test.batch, test.out_channels, output_length);
        print_tensor_rows("Actual Output (RKNN):", results, test.batch, test.out_channels, output_length);
        DiffStats stats = compare_tensors(expected, results, compare_cfg.abs_tol, compare_cfg.rel_tol);
        if (expected.size() == results.size()) {
            int mismatch_batch = 0;
            int mismatch_channel = 0;
            int mismatch_x = 0;
            decode_ncw_index(stats.worst_index, test.batch, test.out_channels, output_length,
                             mismatch_batch, mismatch_channel, mismatch_x);
            std::cout << "Comparison: idx=" << stats.worst_index
                      << " (n=" << mismatch_batch
                      << ", c=" << mismatch_channel
                      << ", x=" << mismatch_x << ")" << std::endl;
            std::cout << "           expected=" << stats.expected_value
                      << ", actual=" << stats.actual_value
                      << ", max_abs_error=" << stats.max_abs_error
                      << ", max_rel_error=" << stats.max_rel_error << std::endl;
            if (!stats.matched) {
                int c1_slot = native_c2 > 0 ? mismatch_channel / native_c2 : 0;
                int c2_slot = native_c2 > 0 ? mismatch_channel % native_c2 : 0;
                size_t native_offset = ((((static_cast<size_t>(mismatch_batch) * native_c1) + c1_slot)
                                         * native_height) * padded_width + mismatch_x) * native_c2 + c2_slot;
                std::cout << "  NC1HWC2 offsets: c1=" << c1_slot
                          << ", c2=" << c2_slot
                          << ", padded_w=" << padded_width
                          << ", native_offset=" << native_offset << std::endl;
                dump_channel_slice("  Expected slice:", expected, test.batch, test.out_channels,
                                   output_length, mismatch_batch, mismatch_channel);
                dump_channel_slice("  Actual slice  :", results, test.batch, test.out_channels,
                                   output_length, mismatch_batch, mismatch_channel);
            }
        } else {
            std::cout << "Comparison: tensor size mismatch (expected_elems=" << expected.size()
                      << ", actual_elems=" << results.size() << ")" << std::endl;
        }
        std::cout << "Status    : match " << (stats.matched ? "YES" : "NO")
                  << " (abs_tol=" << compare_cfg.abs_tol
                  << ", rel_tol=" << compare_cfg.rel_tol << ")" << std::endl;
        std::cout << "Elapsed us: " << duration_us << std::endl;

        destroy_output_mem();
        rknn_destroy(ctx);
    }

    return 0;
}
