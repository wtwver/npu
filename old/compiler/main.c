#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include "rknn_compiler.h"

namespace rknn {

class Logging {
public:
    static int32_t s_GlobalLogLevel;
};

int32_t Logging::s_GlobalLogLevel = 0;

class RKNNCompiler {
public:
    RKNNCompiler();
    ~RKNNCompiler();
    int32_t build();

private:
    std::string model_type_;
    std::string model_path_;
    std::string model_params_;
    std::string output_path_;
    std::string platform_;
    std::vector<std::string> input_names_;
    bool compress_;
    bool onnx_opt_;
    bool conv_eltwise_activation_fuse_;
    bool global_fuse_;
    uint32_t multi_core_model_mode_;
    bool output_optimize_;
    bool enable_argb_group_;
    bool op_group_sram_opt_;
    bool op_group_nbuf_opt_;
    bool enable_flash_attention_;
    bool enable_rnn_loop_;
    bool safe_fuse_;
    bool layout_match_;
};

// External functions (assumed to be defined elsewhere)
extern int32_t get_log_level(const char* env_var, const char* prop, int32_t default_value);
extern bool is_valid_path(const std::string& path);
extern bool is_regular_file(const std::string& path);
extern void log_message(int32_t level, const char* format, ...);
extern std::string get_file_type(const std::string& path);
extern void process_strings(const std::string& input, std::string& output);
extern void append_string(std::vector<std::string>& vec, const std::string& str);
extern void process_input_names(const std::string& input, std::vector<std::string>& output);
extern void configure_multi_core_mode(std::string& config, uint32_t mode);

RKNNCompiler::RKNNCompiler() {
    // Initialize logging level
    Logging::s_GlobalLogLevel = get_log_level("RKNN_LOG_LEVEL", "persist.vendor.rknn.log.level", -1);
    if (Logging::s_GlobalLogLevel < 0) {
        Logging::s_GlobalLogLevel = 0; // Default value
    }
}

int32_t RKNNCompiler::build() {
    // Validate model path
    std::string temp_path;
    process_strings(model_path_, temp_path);
    if (!is_valid_path(temp_path)) {
        log_message(0, "model path illegal, path exists");
        process_strings(model_path_, temp_path);
        if (!is_regular_file(temp_path)) {
            log_message(0, "model path illegal, is_regular");
            return 1;
        }
    }

    // Set model type
    model_type_ = "RKNPU";
    std::string temp_str = model_type_;
    model_type_ = temp_str;

    // Set model path
    temp_str = model_path_;
    model_path_ = temp_str;

    // Process platform
    temp_str = platform_;
    process_strings(temp_str, platform_);

    // Process input names
    process_strings(model_path_, temp_str);
    if (!input_names_.empty()) {
        append_string(input_names_, temp_str);
    }
    process_strings(model_params_, temp_str);
    process_strings(output_path_, temp_str);

    // Parse configuration parameters
    compress_ = false;
    onnx_opt_ = true;
    conv_eltwise_activation_fuse_ = true;
    global_fuse_ = true;
    multi_core_model_mode_ = 7;
    output_optimize_ = true;
    enable_argb_group_ = false;
    op_group_sram_opt_ = false;
    op_group_nbuf_opt_ = false;
    enable_flash_attention_ = false;
    enable_rnn_loop_ = true;
    safe_fuse_ = false;
    layout_match_ = true;

    if (model_params_ != nullptr) {
        char* params = strdup(model_params_.c_str());
        char* token = strstr(params, "compress=");
        if (token) sscanf(token, "compress=%d", &compress_);
        token = strstr(params, "onnx_opt=");
        if (token) sscanf(token, "onnx_opt=%d", &onnx_opt_);
        token = strstr(params, "conv_eltwise_activation_fuse=");
        if (token) sscanf(token, "conv_eltwise_activation_fuse=%d", &conv_eltwise_activation_fuse_);
        token = strstr(params, "global_fuse=");
        if (token) sscanf(token, "global_fuse=%d", &global_fuse_);
        token = strstr(params, "multi-core-model-mode=");
        if (token) sscanf(token, "multi-core-model-mode=%d", &multi_core_model_mode_);
        token = strstr(params, "output_optimize=");
        if (token) sscanf(token, "output_optimize=%d", &output_optimize_);
        token = strstr(params, "enable_argb_group=");
        if (token) sscanf(token, "enable_argb_group=%d", &enable_argb_group_);
        token = strstr(params, "enable_layout_match=");
        if (token) sscanf(token, "enable_layout_match=%d", &layout_match_);
        token = strstr(params, "op_group_sram_opt=");
        if (token) sscanf(token, "op_group_sram_opt=%d", &op_group_sram_opt_);
        token = strstr(params, "op_group_nbuf_opt=");
        if (token) sscanf(token, "op_group_nbuf_opt=%d", &op_group_nbuf_opt_);
        token = strstr(params, "enable_flash_attention=");
        if (token) sscanf(token, "enable_flash_attention=%d", &enable_flash_attention_);
        token = strstr(params, "enable_rnn_loop=");
        if (token) sscanf(token, "enable_rnn_loop=%d", &enable_rnn_loop_);
        token = strstr(params, "safe_fuse=");
        if (token) sscanf(token, "safe_fuse=%d", &safe_fuse_);
        free(params);
    }

    // Validate file type
    std::string file_type = get_file_type(model_path_);
    if (file_type == "WINF") {
        compress_ = false;
        file_type = get_file_type(model_path_);
        if (file_type == "WINE") {
            compress_ = false;
        }
    } else if (file_type == "TREH") {
        enable_rnn_loop_ = false;
    }

    // Validate and set configuration flags
    if (compress_ > 1) {
        log_message(1, "compress std format error, std_len is 0 or 1, fallback to default stds!");
        compress_ = compress_ & 1;
    }
    if (conv_eltwise_activation_fuse_ > 1) {
        log_message(1, "fuse format error, std_len is 0 or 1, fallback to default stds!");
        conv_eltwise_activation_fuse_ = conv_eltwise_activation_fuse_ & 1;
    }
    if (global_fuse_ > 1) {
        log_message(1, "fuse format error, std_len is 0 or 1, fallback to default stds!");
        global_fuse_ = global_fuse_ & 1;
    }
    if (op_group_sram_opt_ > 1) {
        log_message(1, "fuse format error, std_len is 0 or 1, fallback to default stds!");
        op_group_sram_opt_ = op_group_sram_opt_ & 1;
    }
    if (op_group_nbuf_opt_ > 1) {
        log_message(1, "-op_group_nbuf_opt only support 0 or 1, default to 1 now!");
        op_group_nbuf_opt_ = op_group_nbuf_opt_ & 1;
    }
    if (safe_fuse_ > 1) {
        log_message(1, "safe_fuse only support 0 or 1, fallback to default value!");
        safe_fuse_ = safe_fuse_ & 1;
    }

    // Configure multi-core mode
    configure_multi_core_mode(model_type_, multi_core_model_mode_);

    // Log configuration
    log_message(2, "compress = %d, conv_eltwise_activation_fuse = %d, global_fuse = %d, "
                   "multi-core-model-mode = %d, output_optimize = %d, layout_match = %d, "
                   "enable_argb_group = %d, op_group_sram_opt = %d, enable_flash_attention = %d, "
                   "op_group_nbuf_opt = %d, safe_fuse = %d\n",
                   compress_, conv_eltwise_activation_fuse_, global_fuse_, multi_core_model_mode_,
                   output_optimize_, layout_match_, enable_argb_group_, op_group_sram_opt_,
                   enable_flash_attention_, op_group_nbuf_opt_, safe_fuse_);

    // Perform build operation (implementation assumed elsewhere)
    return 0; // Success
}

RKNNCompiler::~RKNNCompiler() {
    // Cleanup code (if any)
}

} // namespace rknn