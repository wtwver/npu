#include <set>
#include <string>
#include <vector>
#include <sys/time.h>
#include <cmath>
#include <cstring>
#include <typeinfo>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "rknn_api.h"
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <regex>

// Register decoding structures and functions
struct BitField {
    std::string name;
    int low;
    int high;
    std::string type;
};

struct Register {
    std::string name;
    std::string domain;
    uint32_t offset;
    std::vector<BitField> fields;
};

std::map<uint32_t, Register> register_map;

// Simple XML parser for registers.xml
bool parse_register_xml(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open XML file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::string current_domain;
    std::string current_reg_name;
    uint32_t current_offset = 0;
    std::vector<BitField> current_fields;

    std::regex domain_regex("<domain name=\"([^\"]+)\"");
    std::regex reg32_regex("<reg32 offset=\"([^\"]+)\" name=\"([^\"]+)\"");
    std::regex bitfield_regex("<bitfield name=\"([^\"]+)\" (?:pos=\"([^\"]+)\"|low=\"([^\"]+)\" high=\"([^\"]+)\") type=\"([^\"]+)\"");

    while (std::getline(file, line)) {
        std::smatch match;

        // Parse domain
        if (std::regex_search(line, match, domain_regex)) {
            current_domain = match[1].str();
            continue;
        }

        // Parse register
        if (std::regex_search(line, match, reg32_regex)) {
            // Save previous register if exists
            if (!current_reg_name.empty() && current_offset != 0) {
                register_map[current_offset] = {current_reg_name, current_domain, current_offset, current_fields};
            }

            // Parse new register
            std::string offset_str = match[1].str();
            current_reg_name = match[2].str();

            // Convert hex offset to uint32_t
            if (offset_str.substr(0, 2) == "0x") {
                current_offset = std::stoul(offset_str.substr(2), nullptr, 16);
            } else {
                current_offset = std::stoul(offset_str, nullptr, 16);
            }

            current_fields.clear();
            continue;
        }

        // Parse bitfield
        if (std::regex_search(line, match, bitfield_regex)) {
            std::string name = match[1].str();
            std::string type = match[5].str();

            int low, high;

            if (match[2].matched) {
                // Single position bitfield
                low = high = std::stoi(match[2].str());
            } else {
                // Range bitfield
                low = std::stoi(match[3].str());
                high = std::stoi(match[4].str());
            }

            current_fields.push_back({name, low, high, type});
            continue;
        }

        // End of register
        if (line.find("</reg32>") != std::string::npos) {
            if (!current_reg_name.empty() && current_offset != 0) {
                register_map[current_offset] = {current_reg_name, current_domain, current_offset, current_fields};
                current_reg_name.clear();
                current_offset = 0;
                current_fields.clear();
            }
        }
    }

    // Save last register if exists
    if (!current_reg_name.empty() && current_offset != 0) {
        register_map[current_offset] = {current_reg_name, current_domain, current_offset, current_fields};
    }

    file.close();
    return true;
}

// Initialize register definitions (subset of key registers from registers.xml)
void init_register_map() {
    // Try to parse XML file first (check both current and parent directory)
    if (parse_register_xml("../registers.xml") || parse_register_xml("registers.xml")) {
        std::cout << "Successfully loaded " << register_map.size() << " registers from XML" << std::endl;
        return;
    }

    // Fallback to hardcoded definitions if XML parsing fails
    std::cout << "XML parsing failed, using hardcoded register definitions" << std::endl;
    // PC registers
    register_map[0x0000] = {"VERSION", "PC", 0x0000, {}};
    register_map[0x0004] = {"VERSION_NUM", "PC", 0x0004, {}};
    register_map[0x0008] = {"OPERATION_ENABLE", "PC", 0x0008, {{"OP_EN", 0, 0, "boolean"}}};
    register_map[0x0010] = {"BASE_ADDRESS", "PC", 0x0010, {{"PC_SOURCE_ADDR", 4, 31, "uint"}, {"PC_SEL", 0, 0, "uint"}}};
    register_map[0x0020] = {"INTERRUPT_MASK", "PC", 0x0020, {
        {"DMA_WRITE_ERROR", 13, 13, "boolean"}, {"DMA_READ_ERROR", 12, 12, "boolean"},
        {"PPU_1", 11, 11, "boolean"}, {"PPU_0", 10, 10, "boolean"},
        {"DPU_1", 9, 9, "boolean"}, {"DPU_0", 8, 8, "boolean"},
        {"CORE_1", 7, 7, "boolean"}, {"CORE_0", 6, 6, "boolean"},
        {"CNA_CSC_1", 5, 5, "boolean"}, {"CNA_CSC_0", 4, 4, "boolean"},
        {"CNA_WEIGHT_1", 3, 3, "boolean"}, {"CNA_WEIGHT_0", 2, 2, "boolean"},
        {"CNA_FEATURE_1", 1, 1, "boolean"}, {"CNA_FEATURE_0", 0, 0, "boolean"}
    }};

    // CNA registers
    register_map[0x1000] = {"S_STATUS", "CNA", 0x1000, {{"STATUS_1", 16, 17, "uint"}, {"STATUS_0", 0, 1, "uint"}}};
    register_map[0x1004] = {"S_POINTER", "CNA", 0x1004, {{"EXECUTER", 16, 16, "uint"}, {"POINTER", 0, 0, "uint"}}};
    register_map[0x1008] = {"OPERATION_ENABLE", "CNA", 0x1008, {{"OP_EN", 0, 0, "boolean"}}};
    register_map[0x100C] = {"CONV_CON1", "CNA", 0x100C, {
        {"DECONV", 16, 16, "uint"}, {"ARGB_IN", 12, 15, "uint"},
        {"PROC_PRECISION", 7, 9, "uint"}, {"IN_PRECISION", 4, 6, "uint"}, {"CONV_MODE", 0, 3, "uint"}
    }};

    // CORE registers
    register_map[0x3000] = {"S_STATUS", "CORE", 0x3000, {{"STATUS_1", 16, 17, "uint"}, {"STATUS_0", 0, 1, "uint"}}};
    register_map[0x3004] = {"S_POINTER", "CORE", 0x3004, {{"EXECUTER", 16, 16, "uint"}, {"POINTER", 0, 0, "uint"}}};
    register_map[0x3008] = {"OPERATION_ENABLE", "CORE", 0x3008, {{"OP_EN", 0, 0, "boolean"}}};

    // DPU registers
    register_map[0x4000] = {"S_STATUS", "DPU", 0x4000, {{"STATUS_1", 16, 17, "uint"}, {"STATUS_0", 0, 1, "uint"}}};
    register_map[0x4004] = {"S_POINTER", "DPU", 0x4004, {{"EXECUTER", 16, 16, "uint"}, {"POINTER", 0, 0, "uint"}}};
    register_map[0x4008] = {"OPERATION_ENABLE", "DPU", 0x4008, {{"OP_EN", 0, 0, "boolean"}}};
    register_map[0x400C] = {"FEATURE_MODE_CFG", "DPU", 0x400C, {
        {"COMB_USE", 31, 31, "uint"}, {"TP_EN", 30, 30, "uint"}, {"RGP_TYPE", 26, 29, "uint"},
        {"NONALIGN", 25, 25, "uint"}, {"SURF_LEN", 9, 24, "uint"}, {"BURST_LEN", 5, 8, "uint"},
        {"CONV_MODE", 3, 4, "uint"}, {"OUTPUT_MODE", 1, 2, "uint"}, {"FLYING_MODE", 0, 0, "uint"}
    }};
    register_map[0x4010] = {"DATA_FORMAT", "DPU", 0x4010, {
        {"OUT_PRECISION", 29, 31, "uint"}, {"IN_PRECISION", 26, 28, "uint"},
        {"EW_TRUNCATE_NEG", 16, 25, "uint"}, {"BN_MUL_SHIFT_VALUE_NEG", 10, 15, "uint"},
        {"BS_MUL_SHIFT_VALUE_NEG", 4, 9, "uint"}, {"MC_SURF_OUT", 3, 3, "uint"},
        {"PROC_PRECISION", 0, 2, "uint"}
    }};
    register_map[0x4014] = {"OFFSET_PEND", "DPU", 0x4014, {{"OFFSET_PEND", 0, 15, "uint"}}};
    register_map[0x4020] = {"DST_BASE_ADDR", "DPU", 0x4020, {{"DST_BASE_ADDR", 0, 31, "uint"}}};
    register_map[0x4024] = {"DST_SURF_STRIDE", "DPU", 0x4024, {{"DST_SURF_STRIDE", 4, 31, "uint"}}};
    register_map[0x4030] = {"DATA_CUBE_WIDTH", "DPU", 0x4030, {{"WIDTH", 0, 12, "uint"}}};
    register_map[0x4034] = {"DATA_CUBE_HEIGHT", "DPU", 0x4034, {
        {"MINMAX_CTL", 22, 24, "uint"}, {"HEIGHT", 0, 12, "uint"}
    }};
    register_map[0x4038] = {"DATA_CUBE_NOTCH_ADDR", "DPU", 0x4038, {
        {"NOTCH_ADDR_1", 16, 28, "uint"}, {"NOTCH_ADDR_0", 0, 12, "uint"}
    }};
    register_map[0x403C] = {"DATA_CUBE_CHANNEL", "DPU", 0x403C, {
        {"ORIG_CHANNEL", 16, 28, "uint"}, {"CHANNEL", 0, 12, "uint"}
    }};
    register_map[0x4040] = {"BS_CFG", "DPU", 0x4040, {
        {"BS_ALU_ALGO", 16, 19, "uint"}, {"BS_ALU_SRC", 8, 8, "uint"},
        {"BS_RELUX_EN", 7, 7, "uint"}, {"BS_RELU_BYPASS", 6, 6, "uint"},
        {"BS_MUL_PRELU", 5, 5, "uint"}, {"BS_MUL_BYPASS", 4, 4, "uint"},
        {"BS_ALU_BYPASS", 1, 1, "uint"}, {"BS_BYPASS", 0, 0, "uint"}
    }};
    register_map[0x4044] = {"BS_ALU_CFG", "DPU", 0x4044, {{"BS_ALU_OPERAND", 0, 31, "uint"}}};
    register_map[0x4048] = {"BS_MUL_CFG", "DPU", 0x4048, {
        {"BS_MUL_OPERAND", 16, 31, "uint"}, {"BS_MUL_SHIFT_VALUE", 8, 13, "uint"},
        {"BS_TRUNCATE_SRC", 1, 1, "uint"}, {"BS_MUL_SRC", 0, 0, "uint"}
    }};
    register_map[0x404C] = {"BS_RELUX_CMP_VALUE", "DPU", 0x404C, {{"BS_RELUX_CMP_DAT", 0, 31, "uint"}}};
    register_map[0x4050] = {"BS_OW_CFG", "DPU", 0x4050, {
        {"RGP_CNTER", 28, 31, "uint"}, {"TP_ORG_EN", 27, 27, "uint"},
        {"SIZE_E_2", 8, 10, "uint"}, {"SIZE_E_1", 5, 7, "uint"},
        {"SIZE_E_0", 2, 4, "uint"}, {"OD_BYPASS", 1, 1, "uint"}, {"OW_SRC", 0, 0, "uint"}
    }};
    register_map[0x4054] = {"BS_OW_OP", "DPU", 0x4054, {{"OW_OP", 0, 15, "uint"}}};
    register_map[0x4058] = {"WDMA_SIZE_0", "DPU", 0x4058, {
        {"TP_PRECISION", 27, 27, "uint"}, {"SIZE_C_WDMA", 16, 26, "uint"}, {"CHANNEL_WDMA", 0, 12, "uint"}
    }};
    register_map[0x405C] = {"WDMA_SIZE_1", "DPU", 0x405C, {
        {"HEIGHT_WDMA", 16, 28, "uint"}, {"WIDTH_WDMA", 0, 12, "uint"}
    }};
    register_map[0x4060] = {"BN_CFG", "DPU", 0x4060, {
        {"BN_ALU_ALGO", 16, 19, "uint"}, {"BN_ALU_SRC", 8, 8, "uint"},
        {"BN_RELUX_EN", 7, 7, "uint"}, {"BN_RELU_BYPASS", 6, 6, "uint"},
        {"BN_MUL_PRELU", 5, 5, "uint"}, {"BN_MUL_BYPASS", 4, 4, "uint"},
        {"BN_ALU_BYPASS", 1, 1, "uint"}, {"BN_BYPASS", 0, 0, "uint"}
    }};
    register_map[0x4064] = {"BN_ALU_CFG", "DPU", 0x4064, {{"BN_ALU_OPERAND", 0, 31, "uint"}}};
    register_map[0x4068] = {"BN_MUL_CFG", "DPU", 0x4068, {
        {"BN_MUL_OPERAND", 16, 31, "uint"}, {"BN_MUL_SHIFT_VALUE", 8, 13, "uint"},
        {"BN_TRUNCATE_SRC", 1, 1, "uint"}, {"BN_MUL_SRC", 0, 0, "uint"}
    }};
    register_map[0x406C] = {"BN_RELUX_CMP_VALUE", "DPU", 0x406C, {{"BN_RELUX_CMP_DAT", 0, 31, "uint"}}};
    register_map[0x4070] = {"EW_CFG", "DPU", 0x4070, {
        {"EW_CVT_TYPE", 31, 31, "uint"}, {"EW_CVT_ROUND", 30, 30, "uint"},
        {"EW_DATA_MODE", 28, 29, "uint"}, {"EW_EQUAL_EN", 21, 21, "uint"},
        {"EW_BINARY_EN", 20, 20, "uint"}, {"EW_ALU_ALGO", 16, 19, "uint"},
        {"EW_RELUX_EN", 10, 10, "uint"}, {"EW_RELU_BYPASS", 9, 9, "uint"},
        {"EW_OP_CVT_BYPASS", 8, 8, "uint"}, {"EW_LUT_BYPASS", 7, 7, "uint"},
        {"EW_OP_SRC", 6, 6, "uint"}, {"EW_MUL_PRELU", 5, 5, "uint"},
        {"EW_OP_TYPE", 2, 2, "uint"}, {"EW_OP_BYPASS", 1, 1, "uint"},
        {"EW_BYPASS", 0, 0, "uint"}
    }};
    register_map[0x4074] = {"EW_CVT_OFFSET_VALUE", "DPU", 0x4074, {{"EW_OP_CVT_OFFSET", 0, 31, "uint"}}};
    register_map[0x4078] = {"EW_CVT_SCALE_VALUE", "DPU", 0x4078, {
        {"EW_TRUNCATE", 22, 31, "uint"}, {"EW_OP_CVT_SHIFT", 16, 21, "uint"}, {"EW_OP_CVT_SCALE", 0, 15, "uint"}
    }};
    register_map[0x407C] = {"EW_RELUX_CMP_VALUE", "DPU", 0x407C, {{"EW_RELUX_CMP_DAT", 0, 31, "uint"}}};
    register_map[0x4080] = {"OUT_CVT_OFFSET", "DPU", 0x4080, {{"OUT_CVT_OFFSET", 0, 31, "uint"}}};
    register_map[0x4084] = {"OUT_CVT_SCALE", "DPU", 0x4084, {
        {"FP32TOFP16_EN", 16, 16, "uint"}, {"OUT_CVT_SCALE", 0, 15, "uint"}
    }};
    register_map[0x4088] = {"OUT_CVT_SHIFT", "DPU", 0x4088, {
        {"CVT_TYPE", 31, 31, "uint"}, {"CVT_ROUND", 30, 30, "uint"},
        {"MINUS_EXP", 12, 19, "uint"}, {"OUT_CVT_SHIFT", 0, 11, "uint"}
    }};
    register_map[0x4090] = {"EW_OP_VALUE_0", "DPU", 0x4090, {{"EW_OPERAND_0", 0, 31, "uint"}}};
    register_map[0x4094] = {"EW_OP_VALUE_1", "DPU", 0x4094, {{"EW_OPERAND_1", 0, 31, "uint"}}};
    register_map[0x4098] = {"EW_OP_VALUE_2", "DPU", 0x4098, {{"EW_OPERAND_2", 0, 31, "uint"}}};
    register_map[0x409C] = {"EW_OP_VALUE_3", "DPU", 0x409C, {{"EW_OPERAND_3", 0, 31, "uint"}}};
    register_map[0x40A0] = {"EW_OP_VALUE_4", "DPU", 0x40A0, {{"EW_OPERAND_4", 0, 31, "uint"}}};
    register_map[0x40A4] = {"EW_OP_VALUE_5", "DPU", 0x40A4, {{"EW_OPERAND_5", 0, 31, "uint"}}};
    register_map[0x40A8] = {"EW_OP_VALUE_6", "DPU", 0x40A8, {{"EW_OPERAND_6", 0, 31, "uint"}}};
    register_map[0x40AC] = {"EW_OP_VALUE_7", "DPU", 0x40AC, {{"EW_OPERAND_7", 0, 31, "uint"}}};
    register_map[0x40C0] = {"SURFACE_ADD", "DPU", 0x40C0, {{"SURF_ADD", 4, 31, "uint"}}};

    // PPU registers
    register_map[0x6000] = {"S_STATUS", "PPU", 0x6000, {{"STATUS_1", 16, 17, "uint"}, {"STATUS_0", 0, 1, "uint"}}};
    register_map[0x6004] = {"S_POINTER", "PPU", 0x6004, {{"EXECUTER", 16, 16, "uint"}, {"POINTER", 0, 0, "uint"}}};
    register_map[0x6008] = {"OPERATION_ENABLE", "PPU", 0x6008, {{"OP_EN", 0, 0, "boolean"}}};

    // GLOBAL registers
    register_map[0xF008] = {"OPERATION_ENABLE", "GLOBAL", 0xF008, {
        {"PPU_RDMA_OP_EN", 6, 6, "boolean"}, {"PPU_OP_EN", 5, 5, "boolean"},
        {"DPU_RDMA_OP_EN", 4, 4, "boolean"}, {"DPU_OP_EN", 3, 3, "boolean"},
        {"CORE_OP_EN", 2, 2, "boolean"}, {"CNA_OP_EN", 0, 0, "boolean"}
    }};
}

uint32_t extract_bits(uint32_t value, int low, int high) {
    uint32_t mask = ((1U << (high - low + 1)) - 1) << low;
    return (value & mask) >> low;
}

std::string get_target_name(uint64_t reg_cmd) {
    if ((reg_cmd >> 56) & 1) return "PC";
    if ((reg_cmd >> 57) & 1) return "CNA";
    if ((reg_cmd >> 59) & 1) return "CORE";
    if ((reg_cmd >> 60) & 1) return "DPU";
    if ((reg_cmd >> 61) & 1) return "DPU_RDMA";
    if ((reg_cmd >> 62) & 1) return "PPU";
    if ((reg_cmd >> 63) & 1) return "PPU_RDMA";
    return "Unknown";
}

void decode_register(uint64_t reg_cmd, int index) {
    uint32_t val = (reg_cmd >> 16) & 0xffffffff;
    uint16_t low = reg_cmd & 0xffff;
    std::string target = get_target_name(reg_cmd);

    printf("[%3d] [%08x] lsb %016lx - %s", index, 8 * index + 0xffef0000, reg_cmd, target.c_str());

    if (register_map.find(low) != register_map.end()) {
        Register& reg = register_map[low];
        printf(" - %s_%s", reg.domain.c_str(), reg.name.c_str());

        if (!reg.fields.empty() && val != 0) {
            printf(" = ");
            bool first = true;
            for (const auto& field : reg.fields) {
                uint32_t field_value = extract_bits(val, field.low, field.high);
                if (field_value != 0 || field.type == "boolean") {
                    if (!first) printf(" | ");
                    if (field.type == "boolean") {
                        if (field_value) {
                            printf("%s_%s_%s", reg.domain.c_str(), reg.name.c_str(), field.name.c_str());
                        }
                    } else {
                        printf("%s_%s_%s(%u)", reg.domain.c_str(), reg.name.c_str(), field.name.c_str(), field_value);
                    }
                    first = false;
                }
            }
            if (first) {
                printf("0x%08x", val);
            }
        } else {
            printf(" = 0x%08x", val);
        }
    } else {
        printf(" - Unknown register 0x%04x = 0x%08x", low, val);
    }
    printf("\n");
}

// Load test data for int32 tensors
static unsigned char *load_int32_data(rknn_tensor_attr *input_attr, int *input_type, int *input_size,
    int *type_bytes, int index)
{
    // Create test data: input 0 = [5,5,5,5,5,5,5,5,5,5], input 1 = [3,3,3,3,3,3,3,3,3,3]
    int* test = (int*)malloc(10 * sizeof(int));
    
    if (index == 0) {
        for (int i = 0; i < 10; ++i) {
            test[i] = 5;  // First input: all 5s
        }
    } else if (index == 1) {
        for (int i = 0; i < 10; ++i) {
            test[i] = 3;  // Second input: all 3s
        }
    }
    
    // Allocate destination buffer
    int* data = (int*)malloc(10 * sizeof(int));
    
    // Copy the data
    memcpy(data, test, 10 * sizeof(int));
    
    printf("Input %d int32 data: ", index);
    for (int i = 0; i < 10; ++i) {
        printf("%d ", data[i]);
    }
    printf("\n");
    
    // Clean up temporary array
    free(test);
    
    return (unsigned char*)data;
}

static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

void dump_tensor_attr(rknn_tensor_attr* attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i) {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
    
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
           "type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
           attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int main() {
    // Initialize register map for decoding
    init_register_map();

    printf("Int32 Addition Test\n");
    printf("Expected: [5,5,5,5,5,5,5,5,5,5] * [3,3,3,3,3,3,3,3,3,3] = [15,15,15,15,15,15,15,15,15,15]\n\n");
    
    // Load model file
    const char* model_path = "../models/mul_int32_1x1.rknn";
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("Failed to open model file: %s\n", model_path);
        return -1;
    }
    struct stat st;
    stat(model_path, &st);
    size_t model_size = st.st_size;
    void* model_data = malloc(model_size);
    if (!model_data) {
        printf("Failed to allocate memory for model\n");
        fclose(fp);
        return -1;
    }
    fread(model_data, 1, model_size, fp);
    fclose(fp);
    
    // RKNN init
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    
    printf("rknn_init success!\n");
    
    // Get sdk and driver version
    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);
    
    rknn_mem_size mem_size;
    ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("total weight size: %u, total internal size: %u\n", mem_size.total_weight_size, mem_size.total_internal_size);
    printf("total dma used size: %zu\n", (size_t)mem_size.total_dma_allocated_size);
    
    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query error! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }
    
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }
    
    // Prepare input data
    unsigned char* input_data[io_num.n_input];
    int input_type[io_num.n_input];  
    int input_layout[io_num.n_input];
    int input_size[io_num.n_input];
    int type_bytes[io_num.n_input];
    
    for (int i = 0; i < io_num.n_input; i++) {
        input_data[i] = NULL;
        input_type[i] = RKNN_TENSOR_INT32;
        input_layout[i] = RKNN_TENSOR_UNDEFINED;
        input_size[i] = input_attrs[i].n_elems * sizeof(int);
        type_bytes[i] = 4;
    }
    
    // Load input data
    for (int i = 0; i < io_num.n_input; i++) {
        input_data[i] = load_int32_data(&input_attrs[i], &input_type[i], 
                                &input_size[i], &type_bytes[i], i);
        if (!input_data[i]) {
            return -1;
        }
    }
    
    // Setup RKNN inputs
    rknn_input inputs[io_num.n_input];
    memset(inputs, 0, io_num.n_input * sizeof(rknn_input));
    for (int i = 0; i < io_num.n_input; i++) {
        inputs[i].index = i;
        inputs[i].pass_through = 0;
        inputs[i].type = (rknn_tensor_type)input_type[i];
        inputs[i].fmt = RKNN_TENSOR_UNDEFINED;
        inputs[i].buf = input_data[i];
        inputs[i].size = input_size[i];
    }
    
    printf("\nInput summary:\n");
    for (int i = 0; i < io_num.n_input; i++) {
        printf("Input %d: ", i);
        int* int_input = (int*)inputs[i].buf;
        for (int j = 0; j < 10 && j < inputs[i].size/sizeof(int); j++) {
            printf("%d ", int_input[j]);
        }
        printf("\n");
    }
    
    // Set input
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    
    rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    
    // Run inference
    printf("\nBegin inference...\n");
    double total_time = 0;
    for (int i = 0; i < 1; ++i) {
        int64_t start_us = getCurrentTimeUs();
        ret = rknn_run(ctx, NULL);
        int64_t elapse_us = getCurrentTimeUs() - start_us;
        if (ret < 0) {
            printf("rknn run error %d\n", ret);
            return -1;
        }
        total_time += elapse_us / 1000.f;
        printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
    }
    printf("Avg elapse Time = %.3fms\n", total_time / 1);
    printf("Avg FPS = %.3f\n", 1 * 1000.f / total_time);
    
    
    void *regmap2 = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, 3, 0x100001000);
    printf("regmap2: %p\n", regmap2);
    if (regmap2 == MAP_FAILED) {
        perror("mmap regmap2 failed");
    } else {
        printf("\nDecoding regmap2 registers:\n");
        printf("========================================\n");
        int64_t npu_regs_map2[1024 / sizeof(int64_t)];
        memcpy(npu_regs_map2, regmap2, 1024);
        for (int i = 0; i < 300; i++) {
            if (npu_regs_map2[i] != 0) {  // Only decode non-zero register commands
                decode_register(npu_regs_map2[i], i);
            }
        }
        printf("========================================\n");
        // It is good practice to unmap when done
        munmap(regmap2, 1024);
    }


    // Get output
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, io_num.n_output * sizeof(rknn_output));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        outputs[i].want_float = 1;
        outputs[i].index = i;
        outputs[i].is_prealloc = 0;
    }
    
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }
    
    const auto out_elems = output_attrs[0].n_elems; 
    
    printf("\nOutput results (int32):\n");
    const auto buf_data = (float *)outputs[0].buf;
    for (size_t idx = 0; idx < out_elems && idx < 100; idx++) {
        printf("%f ", buf_data[idx]);
        if ((idx + 1) % 10 == 0) printf("\n");
    }
    printf("\n");
    
    // Clean up
    for (int i = 0; i < io_num.n_input; i++) {
        if (input_data[i]) {
            free(input_data[i]);
        }
    }
    
    // Release rknn outputs
    rknn_outputs_release(ctx, io_num.n_output, outputs);
    
    // Release rknn context
    rknn_destroy(ctx);
    
    return 0;
}
