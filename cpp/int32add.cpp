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
    printf("Int32 Addition Test\n");
    printf("Expected: [5,5,5,5,5,5,5,5,5,5] + [3,3,3,3,3,3,3,3,3,3] = [8,8,8,8,8,8,8,8,8,8]\n\n");
    
    // Load model file
    const char* model_path = "../int32add.rknn";
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
        int64_t npu_regs_map2[1024 / sizeof(int64_t)];
        memcpy(npu_regs_map2, regmap2, 1024);
        for (int i = 0; i < 300; i++) {
            printf("npu_regs_map2[%d]: 0x%016lx\n", i, npu_regs_map2[i]);
        }
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