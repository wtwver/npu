#include "rknn_api.h"
#include <iostream>
#include <sys/stat.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>

// Function to check if a file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Function to select the appropriate model based on operation and input size
std::string selectModelPath(const std::string& operation, int size) {
    // Try to find a model specific to this size
    std::stringstream specific_model_path_ss;
    specific_model_path_ss << "models/" << operation << "_float16_1x" << size << ".rknn";
    std::string specific_model_path = specific_model_path_ss.str();
    
    // Check if the specific model exists
    if (fileExists(specific_model_path)) {
        std::cout << "Using specific model: " << specific_model_path << std::endl;
        return specific_model_path;
    }
    
    // Fall back to the generic 1x1 model
    std::stringstream generic_model_path_ss;
    generic_model_path_ss << "models/" << operation << "_float16_1x1.rknn";
    std::string generic_model_path = generic_model_path_ss.str();
    std::cout << "Using generic model: " << generic_model_path << std::endl;
    return generic_model_path;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <operation> 1x<size>" << std::endl;
        std::cerr << "Operations: mul, add, sub, div" << std::endl;
        return -1;
    }
    
    // Parse operation from command line argument
    std::string operation = argv[1];
    if (operation != "mul" && operation != "add" && operation != "sub" && operation != "div" && operation != "floor" && operation != "ceil" && operation != "relu") {
        std::cerr << "Error: Operation must be either 'mul', 'add', 'sub', 'div', 'floor', 'ceil', 'relu'" << std::endl;
        return -1;
    }
    
    // Parse input size from command line argument (format: 1xN)
    std::string input_size_str = argv[2];
    if (input_size_str.substr(0, 2) != "1x") {
        std::cerr << "Error: Input size must be in format 1xN" << std::endl;
        return -1;
    }
    
    int size = std::atoi(input_size_str.substr(2).c_str());
    if (size <= 0) {
        std::cerr << "Error: Size must be a positive integer" << std::endl;
        return -1;
    }
    
    // Select the appropriate model based on operation and input size
    std::string model_path = selectModelPath(operation, size);
    
    // Load model
    FILE* fp = fopen(model_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return -1;
    }
    
    struct stat st;
    stat(model_path.c_str(), &st);
    size_t model_size = st.st_size;
    void* model_data = malloc(model_size);
    if (!model_data) {
        std::cerr << "Failed to allocate memory for model" << std::endl;
        fclose(fp);
        return -1;
    }
    
    fread(model_data, 1, model_size, fp);
    fclose(fp);
    
    // Init RKNN
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        std::cerr << "rknn_init failed! ret=" << ret << std::endl;
        return -1;
    }
    
    // Create input arrays of specified size
    std::vector<__fp16> input0(size);
    std::vector<__fp16> input1(size);
    std::vector<float> results(size);
    
    // Initialize inputs with sample values
    for (int i = 0; i < size; i++) {
        // input0[i] = static_cast<__fp16>(i + 1.1);  // 1.1, 2.1, 3.1, ...
        input0[i] = static_cast<__fp16>(-85.0);  // 1.1, 2.1, 3.1, ...
        input1[i] = static_cast<__fp16>(68.0);      // All 2s
    }
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    
        rknn_input inputs[1];
        memset(inputs, 0, 1 * sizeof(rknn_input));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_FLOAT16;
        inputs[0].fmt = RKNN_TENSOR_NCHW;
        inputs[0].buf = input0.data();
        inputs[0].size = size * sizeof(__fp16);
        
        // inputs[1].index = 1;
        // inputs[1].type = RKNN_TENSOR_FLOAT16;
        // inputs[1].fmt = RKNN_TENSOR_NCHW;
        // inputs[1].buf = input1.data();
        // inputs[1].size = size * sizeof(__fp16);
        
        ret = rknn_inputs_set(ctx, 1, inputs);
        if (ret < 0) {
            std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }
        
        // Run inference
        ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            std::cerr << "rknn_run failed! ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }
        
        // Get output
        rknn_output output;
        memset(&output, 0, sizeof(output));
        output.want_float = 1;
        output.index = 0;
        
        ret = rknn_outputs_get(ctx, 1, &output, NULL);
        if (ret < 0) {
            std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return -1;
        }
        
        // Store result
        float* result_buf = (float*)output.buf;
        for (int i = 0; i < size; i++) {
            results[i] = result_buf[i];
        }
        
        // Release output for this iteration
        rknn_outputs_release(ctx, 1, &output);
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Print inputs with better column alignment
    std::cout << "Input0  : ";
    for (int i = 0; i < size; i++) {
        if (i > 0) std::cout << " ";
        if (operation == "div") {
            // For division, we might have float results, so we need more space
            std::cout.width(4);
        } else {
            std::cout.width(2);
        }
        std::cout << input0[i];
    }
    std::cout << std::endl;
    
    std::cout << "Input1  : ";
    for (int i = 0; i < size; i++) {
        if (i > 0) std::cout << " ";
        if (operation == "div") {
            // For division, we might have float results, so we need more space
            std::cout.width(4);
        } else {
            std::cout.width(2);
        }
        std::cout << input1[i];
    }
    std::cout << std::endl;
    
    // Print results with better column alignment
    std::cout << "Results : ";
    for (int i = 0; i < size; i++) {
        if (i > 0) std::cout << " ";
        if (operation == "div") {
            // For division, we need to format floats properly
            if (results[i] == (int)results[i]) {
                // Whole number
                std::cout.width(4);
                std::cout << (int)results[i];
            } else {
                // Decimal number
                std::cout.width(4);
                std::cout.precision(1);
                std::cout << std::fixed << results[i];
            }
        } else {
            // For other operations, check if result is a whole number
            if (results[i] == (int)results[i]) {
                // Whole number
                std::cout.width(2);
                std::cout << (int)results[i];
            } else {
                // Decimal number
                std::cout.width(4);
                std::cout.precision(1);
                std::cout << std::fixed << results[i];
            }
        }
    }
    std::cout << std::endl;
    
    // Show expected results with better column alignment
    std::cout << "Expected: ";
    if (operation == "mul") {
        for (int i = 0; i < size; i++) {
            if (i > 0) std::cout << " ";
            float expected = static_cast<float>(input0[i]) * static_cast<float>(input1[i]);
            if (expected == (int)expected) {
                // Whole number
                std::cout.width(4);
                std::cout << (int)expected;
            } else {
                // Decimal number
                std::cout.width(4);
                std::cout.precision(1);
                std::cout << std::fixed << expected;
            }
        }
    } else if (operation == "add") {
        for (int i = 0; i < size; i++) {
            if (i > 0) std::cout << " ";
            float expected = static_cast<float>(input0[i]) + static_cast<float>(input1[i]);
            if (expected == (int)expected) {
                // Whole number
                std::cout.width(4);
                std::cout << (int)expected;
            } else {
                // Decimal number
                std::cout.width(4);
                std::cout.precision(1);
                std::cout << std::fixed << expected;
            }
        }
    } else if (operation == "sub") {
        for (int i = 0; i < size; i++) {
            if (i > 0) std::cout << " ";
            float expected = static_cast<float>(input0[i]) - static_cast<float>(input1[i]);
            if (expected == (int)expected) {
                // Whole number
                std::cout.width(4);
                std::cout << (int)expected;
            } else {
                // Decimal number
                std::cout.width(4);
                std::cout.precision(1);
                std::cout << std::fixed << expected;
            }
        }
    } else { // div
        for (int i = 0; i < size; i++) {
            if (i > 0) std::cout << " ";
            float expected = static_cast<float>(input0[i]) / static_cast<float>(input1[i]);
            if (expected == (int)expected) {
                // Whole number
                std::cout.width(4);
                std::cout << (int)expected;
            } else {
                // Decimal number
                std::cout.width(4);
                std::cout.precision(1);
                std::cout << std::fixed << expected;
            }
        }
    }
    std::cout << std::endl;
    
    // Show timing information
    std::cout << "Processing time for " << size << " elements (" << operation << "): " << duration.count() << " microseconds" << std::endl;
    
    // Cleanup
    rknn_destroy(ctx);
    return 0;
}