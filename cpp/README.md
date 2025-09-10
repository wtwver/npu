g++ -o ops_int32 ops_int32.cpp -I. -lrknnrt -std=c++11

./ops_int32 mul 1x5
./ops_int32 add 1x3
./ops_int32 sub 1x4

The implementation demonstrates the core RKNN workflow:
1. Model loading and initialization (using local model files)
2. Input tensor setup with proper metadata
3. Inference execution
4. Output retrieval and display
5. Resource cleanup

Supported models:
- mul_int32_1x1.rknn (and size-specific variants)
- add_int32_1x1.rknn (and size-specific variants)
- sub_int32_1x1.rknn (and size-specific variants)