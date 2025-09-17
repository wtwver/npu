g++ -o ops_int32 ops_int32.cpp -I../include -lrknnrt -std=c++11
g++ -o ops_int8 ops_int8.cpp -I../include -lrknnrt -std=c++11
g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11

gdb -x ops.gdb --args ./ops_int32 and 1x1
./ops_int32 add 1x3
./ops_int32 sub 1x4
./ops_int32 div 1x5

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
- div_int32_1x1.rknn (and size-specific variants)
- mul_int8_1x1.rknn (and size-specific variants)
- add_int8_1x1.rknn (and size-specific variants)
- sub_int8_1x1.rknn (and size-specific variants)
- div_int8_1x1.rknn (and size-specific variants)
- mul_float16_1x1.rknn (and size-specific variants)
- add_float16_1x1.rknn (and size-specific variants)
- sub_float16_1x1.rknn (and size-specific variants)
- div_float16_1x1.rknn (and size-specific variants)