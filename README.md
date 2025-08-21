fix matmul result is wrong

# TODO
1. first run convert onnx to rknn
python3 -m rknn.api.rknn_convert -t rk3588 -i /home/orangepi/npu/models/add_1.onnx -o /home/orangepi/npu/models/

2. run rknn_benchmark ensure the ops run correctly
./rknn_benchmark models/add_1.rknn 

3. convert and run for each onnx model

## 📊 ONNX & RKNN Model Testing Results

### tinygrad ops table

| ID | Operation  | CPU Time | NPU FPS  | NPU Latency | NPU Status | CPU Status | Category        | RKNN Support          |
|----|------------|----------|----------|-------------|------------|------------|-----------------|-----------------------|
| 1  | zeros      | -        | 930      | 1.07ms      | ✅         | ✅         | Tensor Creation | pytorch zeros  |
| 2  | ones       | -        | 875      | 1.14ms      | ✅         | ✅         | Tensor Creation | pytorch ones  |
| 3  | empty      | -        | -        | -           | ✅         | ✅         | Tensor Creation | pytorch empty  |
| 4  | full       | -        | -        | -           | ❌         | ✅         | Tensor Creation | ❌ (All outputs are constants) |
| 5  | eye        | -        | -        | -           | ❌         | ❌         | Tensor Creation | ❌ (Eye not in opset 10) |
| 6  | arange     | -        | -        | -           | ❌         | ✅         | Tensor Creation | ❌ (All outputs are constants) |
| 7  | linspace   | -        | -        | -           | ❌         | ✅         | Tensor Creation | ❌ (All outputs are constants) |
| 8  | add        | -        | 8,703    | 0.11ms      | ✅         | ✅         | Arithmetic      | ✅ (Add)              |
| 9  | sub        | -        | 5,155    | 0.19ms      | ✅         | ✅         | Arithmetic      | ✅ (Sub)              |
| 10 | mul        | -        | 14,970   | 0.07ms      | ✅         | ✅         | Arithmetic      | ✅ (Mul)              |
| 11 | div        | -        | 16,502   | 0.06ms      | ✅         | ✅         | Arithmetic      | ✅ (Div)              |
| 12 | mod        | -        | -        | -           | ❌         | ✅         | Arithmetic      | ⚠️ (Scalar divisor only)   |
| 13 | pow        | -        | -        | -           | ✅         | ✅         | Arithmetic      | ✅ (Pow)              |
| 14 | neg        | -        | 18,051   | 0.06ms      | ✅         | ✅         | Arithmetic      | ❌ (Neg)              |
| 15 | eq         | -        | -        | -           | ⚠️         | ✅         | Comparison      | ⚠️ (INT32 only)       |
| 16 | ne         | -        | -        | -           | ⚠️         | ✅         | Comparison      | ⚠️ (INT32 not supported) |
| 17 | lt         | -        | -        | -           | ⚠️         | ✅         | Comparison      | ⚠️ (INT32 not supported) |
| 18 | le         | -        | -        | -           | ⚠️         | ✅         | Comparison      | ⚠️ (INT32 not supported) |
| 19 | gt         | -        | -        | -           | ⚠️         | ✅         | Comparison      | ⚠️ (INT32 not supported) |
| 20 | ge         | -        | -        | -           | ⚠️         | ✅         | Comparison      | ⚠️ (INT32 not supported) |
| 21 | sin        | -        | 4,717    | 0.21ms      | ✅         | ✅         | Math            | ✅ (Sin)              |
| 22 | cos        | -        | 4,755    | 0.21ms      | ✅         | ✅         | Math            | ✅ (Cos)              |
| 23 | exp        | -        | 1,320    | 0.76ms      | ✅         | ✅         | Math            | ✅ (Exp)              |
| 24 | log        | -        | 4,292    | 0.23ms      | ✅         | ✅         | Math            | ✅ (Log)              |
| 25 | sqrt       | -        | 4,857    | 0.21ms      | ✅         | ✅         | Math            | ✅ (Sqrt)             |
| 26 | abs        | -        | 3,702    | 0.27ms      | ✅         | ✅         | Math            | ✅ (Auto-converted to LeakyRelu)         |
| 27 | relu       | -        | 18,868   | 0.05ms      | ✅         | ✅         | Activation      | ✅ (Relu)             |
| 28 | sigmoid    | -        | 3,269    | 0.31ms      | ✅         | ✅         | Activation      | ✅ (Sigmoid)          |



# How to convert rknn model
python3 -m rknn.api.rknn_convert -t rk3588 -i /home/orangepi/npu/models/add_1.onnx -o /home/orangepi/npu/models/


# How to dump GEM

run gdb --args ./matmul_api_demo
break rknn_destroy_mem


when break, run ./hello to dump gem to a file
try a different input format like int8 to fp16 in ./matal_api_demo
diff the difference in dump gem

find out which gem represent instruction, which for input A and inputB and outputC


# How to decode dump gem1

python3 decode.py --xml registers.xml --dump dump/gem1_regdump.bin

```
EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
EMIT(REG_CNA_DCOMP_REGNUM, 0x0);
EMIT(REG_CNA_DCOMP_CTRL, 0x0);
EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_GROUP_LINE_OFF(1));
EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_GROUP_LINE_OFF(1));
EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(5));
EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(1) | CNA_DATA_SIZE0_DATAIN_HEIGHT(4));
EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(63) | CNA_DATA_SIZE1_DATAIN_CHANNEL(64));
EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(1));
EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(4));
EMIT(REG_CNA_WEIGHT_SIZE0, 0x800);
EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(64));
EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(1) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(1) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(32));
EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(1));
EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) | CNA_CVT_CON0_CVT_BYPASS(1));
EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
EMIT(REG_CNA_FC_CON0, 0x0);
EMIT(REG_CNA_FC_CON1, 0x0);
EMIT(REG_CNA_PAD_CON0, 0x0);
EMIT(REG_CNA_FEATURE_DATA_ADDR, 0x0);
EMIT(REG_CNA_FC_CON2, 0x0);
EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(4));
EMIT(REG_CNA_DMA_CON2, 0x0);
EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(1) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(4));
EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(64));
EMIT(REG_CNA_DCOMP_CTRL, 0x0);
EMIT(REG_CNA_DCOMP_REGNUM, 0x0);
EMIT(REG_CNA_DCOMP_ADDR0, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT0, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT1, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT2, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT3, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT4, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT5, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT6, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT7, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT8, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT9, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT10, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT11, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT12, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT13, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT14, 0x0);
EMIT(REG_CNA_DCOMP_AMOUNT15, 0x0);
EMIT(REG_CNA_CVT_CON5, 0x0);
EMIT(REG_CNA_PAD_CON1, 0x0);
EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_QD_EN(1));
EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(3));
EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(31));
EMIT(REG_CORE_CLIP_TRUNCATE, 0x0);
800 3030 0
EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(4));
EMIT(REG_DPU_OFFSET_PEND, 0x0);
EMIT(REG_DPU_DST_BASE_ADDR, 0x0);
EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
EMIT(REG_DPU_DATA_CUBE_WIDTH, 0x0);
EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(3));
EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1(7) | DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0(7));
EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(31) | DPU_DATA_CUBE_CHANNEL_CHANNEL(31));
EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
EMIT(REG_DPU_BS_ALU_CFG, 0x0);
EMIT(REG_DPU_BS_MUL_CFG, 0x0);
EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0x0);
EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(7) | DPU_BS_OW_CFG_SIZE_E_1(7) | DPU_BS_OW_CFG_SIZE_E_0(7));
EMIT(REG_DPU_BS_OW_OP, 0x0);
EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(31));
EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(3));
EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
EMIT(REG_DPU_BN_ALU_CFG, 0x0);
EMIT(REG_DPU_BN_MUL_CFG, 0x0);
EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0x0);
EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0x0);
EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0x0);
EMIT(REG_DPU_OUT_CVT_OFFSET, 0x0);
EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
EMIT(REG_DPU_OUT_CVT_SHIFT, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_0, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_1, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_2, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_3, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_4, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_5, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_6, 0x0);
EMIT(REG_DPU_EW_OP_VALUE_7, 0x0);
EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(8));
1000 40c4 0
EMIT(REG_DPU_LUT_ACCESS_CFG, 0x0);
EMIT(REG_DPU_LUT_ACCESS_DATA, 0x0);
EMIT(REG_DPU_LUT_CFG, 0x0);
EMIT(REG_DPU_LUT_INFO, 0x0);
EMIT(REG_DPU_LUT_LE_START, 0x0);
EMIT(REG_DPU_LUT_LE_END, 0x0);
EMIT(REG_DPU_LUT_LO_START, 0x0);
EMIT(REG_DPU_LUT_LO_END, 0x0);
EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0x0);
EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0x0);
EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0x0);
EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0x0);
```


# Capure benmark rknn

./rknn_benchmark resnet18_for_rk3588.rknn 
./rknn_benchmark models/add_1.rknn 


### Run Inference on RKNN Lite
```python
import sys
sys.path.insert(0, '/tmp/extracted_rknn_lite')
from rknnlite.api import RKNNLite

rknn_lite = RKNNLite()
rknn_lite.load_rknn('model.rknn')
rknn_lite.init_runtime()
outputs = rknn_lite.inference(inputs=[input_data])
rknn_lite.release()
```
