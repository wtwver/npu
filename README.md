fix matmul result is wrong

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

/home/orangepi/ezrknn-toolkit2/rknpu2/examples/rknn_benchmark/install/rknn_benchmark_Linux/rknn_benchmark /home/orangepi/ezrknn-toolkit2/rknpu2/examples/rknn_benchmark/resnet18_for_rk3588.rknn 





# Result

Gem 1: Instructions (contains NPU register commands)
- Size: Varies depending on model complexity
- Contains register configuration for NPU operations
- First 8 bytes are typically 0x0000000000000000
- Following that are 64-bit instructions with register addresses and values
- Instructions target various NPU components (PC, CNA, CORE, DPU, etc.)

Gem 2: Model weights/biases
- Size: Depends on model parameters
- Contains the neural network weights in packed format
- Based on the analysis in hello.c, these are 64-bit values with bitfields:
  - Bits 63-56: Destination flags (PC, CNA, CORE, DPU, etc.)
  - Bits 55: Operation enable flag
  - Bits 48-32: Upper address bits
  - Bits 31-16: Register value
  - Bits 15-0: Register address

Gem 3: Instruction buffer
- Size: Typically smaller than Gem 1
- Contains instruction sequences for NPU execution
- Organized in 40-byte blocks with specific structure:
  - instrs[0]: Always 0
  - instrs[1]: Non-monotonous counter
  - instrs[2]: Operation type (0x1d, 0x60, 0x18, 0xd)
  - instrs[3]: Flags (0x300, 0xc00)
  - instrs[4]: Constant (0x1ffff)
  - instrs[5]: Mode flags (0, 0x100, 0x800, 0x200)
  - instrs[6]: Operation codes (0x7c, 0x1a, 0x45, 0x6a)
  - instrs[7]: Address offset/delta
  - instrs[8]: Address in Gem 2 (weights)

Gem 4-6: Input/Output tensors
- These GEM objects represent the input data (A, B) and output data (C) for matrix multiplication
- Size: Based on tensor dimensions (e.g., for 32x32x32 matmul: 32*32*4 = 4KB per tensor)
- Format: Depends on data type (FP32, INT8, FP16)
- Layout: Typically row-major order

# Explain

NPU uses DRM_IOCTL_GEM_FLINK to get the GEM object.

The RK3588 NPU driver uses GEM (Graphics Execution Manager) objects to manage memory buffers for neural network operations. Each GEM object has a unique handle that can be shared between processes using the DRM_IOCTL_GEM_FLINK ioctl.

```
int fd = open("/dev/dri/card1", O_RDWR);
```

The process for dumping GEM memory:
1. Open the DRM device (/dev/dri/card1)
2. Use DRM_IOCTL_GEM_OPEN to get the GEM handle and size by name
3. Use DRM_IOCTL_RKNPU_MEM_MAP to get the memory mapping offset
4. Use mmap() to map the GEM memory into the process address space
5. Read/dump the memory contents to a file or analyze in memory

Based on the analysis in hello.c, the different GEM objects have specific purposes:
- Gem 1: Contains the instruction stream for the NPU
- Gem 2: Contains model weights and parameters
- Gem 3: Contains detailed instruction sequences
- Gem 4-6: Represent the input tensors (A, B) and output tensor (C) for operations
