#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include <sys/ioctl.h>
#include <libdrm/drm.h>
#include "rknpu-ioctl.h"
#include "npu_interface.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "npu_cna.h"
#include "npu_dpu.h"
#include "npu_matmul.h"
#include "rkt_registers.h"

#define MAX_M 384 
#define MAX_K 4096 
#define MAX_N 4096 

int create_flink_name(int fd, uint32_t handle, uint32_t *flink_name) {
    struct drm_gem_flink flink_req = {
        .handle = handle,
        .name = 0
    };

    int ret = ioctl(fd, DRM_IOCTL_GEM_FLINK, &flink_req);
    if (ret < 0) {
        printf("ERROR: DRM_IOCTL_GEM_FLINK failed: %s (%d)\n", strerror(errno), errno);
        return ret;
    }

    *flink_name = flink_req.name;
    printf("SUCCESS: Created flink name %u for handle %u\n", *flink_name, handle);
    return 0;
}

int open_gem_by_flink(int fd, uint32_t flink_name, uint32_t *handle, uint64_t *size) {
    struct drm_gem_open gopen = {
        .name = flink_name,
        .handle = 0,
        .size = 0
    };
    
    int ret = ioctl(fd, DRM_IOCTL_GEM_OPEN, &gopen);
    if (ret < 0) {
        printf("DRM_IOCTL_GEM_OPEN failed: %s\n", strerror(errno));
        return ret;
    }
    
    *handle = gopen.handle;
    *size = gopen.size;
    printf("Opened GEM object with flink name %u: handle=%u, size=%lu\n", 
           flink_name, *handle, *size);
    return 0;
}

float rand_float() {
    return rand()/(float)RAND_MAX;
}

void matmul_fp32(int m, int k, int n, _Float16 *src0 , _Float16 *src1, float* dst) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        float sum = 0;
        for (int l = 0; l < k; l++) {
          sum += src0[i*k + l] * src1[j*k + l];
        }
       dst[i*n + j] = sum;
      }
    }
}

int weight_fp16(int C, int k, int c) {
  int dst =0;
  int kpg = ((k-1)/16);
  int cpg = ((c-1)/32);
  dst = ((cpg*32)*16)+ (kpg*16*C);
  dst = dst + ((c-1)%32) + (((k-1)%16)*32);
  return dst;
}

int feature_data(int C, int H, int W, int C2, int c, int h, int w) {
  int plane = (c-1)/C2;
  int src = plane * H * W * C2;
  int offset = (c-1) % C2;
  int pos = src + C2 * ((h-1) * W + (w-1)) + offset;
  return pos;
}

static inline uint64_t EMIT(uint32_t reg, uint32_t value){
  uint32_t target = rkt_get_target(reg) + 0x1;

  uint64_t packed_value = 0;
  packed_value = ((uint64_t)target) << 48;
  packed_value |= ((uint64_t)value) << 16;
  packed_value |= (uint64_t)reg;

  return packed_value;
}

// only using cna & core, dpu outputs to memory
void gen_matmul_task(uint64_t *ops, npu_cna_desc *cna_desc, npu_core_desc *core_desc, npu_dpu_desc *dpu_desc) {
  uint32_t value;

  printf("DEBUG: gen_matmul_task called\n");
  printf("DEBUG: cna_desc->datain_channel=%u, cna_desc->weight_kernels=%u\n", 
         cna_desc->datain_channel, cna_desc->weight_kernels);
  printf("DEBUG: Writing ops[0] to ops[10]\n");

  ops[0] = EMIT(REG_DPU_S_POINTER, 
      DPU_S_POINTER_POINTER_PP_MODE(1) | 
      DPU_S_POINTER_EXECUTER_PP_EN(1) | 
      DPU_S_POINTER_POINTER_PP_EN(1)
    );
    
  ops[1] = EMIT(REG_CNA_CONV_CON1, 
      CNA_CONV_CON1_PROC_PRECISION(cna_desc->proc_precision) | 
      CNA_CONV_CON1_IN_PRECISION(cna_desc->in_precision) |
      CNA_CONV_CON1_CONV_MODE(cna_desc->conv_mode)
    );
  ops[2] = EMIT(REG_CNA_CONV_CON2,
      CNA_CONV_CON2_KERNEL_GROUP(cna_desc->kernel_groups) |
      CNA_CONV_CON2_FEATURE_GRAINS(cna_desc->feature_grains)
    );
  ops[3] = EMIT(REG_CNA_CONV_CON3, 
      CNA_CONV_CON3_CONV_Y_STRIDE(cna_desc->conv_y_stride) | 
      CNA_CONV_CON3_CONV_X_STRIDE(cna_desc->conv_x_stride)
    );
  ops[4] = EMIT(REG_CNA_DATA_SIZE0, 
      CNA_DATA_SIZE0_DATAIN_WIDTH(cna_desc->datain_width) | 
      CNA_DATA_SIZE0_DATAIN_HEIGHT(cna_desc->datain_height)
    );
  ops[5] = EMIT(REG_CNA_DATA_SIZE1, 
      CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(cna_desc->datain_channel - 1) | 
      CNA_DATA_SIZE1_DATAIN_CHANNEL(cna_desc->datain_channel)
    );
  ops[6] = EMIT(REG_CNA_DATA_SIZE2, 
      CNA_DATA_SIZE2_DATAOUT_WIDTH(cna_desc->dataout_width)
    );
  ops[7] = EMIT(REG_CNA_DATA_SIZE3, 
      CNA_DATA_SIZE3_DATAOUT_ATOMICS(cna_desc->dataout_atomics)
    );
  ops[8] = EMIT(REG_CNA_WEIGHT_SIZE0, 
      CNA_WEIGHT_SIZE0_WEIGHT_BYTES(cna_desc->weight_bytes)
    );
  ops[9] = EMIT(REG_CNA_WEIGHT_SIZE1, 
      CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(cna_desc->weight_bytes_per_kernel)
    );
  ops[10] = EMIT(REG_CNA_WEIGHT_SIZE2, 
      CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(cna_desc->weight_width) | 
      CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(cna_desc->weight_height) |
      CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(cna_desc->weight_kernels)
    );
  ops[11] = EMIT(REG_CNA_CBUF_CON0, 
      CNA_CBUF_CON0_WEIGHT_BANK(cna_desc->weight_bank) | 
      CNA_CBUF_CON0_DATA_BANK(cna_desc->data_bank)
    );
  ops[12] = EMIT(REG_CNA_CBUF_CON1, 
      CNA_CBUF_CON1_DATA_ENTRIES(cna_desc->data_entries)
    );
  ops[13] = EMIT(REG_CNA_CVT_CON0, 
      CNA_CVT_CON0_DATA_SIGN(cna_desc->data_sign) | 
      CNA_CVT_CON0_CVT_TYPE(cna_desc->cvt_type) | 
      CNA_CVT_CON0_CVT_BYPASS(cna_desc->cvt_bypass)
    );
  ops[14] = EMIT(REG_CNA_CVT_CON1, 
      CNA_CVT_CON1_CVT_SCALE0(cna_desc->cvt_scale0)
    );
  ops[15] = EMIT(REG_CNA_CVT_CON2, 
      CNA_CVT_CON2_CVT_SCALE1(cna_desc->cvt_scale1)
    );
  ops[16] = EMIT(REG_CNA_CVT_CON3, 
      CNA_CVT_CON3_CVT_SCALE2(cna_desc->cvt_scale2)
    );
  ops[17] = EMIT(REG_CNA_CVT_CON4, 
      CNA_CVT_CON4_CVT_SCALE3(cna_desc->cvt_scale3)
    );
  ops[18] = EMIT(REG_CNA_FC_CON0, 
      CNA_FC_CON0_FC_SKIP_EN(cna_desc->fc_skip_en)
    );
  ops[19] = EMIT(REG_CNA_FC_CON1, 
      CNA_FC_CON1_DATA_OFFSET(cna_desc->data_offset)
    ); 
  ops[20] = EMIT(REG_CNA_PAD_CON0, 
      CNA_PAD_CON0_PAD_LEFT(cna_desc->pad_left) | 
      CNA_PAD_CON0_PAD_TOP(cna_desc->pad_top)
    );

  printf("DEBUG: Writing ops[21] to ops[30]\n");
  ops[21] = EMIT(REG_CNA_FEATURE_DATA_ADDR, cna_desc->feature_base_addr);
  ops[22] = EMIT(REG_CNA_FC_CON2, 
      CNA_FC_CON2_WEIGHT_OFFSET(cna_desc->weight_offset)
    );
  ops[23] = EMIT(REG_CNA_DMA_CON0, 
      CNA_DMA_CON0_WEIGHT_BURST_LEN(cna_desc->weight_burst_len) | 
      CNA_DMA_CON0_DATA_BURST_LEN(cna_desc->data_burst_len)
    );
  ops[24] = EMIT(REG_CNA_DMA_CON1, 
      CNA_DMA_CON1_LINE_STRIDE(cna_desc->line_stride)
    );
  ops[25] = EMIT(REG_CNA_DMA_CON2, 
      CNA_DMA_CON2_SURF_STRIDE(cna_desc->surf_stride)
    );
  ops[26] = EMIT(REG_CNA_FC_DATA_SIZE0, 
      CNA_FC_DATA_SIZE0_DMA_WIDTH(cna_desc->dma_width) | 
      CNA_FC_DATA_SIZE0_DMA_HEIGHT(cna_desc->dma_height)
    );
  ops[27] = EMIT(REG_CNA_FC_DATA_SIZE1, 
      CNA_FC_DATA_SIZE1_DMA_CHANNEL(cna_desc->dma_channel)
    );
  ops[28] = EMIT(REG_CNA_DCOMP_CTRL, 0x0);
  ops[29] = EMIT(REG_CNA_DCOMP_REGNUM, 0x0);
  ops[30] = EMIT(REG_CNA_DCOMP_ADDR0, cna_desc->decompress_addr0);

  printf("DEBUG: Writing ops[31] to ops[50]\n");
  ops[31] = EMIT(REG_CNA_DCOMP_AMOUNT0, 0x0);
  ops[32] = EMIT(REG_CNA_DCOMP_AMOUNT1, 0x0);
  ops[33] = EMIT(REG_CNA_DCOMP_AMOUNT2, 0x0);
  ops[34] = EMIT(REG_CNA_DCOMP_AMOUNT3, 0x0);
  ops[35] = EMIT(REG_CNA_DCOMP_AMOUNT4, 0x0);
  ops[36] = EMIT(REG_CNA_DCOMP_AMOUNT5, 0x0);
  ops[37] = EMIT(REG_CNA_DCOMP_AMOUNT6, 0x0);
  ops[38] = EMIT(REG_CNA_DCOMP_AMOUNT7, 0x0);
  ops[39] = EMIT(REG_CNA_DCOMP_AMOUNT8, 0x0);
  ops[40] = EMIT(REG_CNA_DCOMP_AMOUNT9, 0x0);
  ops[41] = EMIT(REG_CNA_DCOMP_AMOUNT10, 0x0);
  ops[42] = EMIT(REG_CNA_DCOMP_AMOUNT11, 0x0);
  ops[43] = EMIT(REG_CNA_DCOMP_AMOUNT12, 0x0);
  ops[44] = EMIT(REG_CNA_DCOMP_AMOUNT13, 0x0);
  ops[45] = EMIT(REG_CNA_DCOMP_AMOUNT14, 0x0);
  ops[46] = EMIT(REG_CNA_DCOMP_AMOUNT15, 0x0);
  ops[47] = EMIT(REG_CNA_CVT_CON5, 0x0);
  ops[48] = EMIT(REG_CNA_PAD_CON1, 0x0);
  ops[49] = EMIT(REG_CORE_MISC_CFG, 
      CORE_MISC_CFG_PROC_PRECISION(core_desc->proc_precision) | 
      CORE_MISC_CFG_QD_EN(core_desc->qd_en)
    );
  ops[50] = EMIT(REG_CORE_DATAOUT_SIZE_0, 
      CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(core_desc->dataout_height) | 
      CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(core_desc->dataout_width)
    );
  ops[51] = EMIT(REG_CORE_DATAOUT_SIZE_1, 
      CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(core_desc->dataout_channel)
    );
  ops[52] = EMIT(REG_CORE_CLIP_TRUNCATE, 0x0);
  
  // 801 3030 0
  // ops[53] = NPUOP(OP_REG_CORE, 0x0, CORE_3030);
  ops[53] = NPUOP(OP_REG_CORE, 0x0, CORE_3030);

  printf("DEBUG: Writing ops[54] to ops[70]\n");
  ops[54] = EMIT(REG_DPU_FEATURE_MODE_CFG, 
      DPU_FEATURE_MODE_CFG_BURST_LEN(dpu_desc->burst_len) | 
      DPU_FEATURE_MODE_CFG_CONV_MODE(dpu_desc->conv_mode) |
      DPU_FEATURE_MODE_CFG_OUTPUT_MODE(dpu_desc->output_mode) | 
      DPU_FEATURE_MODE_CFG_FLYING_MODE(dpu_desc->flying_mode)
    );
  ops[55] = EMIT(REG_DPU_DATA_FORMAT, 
      DPU_DATA_FORMAT_OUT_PRECISION(dpu_desc->out_precision) |
      DPU_DATA_FORMAT_IN_PRECISION(dpu_desc->in_precision) |
      DPU_DATA_FORMAT_PROC_PRECISION(dpu_desc->proc_precision)
    );
  ops[56] = EMIT(REG_DPU_OFFSET_PEND, 0x0);
  ops[57] = EMIT(REG_DPU_DST_BASE_ADDR, dpu_desc->dst_base_addr);
  ops[58] = EMIT(REG_DPU_DST_SURF_STRIDE, 
      DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(dpu_desc->dst_surf_stride)
    );
  ops[59] = EMIT(REG_DPU_DATA_CUBE_WIDTH, 
      DPU_DATA_CUBE_WIDTH_WIDTH(dpu_desc->width)
    );
  ops[60] = EMIT(REG_DPU_DATA_CUBE_HEIGHT, 
      DPU_DATA_CUBE_HEIGHT_HEIGHT(dpu_desc->height)
    );
  ops[61] = EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0x0);
  ops[62] = EMIT(REG_DPU_DATA_CUBE_CHANNEL, 
      DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(dpu_desc->channel) | 
      DPU_DATA_CUBE_CHANNEL_CHANNEL(dpu_desc->channel)
    );

    
  ops[63] = EMIT(DPU_BS_CFG,
      DPU_BS_CFG_BS_RELU_BYPASS(dpu_desc->bs_relu_bypass) |
      DPU_BS_CFG_BS_MUL_BYPASS(dpu_desc->bs_mul_bypass) |
      DPU_BS_CFG_BS_ALU_BYPASS(dpu_desc->bs_alu_bypass) |
      DPU_BS_CFG_BS_BYPASS(dpu_desc->bs_bypass));
  ops[64] = EMIT(DPU_BS_ALU_CFG, 0x0);
  ops[65] = EMIT(DPU_BS_MUL_CFG, 0x0);
  ops[66] = EMIT(DPU_BS_RELUX_CMP_VALUE, 0x0);
  value = ((dpu_desc->size_e_2 & 0x7) << 8) | ((dpu_desc->size_e_1 & 0x7) << 5) |
    ((dpu_desc->size_e_0 & 0x7) << 2) | ((dpu_desc->od_bypass & 0x1) << 1);
  ops[67] = EMIT(DPU_BS_OW_CFG, value);
  ops[68] = EMIT(DPU_BS_OW_OP, 0x0);
  ops[69] = EMIT(DPU_WDMA_SIZE_0, dpu_desc->channel_wdma & 0x1FFF);
  value = ((dpu_desc->height_wdma & 0x1FFF) << 16) | (dpu_desc->width_wdma & 0x1FFF);
  ops[70] = EMIT(DPU_WDMA_SIZE_1, value);
  value = ((dpu_desc->bn_relu_bypass & 0x1) << 6) | ((dpu_desc->bn_mul_bypass &0x1) << 4) |
    ((dpu_desc->bn_alu_bypass & 0x1) << 1) | (dpu_desc->bn_bypass & 0x1);
  ops[71] = EMIT(DPU_BN_CFG, value);
  ops[72] = EMIT(DPU_BN_ALU_CFG, 0x0);
  ops[73] = EMIT(DPU_BN_MUL_CFG, 0x0);
  ops[74] = EMIT(DPU_BN_RELUX_CMP_VALUE, 0x0);
  value = ((dpu_desc->ew_relu_bypass & 0x1) << 9) | ((dpu_desc->ew_op_cvt_bypass & 0x1) << 8) |
    ((dpu_desc->ew_lut_bypass & 0x1) <<7) | ((dpu_desc->ew_op_bypass & 0x1) << 1) |
    (dpu_desc->ew_bypass & 0x1);
  ops[75] = EMIT(DPU_EW_CFG, value);
  ops[76] = EMIT(DPU_EW_CVT_OFFSET_VALUE, 0x0);
  ops[77] = EMIT(DPU_EW_CVT_SCALE_VALUE, 0x1);
  ops[78] = EMIT(DPU_EW_RELUX_CMP_VALUE, 0x0);
  ops[79] = EMIT(DPU_OUT_CVT_OFFSET, 0x0);
  value = ((dpu_desc->fp32tofp16_en & 0x1) << 16) | (dpu_desc->out_cvt_scale & 0xFFFF);
  ops[80] = EMIT(DPU_OUT_CVT_SCALE, value);
  ops[81] = EMIT(DPU_OUT_CVT_SHIFT, 0x0);
  ops[82] = EMIT(DPU_EW_OP_VALUE_0, 0x0);
  ops[83] = EMIT(DPU_EW_OP_VALUE_1, 0x0);
  ops[84] = EMIT(DPU_EW_OP_VALUE_2, 0x0);
  ops[85] = EMIT(DPU_EW_OP_VALUE_3, 0x0);
  ops[86] = EMIT(DPU_EW_OP_VALUE_4, 0x0);
  ops[87] = EMIT(DPU_EW_OP_VALUE_5, 0x0);
  ops[88] = EMIT(DPU_EW_OP_VALUE_6, 0x0);
  ops[89] = EMIT(DPU_EW_OP_VALUE_7, 0x0);
  value = ((dpu_desc->surf_add & 0xFFFFFFF) << 4);
  ops[90] = EMIT(DPU_SURFACE_ADD, value);
  ops[91] = EMIT(DPU_40C4, 0x0);
  ops[92] = EMIT(DPU_LUT_ACCESS_CFG, 0x0);
  ops[93] = EMIT(DPU_LUT_ACCESS_DATA, 0x0);
  ops[94] = EMIT(DPU_LUT_CFG, 0x0);
  ops[95] = EMIT(DPU_LUT_INFO, 0x0);
  ops[96] = EMIT(DPU_LUT_LE_START, 0x0);
  ops[97] = EMIT(DPU_LUT_LE_END, 0x0);
  ops[98] = EMIT(DPU_LUT_LO_START, 0x0);
  ops[99] = EMIT(DPU_LUT_LO_END, 0x0);
  ops[100] = EMIT(DPU_LUT_LE_SLOPE_SCALE, 0x0);
  ops[101] = EMIT(DPU_LUT_LE_SLOPE_SHIFT, 0x0);
  ops[102] = EMIT(DPU_LUT_LO_SLOPE_SCALE, 0x0);
  ops[103] = EMIT(DPU_LUT_LO_SLOPE_SHIFT, 0x0);
  ops[104] = EMIT(0x0, 0x0);  // Convert OP_NONE to EMIT with 0 values
  ops[105] = EMIT(PC_REGISTER_AMOUNTS, 0x0);
  ops[106] = EMIT(0x0, 0x0);  // Convert OP_40 to EMIT with 0 values
  ops[107] = NPUOP(OP_ENABLE, (PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE), PC_OPERATION_ENABLE);

  printf("DEBUG: gen_matmul_task completed successfully\n");
  printf("DEBUG: Total operations written: 108 (ops[0] to ops[107])\n");
  printf("DEBUG: Expected regcfg_amount: 104\n");
  printf("DEBUG: Mismatch: 108 - 104 = 4 operations extra\n");
}

/*
 * Simplified version of matrix mutliplication because :
 * a) we fail if cbuf storage is exceeded ie M,K,N get too large
 * b) because of (a) only generates a single task
 *
 * task memory needs to hold at laest 112 values
 * TODO: Fix a) & b) 
 *
 */
int gen_matmul_fp16(matmul_params_t *params) {

   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc;

   unsigned int fd_bytes;
   unsigned int fd_banks;
   unsigned int weight_banks;
   int surf_stride;

   // Add debug output
   printf("DEBUG: gen_matmul_fp16 called with params: m=%d, k=%d, n=%d\n", params->m, params->k, params->n);

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = precision_float16;
   cna_desc.proc_precision = precision_float16;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = params->m+1;
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height * 
     cna_desc.datain_channel * sizeof(__fp16);
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels; 

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(__fp16);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks +1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE)==0) ? weight_banks : weight_banks + 1;
   
   // Add debug output for CBUF calculations
   printf("DEBUG: CBUF calculations:\n");
   printf("  fd_bytes=%u, NPU_CBUF_BANK_SIZE=%u\n", fd_bytes, NPU_CBUF_BANK_SIZE);
   printf("  weight_bytes_per_kernel=%u\n", cna_desc.weight_bytes_per_kernel);
   printf("  weight_bytes=%u\n", cna_desc.weight_bytes);
   printf("  fd_banks calculation: %u / %u = %u, remainder %u\n", 
          fd_bytes, NPU_CBUF_BANK_SIZE, fd_bytes / NPU_CBUF_BANK_SIZE, fd_bytes % NPU_CBUF_BANK_SIZE);
   printf("  weight_banks calculation: %u / %u = %u, remainder %u\n", 
          cna_desc.weight_bytes, NPU_CBUF_BANK_SIZE, cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE, cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE);
   
   if ((fd_banks) > NPU_CBUF_BANKS-1) {
     printf("DEBUG: ERROR: fd_banks (%u) > NPU_CBUF_BANKS-1 (%u), returning -1\n", fd_banks, NPU_CBUF_BANKS-1);
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
        printf("DEBUG: weight_banks recalculated to %u\n", weight_banks);
        printf("DEBUG: Total banks used: %u + %u = %u (max: %u)\n", fd_banks, weight_banks, fd_banks + weight_banks, NPU_CBUF_BANKS);
       } else {
         printf("DEBUG: ERROR: weight_bytes_per_kernel (%u) > NPU_CBUF_BANK_SIZE (%u), returning -2\n", 
                cna_desc.weight_bytes_per_kernel, NPU_CBUF_BANK_SIZE);
         return -2;
       }
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 32;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % 32) == 0) ? 
     cna_desc.data_entries : cna_desc.data_entries +1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_desc.line_stride * ((cna_desc.datain_height / 4)-1);
   surf_stride = surf_stride < 0 ? surf_stride + 1 : surf_stride;
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = precision_float16;
   core_desc.qd_en = 1;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels -1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = (params->fp32tofp16==0) ? precision_float32 : precision_float16;
   dpu_desc.in_precision = precision_float16;
   dpu_desc.proc_precision = precision_float16;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width ;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass =1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass =1;
   dpu_desc.ew_op_bypass =1;
   dpu_desc.ew_lut_bypass =1;
   dpu_desc.ew_op_cvt_bypass =1;
   dpu_desc.ew_relu_bypass=1;
   dpu_desc.fp32tofp16_en = params->fp32tofp16 & 0x1;
   dpu_desc.out_cvt_scale =1;
   if (params->fp32tofp16 ==0) {
     dpu_desc.size_e_2 = 3;
     dpu_desc.size_e_1 = 3;
     dpu_desc.size_e_0 = 3;
   } else {
     dpu_desc.size_e_2 = 1;
     dpu_desc.size_e_1 = 1;
     dpu_desc.size_e_0 = 1;
   }
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;
   dpu_desc.surf_add = (!params->fp32tofp16) ? dpu_desc.dst_surf_stride * 4 : dpu_desc.dst_surf_stride * 2;

   gen_matmul_task(params->tasks,&cna_desc,&core_desc,&dpu_desc);

   return 0;
}

_Float16 matrixA[(MAX_M*MAX_K)];
_Float16 matrixB[(MAX_N*MAX_K)];
float expected_result[MAX_M*MAX_N];
uint64_t npu_regs[112];

int main(int argc, char **argv) {
    if (argc !=4) {
      printf("Invalid number of args %d, needs to supply M K N ie matmul_fp16 <M> <K> <N>\n",argc);
      return -1; 
    }

    int ret=0;
    unsigned int M = atoi(argv[1]); 
    unsigned int K = atoi(argv[2]); 
    unsigned int N = atoi(argv[3]);

    uint64_t regcmd_dma, regcmd_obj, tasks_dma, tasks_obj, input_dma, input_obj, weights_dma, weights_obj, output_dma, output_obj;
    uint32_t regcmd_handle, tasks_handle, input_handle, weights_handle, output_handle;
    uint32_t regcmd_flink, tasks_flink, input_flink, weights_flink, output_flink;

    // check matric dimensin
    if (M<=0 || M>MAX_M || ((M%4)!=0 && M!=1) || K<=0 || K>MAX_K || (K%32)!=0 || N<=0 || N>MAX_N || (N%16)!=0) {
        printf("Invalid matrix dimensions: M=%d (must be >0, <=%d, multiple of 4 or 1), K=%d (must be >0, <=%d, multiple of 32), N=%d (must be >0, <=%d, multiple of 16)\n", M, MAX_M, K, MAX_K, N, MAX_N);
        return -1;
    }

    // allocate mem
    int fd = npu_open();
    uint64_t *regcmd         = mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);
    struct rknpu_task *tasks = mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);
    void *input              = mem_allocate(fd, M*K*sizeof(__fp16), &input_dma, &input_obj, 0, &input_handle);
    void *weights            = mem_allocate(fd, N*K*sizeof(__fp16), &weights_dma, &weights_obj, 0, &weights_handle);
    void *output             = mem_allocate(fd, M*N*sizeof(float), &output_dma, &output_obj, 0, &output_handle);
    printf("input dma is %lx, output dma is %lx, weights dma is %lx\n", input_dma, output_dma, weights_dma);
    if ((regcmd == NULL) || (tasks == NULL) || (input == NULL) || (weights == NULL) || (output == NULL)) {
      printf("Failed to allocate memory \n");
      exit(1);
    }
    
    if (create_flink_name(fd, regcmd_handle, &regcmd_flink) < 0 ||
        create_flink_name(fd, tasks_handle, &tasks_flink) < 0 ||
        create_flink_name(fd, input_handle, &input_flink) < 0 ||
        create_flink_name(fd, weights_handle, &weights_flink) < 0 ||
        create_flink_name(fd, output_handle, &output_flink) < 0) {
        printf("Failed to create flink name for one or more GEM objects\n");
        goto cleanup;
    }
    printf("Created flink names: regcmd=%u, tasks=%u, input=%u, weights=%u, output=%u\n",
           regcmd_flink, tasks_flink, input_flink, weights_flink, output_flink);
  
    npu_reset(fd);
  
    matmul_params_t params;
    params.m = M;
    params.k = K;
    params.n = N;
    params.input_dma = input_dma;
    params.weights_dma = weights_dma;
    params.output_dma = output_dma;
    params.tasks = (uint64_t *) &npu_regs;
    params.fp32tofp16 = 0;
    ret = gen_matmul_fp16(&params);
    if (ret !=0) {
      printf("gen_matmul_fp16 failed %d\n",ret);
      goto cleanup;
    }
    
    memcpy(regcmd,npu_regs,sizeof(npu_regs));
  
    tasks[0].flags  = 0;
    tasks[0].op_idx = 0;
    tasks[0].enable_mask = 0xd;
    tasks[0].int_mask = 0x300; // wait for DPU to finish
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status =0;
    tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t)-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_dma;
  
    memset((void *)input,0,M*K*sizeof(_Float16));
    memset((void *)weights,0,K*N*sizeof(_Float16));
    memset((void *)output,0,M*N*sizeof(float));
  
    srand(time(NULL));
  
    // Need to use whole numbers for now as decimals return a slighty 
    // different result compared to ARM float calculations. Hence Rockchip
    // examples don't perform a exact comparison between expected and acutal
    // results from the matrix mutlipcation for fp16. Need to know why?
    for (int i = 0; i < M*K; i++) {
      matrixA[i] = (int)(10.0*rand_float()); 
    }
    
    for (int i = 0; i < N*K; i++) {
      matrixB[i] = (int)(10.0*rand_float());
    }
  
    _Float16 *weights_fp16 = weights;
     
    for(int n=1;n<=N;n++) {
      for(int k=1;k<=K;k++) {
        weights_fp16[weight_fp16(K,n,k)]= matrixB[((n-1)*K)+(k-1)];
      }
    }
   
    _Float16 *feature_data_fp16 = (_Float16*) input;
  
    for (int m=1;m<=M;m++) {
      for (int k=1;k<=K;k++) {
        feature_data_fp16[feature_data(K,M,1,8,k,m,1)]= matrixA[((m-1)*K)+(k-1)];
      }
    }
  
    matmul_fp32(M,K,N,(_Float16 *)&matrixA, (_Float16 *)&matrixB, (float *)&expected_result);
  
    struct rknpu_submit submit = {
      .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
      .timeout = 6000,
      .task_start = 0,
      .task_number = 1,
      .task_counter = 0,
      .priority = 0,
      .task_obj_addr = tasks_obj,
      .regcfg_obj_addr = 0,
      .task_base_addr = 0,
      .user_data = 0,
      .core_mask = 1,
      .fence_fd = -1,
      .subcore_task = { // Only use core 1, nothing for core 2/3
        {
          .task_start = 0,
          .task_number = 1,
        }, { 1, 0}, {2, 0}, {0,0}, {0,0}
      },
    };
    ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
    printf("RKNPU_SUBMIT returned %d\n", ret);
    if (ret <0) {
      return ret;
    }
  
    printf("=========================================================================================================\n");
    float *output_data = (float*) output;
    for (int m=1;m<=M;m++) {
      for (int n=1;n<N;n++) {
        float actual = output_data[feature_data(N, M, 1, 4, n, m, 1)];
        float expected = expected_result[((m-1)*N)+(n-1)];
        int32_t *e, *a;
        e = (int32_t *)&expected;
        a = (int32_t *)&actual;
        if (actual != expected) {
          printf("\nmismatch m:%d n:%d  expected:%6.5f acutal:%6.5f %x %x\n",m,n,expected,actual,*e,*a);
          ret = -1;
        }
      }
    }
    if (ret == 0) {
     printf("Multiplication of [%d,%d] x [%d,%d] succesful \n",M,K,N,K);	  
    }
    printf("=========================================================================================================\n");
  
    // Demonstrate how to use flink names to access the GEM objects
    printf("\n=== Demonstrating GEM FLINK functionality ===\n");
    printf("These flink names can be used by other processes to access the same memory:\n");
    printf("regcmd: %u, tasks: %u, input: %u, weights: %u, output: %u\n",
           regcmd_flink, tasks_flink, input_flink, weights_flink, output_flink);
    
    // Example: Open the output GEM object using its flink name
    uint32_t reopened_handle;
    uint64_t reopened_size;
    if (open_gem_by_flink(fd, output_flink, &reopened_handle, &reopened_size) == 0) {
      printf("Successfully reopened output GEM object via flink name %u\n", output_flink);
      printf("Reopened handle: %u, size: %lu\n", reopened_handle, reopened_size);
      
      // Map the reopened GEM object
      struct rknpu_mem_map mem_map = { .handle = reopened_handle, .offset = 0 };
      int map_ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
      if (map_ret >= 0) {
        void *reopened_map = mmap(NULL, reopened_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);
        if (reopened_map != MAP_FAILED) {
          printf("Successfully mapped reopened GEM object at %p\n", reopened_map);
          
          // Verify the data is accessible
          float *reopened_output = (float*)reopened_map;
          printf("First few values from reopened output: %f, %f, %f\n", 
                 reopened_output[0], reopened_output[1], reopened_output[2]);
          
          munmap(reopened_map, reopened_size);
        }
      }
    }
  
  cleanup:
    munmap(regcmd,1024);
    munmap(tasks,1024);
    munmap(input,M*K*sizeof(_Float16));
    munmap(weights,N*K*sizeof(_Float16));
    munmap(output,M*N*sizeof(float));
  
    mem_destroy(fd, regcmd_handle, regcmd_obj);
    mem_destroy(fd, tasks_handle, tasks_obj );
    mem_destroy(fd, input_handle, input_obj);
    mem_destroy(fd, weights_handle, weights_obj);
    mem_destroy(fd, output_handle, output_obj);
  
    npu_close(fd);
    return ret;
  }
