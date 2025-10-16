/*
* Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
*
* This file is part of rk3588-npu.
*
* rk3588-npu is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* rk3588-npu is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef RKNNOPS_H
#define RKNNOPS_H


#include <sys/ioctl.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <libdrm/drm.h>
#include "rknpu-ioctl.h"
#include "rknn_api.h"
#include "rkt_registers.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
   uint64_t *data;         // Pointer to the array memory
   size_t size;       // Current number of elements
   size_t capacity;   // Allocated capacity of the array
} DynamicArray;

DynamicArray regs;


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

// Initialize the dynamic array
void initArray(DynamicArray *arr, size_t initialCapacity) {
   arr->data = (uint64_t *)malloc(initialCapacity * sizeof(uint64_t));
   arr->size = 0;
   arr->capacity = initialCapacity;
}

// Push a new element to the dynamic array
void push(DynamicArray *arr, uint64_t value) {
   if (arr->size == arr->capacity) {
      // Increase capacity (e.g., double it)
      arr->capacity *= 2;
      arr->data = (uint64_t *)realloc(arr->data, arr->capacity * sizeof(uint64_t));
      if (arr->data == NULL) {
         fprintf(stderr, "Memory allocation failed\n");
         exit(1);
      }
   }
   arr->data[arr->size] = value;
   arr->size++;
}

void freeArray(DynamicArray *arr) {
   free(arr->data);
   arr->data = NULL;
   arr->size = 0;
   arr->capacity = 0;
}

static uint32_t current_alu_algorithm = 2; // Default to Add (2)
void set_alu_algorithm(uint32_t algo) {
   current_alu_algorithm = algo;
}

static void
emit_raw(DynamicArray *arr, uint32_t target, uint32_t reg,
         uint64_t value)
{
   uint64_t packed_value = 0;
   packed_value = ((uint64_t)target) << 48;
   packed_value |= ((uint64_t)value) << 16;
   packed_value |= (uint64_t)reg;

   push(arr, packed_value);
}

static void
emit(uint32_t reg, uint64_t value)
{
   uint32_t target = rkt_get_target(reg) + 0x1;
   emit_raw(&regs, target, reg, value);
}

#define EMIT(offset, value) emit(offset, value);

// static inline uint64_t EMIT(uint32_t reg, uint32_t value){
//    uint32_t target = rkt_get_target(reg) + 0x1;
 
//    uint64_t packed_value = 0;
//    packed_value = ((uint64_t)target) << 48;
//    packed_value |= ((uint64_t)value) << 16;
//    packed_value |= (uint64_t)reg;
 
//    return packed_value;
// }

struct MemHandles {
   void* input;
   void* weights;
   void* output;
   uint64_t input_dma, input_obj;
   uint64_t weights_dma, weights_obj;
   uint64_t output_dma, output_obj;
   uint64_t tasks_obj;
};

int get_type_size(rknn_tensor_type type){
   switch (type){
      case RKNN_TENSOR_INT8:
            return sizeof(int8_t);
      case RKNN_TENSOR_UINT8:
            return sizeof(uint8_t);
      case RKNN_TENSOR_INT16:
            return sizeof(int16_t);
      case RKNN_TENSOR_UINT16:
            return sizeof(uint16_t);
      case RKNN_TENSOR_INT32:
            return sizeof(int32_t);
      case RKNN_TENSOR_UINT32:
            return sizeof(uint32_t);
      case RKNN_TENSOR_INT64:
            return sizeof(int64_t);
      case RKNN_TENSOR_FLOAT16:
            return sizeof(__fp16);
      case RKNN_TENSOR_FLOAT32:
            return sizeof(float);
      default:
            printf("    get_type_size error: not support dtype %d\n", type);
            return 0;
   }
}

void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle) {
   int ret;
   struct rknpu_mem_create mem_create = {
      .flags = flags | RKNPU_MEM_NON_CACHEABLE,
      .size = size,
   };

   ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
   if(ret < 0)  {
      printf("RKNPU_MEM_CREATE failed %d\n",ret);
      return NULL;
   }

   struct rknpu_mem_map mem_map = { .handle = mem_create.handle, .offset=0 };
   ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
   if(ret < 0) {
      printf("RKNPU_MEM_MAP failed %d\n",ret);
      return NULL;
   }	
   void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);

   *dma_addr = mem_create.dma_addr;
   *obj = mem_create.obj_addr;
   if (handle) *handle = mem_create.handle;  // Return the GEM handle if requested
   return map;
}

void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr) {
   int ret;
   struct rknpu_mem_destroy destroy = {
      .handle = handle ,
      .obj_addr = obj_addr
   };

   ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);
   if (ret <0) {
      printf("RKNPU_MEM_DESTROY failed %d\n",ret);
   }
}

int getDeviceFd()
{
   int fd = open("/dev/dri/card1", O_RDWR);
   if(fd<0) {
      printf("Failed to open /dev/dri/card1");
      exit(1);
   }
   return fd;  
}

int npu_reset(int fd) {
   struct rknpu_action act = {
     .flags = RKNPU_ACT_RESET,
   };
   return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);	
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

void regcmd_helper(input_dma, weights_dma, output_dma){
   struct {
      int dst_base_addr_offset;         // REG_DPU_DST_BASE_ADDR
      int data_cube_width;              // REG_DPU_DATA_CUBE_WIDTH
      int wdma_size_1;                  // REG_DPU_WDMA_SIZE_1
      int rdma_data_cube_width;         // REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH
      int rdma_src_base_addr_offset;    // REG_DPU_RDMA_RDMA_SRC_BASE_ADDR
      int rdma_ew_base_addr_offset;     // REG_DPU_RDMA_RDMA_EW_BASE_ADDR
      int rdma_surf_notch;              // REG_DPU_RDMA_RDMA_SURF_NOTCH
      int rdma_ew_surf_notch;           // REG_DPU_RDMA_RDMA_EW_SURF_NOTCH
     } params[3] = {
      {0x0, 0, 0, 10, 0x0, 0x0, 0, 0},
      // width = (rdma_data_cube_width + 1) * 8
  };
   for (int i = 0; i < 1; ++i) {
      // matmul
      if (current_alu_algorithm == 11) {
         EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
         EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
         EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(33));
         EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
         EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(1) | CNA_DATA_SIZE0_DATAIN_HEIGHT(32));
         EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(31) | CNA_DATA_SIZE1_DATAIN_CHANNEL(32));
         EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(1));
         EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(32));
         EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000800);
         EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(64));
         EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(1) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(1) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(32));
         EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
         EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(1));
         EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) | CNA_CVT_CON0_CVT_BYPASS(1));
         EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
         EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
         EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
         EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
         EMIT(REG_CNA_FC_CON0, 0x00000000);
         EMIT(REG_CNA_FC_CON1, 0x00000000);
         EMIT(REG_CNA_PAD_CON0, 0x00000000);
         EMIT(REG_CNA_FEATURE_DATA_ADDR, CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR(input_dma));
         EMIT(REG_CNA_FC_CON2, 0x00000000);
         EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
         EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(4));
         EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(28));
         EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(1) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(32));
         EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(32));
         EMIT(REG_CNA_DCOMP_CTRL, 0x00000000);
         EMIT(REG_CNA_DCOMP_REGNUM, 0x00000000);
         EMIT(REG_CNA_DCOMP_ADDR0, CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0(weights_dma));
         EMIT(REG_CNA_DCOMP_AMOUNT0, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT1, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT2, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT3, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT4, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT5, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT6, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT7, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT8, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT9, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT10, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT11, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT12, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT13, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT14, 0x00000000);
         EMIT(REG_CNA_DCOMP_AMOUNT15, 0x00000000);
         EMIT(REG_CNA_CVT_CON5, 0x00000000);
         EMIT(REG_CNA_PAD_CON1, 0x00000000);
         EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2) | CORE_MISC_CFG_QD_EN(1));
         EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(31));
         EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(31));
         EMIT(REG_CORE_CLIP_TRUNCATE, 0x00000000);
         // [ffef01a8] lsb 0801000000003030 - CORE Unknown
         emit_raw(&regs, CORE | 0x1, 0x3030, 0);

         EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
         EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(5) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
         EMIT(REG_DPU_OFFSET_PEND, 0x00000000);
         EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma));
         EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(32));
         EMIT(REG_DPU_DATA_CUBE_WIDTH, 0x00000000);
         EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(31));
         EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0x00000000);
         EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(31) | DPU_DATA_CUBE_CHANNEL_CHANNEL(31));
         EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
         EMIT(REG_DPU_BS_ALU_CFG, 0x00000000);
         EMIT(REG_DPU_BS_MUL_CFG, 0x00000000);
         EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0x00000000);
         EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) | DPU_BS_OW_CFG_SIZE_E_1(3) | DPU_BS_OW_CFG_SIZE_E_0(3) | DPU_BS_OW_CFG_OD_BYPASS(1));
         EMIT(REG_DPU_BS_OW_OP, 0x00000000);
         EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(31));
         EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(31));
         EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
         EMIT(REG_DPU_BN_ALU_CFG, 0x00000000);
         EMIT(REG_DPU_BN_MUL_CFG, 0x00000000);
         EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0x00000000);
         EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
         EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0x00000000);
         EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
         EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0x00000000);
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x00000000);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
         EMIT(REG_DPU_OUT_CVT_SHIFT, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_0, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_1, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_2, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_3, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_4, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_5, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_6, 0x00000000);
         EMIT(REG_DPU_EW_OP_VALUE_7, 0x00000000);
         EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(128));
         // [ffef02d8] lsb 00010000000040c4 - noone Unknown
         emit_raw(&regs, 0x0 | 0x1, 0x40c4, 0);


         // EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1)); 
         emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));

      }
      // RELU
      else if (current_alu_algorithm == 10) {
         EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
         EMIT(REG_DPU_RDMA_RDMA_S_POINTER, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
         EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2) | DPU_FEATURE_MODE_CFG_FLYING_MODE(1));
         EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
         EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma));
         EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(4));
         EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(1));
         EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(7) | DPU_DATA_CUBE_CHANNEL_CHANNEL(7));
         EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1));
         EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_OD_BYPASS(1));
         EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(7));
         EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_WIDTH_WDMA(1));
         EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
         EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
         EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
         EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(4));
         EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(1));
         EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(7));
         EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(weights_dma));
         EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
         EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1));
         EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(2));
         EMIT(REG_DPU_RDMA_RDMA_WEIGHT, DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));
         // EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(12));

         emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(12) | PC_OPERATION_ENABLE_OP_EN(0));
         EMIT(REG_PC_VERSION, 0x00020000);
      }
      else {
         EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
         EMIT(REG_DPU_RDMA_RDMA_S_POINTER, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
         EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2) | DPU_FEATURE_MODE_CFG_FLYING_MODE(1));
         EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(5) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
         EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma)+ params[i].dst_base_addr_offset);
         EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
         EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(params[i].data_cube_width));
         EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(7) | DPU_DATA_CUBE_CHANNEL_CHANNEL(7));
         EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
         EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_OD_BYPASS(1));
         EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(7));
         EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_WIDTH_WDMA(params[i].wdma_size_1));
         EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
         EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_ALU_ALGO(current_alu_algorithm) | DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1));
         EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
         
         EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(1));
         EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(params[i].rdma_data_cube_width));
         EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(7));
         EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(input_dma)+ params[i].rdma_src_base_addr_offset);
         EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(2));
         EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR(weights_dma)+ params[i].rdma_ew_base_addr_offset);
         
         // 1x1 is 1, 1x2 is 4
         EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(1));
   
         EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1));
         EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(params[i].rdma_surf_notch));
         EMIT(REG_DPU_RDMA_RDMA_WEIGHT, DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));
         EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(params[i].rdma_ew_surf_notch));
         // EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(12));
         
         // MIN MAX
         if (current_alu_algorithm == 0 || current_alu_algorithm == 1){
            // EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(params[i].data_cube_width));
            // EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_MINMAX_CTL(1) | DPU_DATA_CUBE_HEIGHT_HEIGHT(0));
            EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(0));
   
            EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_EQUAL_EN(1) | DPU_EW_CFG_EW_ALU_ALGO(current_alu_algorithm) | DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1));
            // EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_BINARY_EN(1) | DPU_EW_CFG_EW_ALU_ALGO(current_alu_algorithm) | DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1));
         }
   
         // MUL
         if (current_alu_algorithm == 9){
            // EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(0) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1) | DPU_EW_CFG_EW_OP_TYPE(1))
            EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1) | DPU_EW_CFG_EW_OP_TYPE(1));
         }
         // DIV
         else if(current_alu_algorithm == 3 ){
            // swap input and weights for DIV
            EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(weights_dma)+ params[i].rdma_src_base_addr_offset);
            EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR(input_dma)+ params[i].rdma_ew_base_addr_offset);
   
            EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_ALU_ALGO(current_alu_algorithm) | DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1));
            EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(0));
            EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(0) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1));      
   
            // testing truncate
            // EMIT(REG_DPU_OUT_CVT_OFFSET, 0x0000000);
            // EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(0) );
            // EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_CVT_ROUND(0));
         }
         emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(12) | PC_OPERATION_ENABLE_OP_EN(0));
         EMIT(REG_PC_VERSION, 0x00020000);
      }
   
   }
}

float rand_float() {
   return rand()/(float)RAND_MAX;
}

struct MemHandles createRegCmd(int fd, size_t input_size, size_t weights_size, size_t output_size, uint32_t alu_algorithm)
{
   set_alu_algorithm(alu_algorithm);

   uint64_t regcmd_dma, regcmd_obj;
   uint32_t regcmd_handle;
   
   uint64_t tasks_dma, tasks_obj;
   uint32_t tasks_handle;
   uint64_t input_dma, input_obj;
   uint32_t input_handle;
   uint64_t weights_dma, weights_obj;
   uint32_t weights_handle;
   uint64_t output_dma, output_obj;
   uint32_t output_handle;
   
   // uint64_t *regcmd = (uint64_t *)(mem_allocate(fd, 4096, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle));
   // struct rknpu_task *tasks = (rknpu_task *)mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);
   uint64_t *regcmd         = mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);
   struct rknpu_task *tasks = mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);
   // void *input = mem_allocate(fd, input_size, &input_dma, &input_obj, 0, &input_handle);  
   // void *weights = mem_allocate(fd, weights_size, &weights_dma, &weights_obj, 0, &weights_handle);
   // void *output = mem_allocate(fd, output_size, &output_dma, &output_obj, 0, &output_handle);
   printf("%d %d %d\n", input_size, weights_size, output_size);
   void *input              = mem_allocate(fd, input_size, &input_dma, &input_obj, 0, &input_handle);
   void *weights            = mem_allocate(fd, weights_size, &weights_dma, &weights_obj, 0, &weights_handle);
   void *output             = mem_allocate(fd, output_size, &output_dma, &output_obj, 0, &output_handle);


   uint32_t regcmd_flink, tasks_flink, input_flink, weights_flink, output_flink;

   if (create_flink_name(fd, regcmd_handle, &regcmd_flink) < 0 ||
      create_flink_name(fd, tasks_handle, &tasks_flink) < 0 ||
      create_flink_name(fd, input_handle, &input_flink) < 0 ||
      create_flink_name(fd, weights_handle, &weights_flink) < 0 ||
      create_flink_name(fd, output_handle, &output_flink) < 0) {
      printf("Failed to create flink name for one or more GEM objects\n");
   }
   printf("Created flink names: regcmd=%u, tasks=%u, input=%u, weights=%u, output=%u\n",
      regcmd_flink, tasks_flink, input_flink, weights_flink, output_flink);
   npu_reset(fd);
   
   regcmd_helper(input_dma, weights_dma, output_dma);

   printf("regs.size %d\n", regs.size);
   // uint64_t npu_regs_a[regs.size];
   uint64_t npu_regs_a[112];
   memset(npu_regs_a, 0, sizeof(npu_regs_a));
   
   memcpy(npu_regs_a, regs.data, regs.size * sizeof(uint64_t));
   memcpy(regcmd, npu_regs_a, sizeof(npu_regs_a));

   tasks[0].flags  = 0;
   tasks[0].op_idx = 0;
   tasks[0].enable_mask = 0xd;
   tasks[0].int_mask = 0x300;
   tasks[0].int_clear = 0x1ffff;
   tasks[0].int_status = 0;
   tasks[0].regcfg_amount = sizeof(npu_regs_a)/sizeof(uint64_t);
   // hardcoded for matmul test
   // tasks[0].regcfg_amount = 104;
   printf("tasks[0].regcfg_amount %d", tasks[0].regcfg_amount) ;
   tasks[0].regcfg_offset = 0;
   tasks[0].regcmd_addr = regcmd_dma;

   struct MemHandles handles;
   handles.input = input;
   handles.weights = weights;
   handles.output = output;
   handles.input_dma = input_dma;
   handles.input_obj = input_obj;
   handles.weights_dma = weights_dma;
   handles.weights_obj = weights_obj;
   handles.output_dma = output_dma;
   handles.output_obj = output_obj;
   handles.tasks_obj = tasks_obj;
   return handles;
}

int submitTask(int fd, uint64_t tasks_obj){
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
      }, { 1, 0}, {2, 0}
      },
   };
   return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
}

__fp16* float16_matmul(__fp16* a, __fp16* b, uint32_t alu_algorithm, int M, int N, int K)
{
   int fd = getDeviceFd();
   npu_reset(fd);
   rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;

   size_t input_size   = (size_t)M * K * sizeof(__fp16);
   size_t weights_size = (size_t)N * K * sizeof(__fp16);
   size_t output_size  = (size_t)M * N * sizeof(float);

   struct MemHandles handles = createRegCmd(fd, input_size, weights_size, output_size, alu_algorithm);
   __fp16 *weights_fp16 = (__fp16*)(handles.weights);
   __fp16 *feature_data_fp16 = (__fp16*)(handles.input);
   __fp16 *output_data = (__fp16*)(handles.output);
   memset((void *)weights_fp16,      0, weights_size);
   memset((void *)feature_data_fp16, 0, input_size);
   memset((void *)output_data,       0, output_size);

   for(int n=1; n<=N; n++) {
       for(int k=1; k<=K; k++) {
           weights_fp16[weight_fp16(K,n,k)] = b[((n-1)*K)+(k-1)];
       }
   }
   for (int m=1; m<=M; m++) {
       for (int k=1; k<=K; k++) {
           feature_data_fp16[feature_data(K,M,1,8,k,m,1)] = a[((m-1)*K)+(k-1)];
       }
   }

   int ret = submitTask(fd, handles.tasks_obj);
   if(ret < 0) {
      printf("RKNPU_SUBMIT failed %d\n",ret);
      return NULL;
   }
   return output_data;
}

__fp16* float16_alu_op(__fp16* a, __fp16* b, uint32_t alu_algorithm, int size)
{
   int fd = getDeviceFd();
   npu_reset(fd);
   rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;

   size_t bytes = (size_t)size * get_type_size(dtype);
   struct MemHandles handles = createRegCmd(fd, bytes, bytes, bytes, alu_algorithm);
   __fp16 *weights_fp16 = (__fp16*)(handles.weights);
   __fp16 *feature_data_fp16 = (__fp16*)(handles.input);
   __fp16 *output_data = (__fp16*)(handles.output);
   // float* output_data_float = (float*)(handles.output);

   memcpy(weights_fp16,      a, bytes);
   memcpy(feature_data_fp16, b, bytes);

   int ret = submitTask(fd, handles.tasks_obj);
   if(ret < 0) {
      printf("RKNPU_SUBMIT failed %d\n",ret);
      return NULL;
   }

   // __fp16 *output_data_fp16 = (__fp16*)(handles.output);
   // printf("\nMethod 1 - Correct fp16 casting: fp16=%f fp32=%f\n", 
         //  output_data_fp16[0], (float)output_data_fp16[0]);

   float* output_data_float = (float*)(handles.output);
   printf("\nMethod 2 - Wrong float casting: fp16=%f fp32=%f\n", 
          (__fp16)output_data_float[0], output_data_float[0]);

   return output_data;
}

int16_t* int16_alu_op(int16_t* a, int16_t* b, uint32_t alu_algorithm)
{
   int fd = getDeviceFd();
   npu_reset(fd);
   rknn_tensor_type dtype = RKNN_TENSOR_INT16;

   size_t bytes = get_type_size(dtype);
   struct MemHandles handles = createRegCmd(fd, bytes, bytes, bytes, alu_algorithm);
   int16_t *weights_int16 = (int16_t*)(handles.weights);
   int16_t *feature_data_int16 = (int16_t*)(handles.input);
   int16_t *output_data = (int16_t*)(handles.output);

   memcpy(weights_int16, a, bytes);
   memcpy(feature_data_int16, b, bytes);

   int ret = submitTask(fd, handles.tasks_obj);
   if(ret < 0) {
         printf("RKNPU_SUBMIT failed %d\n",ret);
         return NULL;
   }
   return output_data;
}

int8_t* int8_alu_op(int8_t* a, int8_t* b, uint32_t alu_algorithm)
{
   int fd = getDeviceFd();
   npu_reset(fd);

   rknn_tensor_type dtype = RKNN_TENSOR_INT8;

   size_t bytes = get_type_size(dtype);
   struct MemHandles handles = createRegCmd(fd, bytes, bytes, bytes, alu_algorithm);
   int8_t *weights_int8 = (int8_t*)(handles.weights);
   int8_t *feature_data_int8 = (int8_t*)(handles.input);
   int8_t *output_data = (int8_t*)(handles.output);

   memcpy(weights_int8, a, bytes);
   memcpy(feature_data_int8, b, bytes);

   int ret = submitTask(fd, handles.tasks_obj);
   if(ret < 0) {
         printf("RKNPU_SUBMIT failed %d\n",ret);
         return NULL;
   }
   return output_data;
}

#ifdef __cplusplus
}
#endif

#endif /* RKNNOPS_H */