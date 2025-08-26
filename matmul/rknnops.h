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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
   uint64_t *data;         // Pointer to the array memory
   size_t size;       // Current number of elements
   size_t capacity;   // Allocated capacity of the array
} DynamicArray;

DynamicArray regs;


/**
 * @brief Create a flink name for a GEM handle
 * @param fd DRM device file descriptor
 * @param handle GEM handle to create flink name for
 * @param flink_name Pointer to store the resulting flink name
 * @return 0 on success, negative error code on failure
 */
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

/**
 * @brief Open a GEM object by flink name
 * @param fd DRM device file descriptor
 * @param flink_name Flink name of the GEM object
 * @param handle Pointer to store the resulting GEM handle
 * @param size Pointer to store the GEM object size
 * @return 0 on success, negative error code on failure
 */
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

// Free the allocated memory of the array
void freeArray(DynamicArray *arr) {
   free(arr->data);
   arr->data = NULL;
   arr->size = 0;
   arr->capacity = 0;
}

// Global variable to store the current ALU algorithm
static uint32_t current_alu_algorithm = 2; // Default to Add (2)

// Function to set the ALU algorithm
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


/* ============================================================================
 * RK3588 NPU Operations Library
 * ============================================================================
 * This library provides high-level operations for the RockChip RK3588 NPU
 * including memory management, data operations, and NPU task execution.
 */

/* ============================================================================
 * Basic Data Types and Operations
 * ============================================================================ */



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



struct MemHandles createRegCmd(int fd, int type_size, uint32_t alu_algorithm)
{

    // Set the ALU algorithm for this operation
    set_alu_algorithm(alu_algorithm);
    uint64_t tasks_dma, tasks_obj;
    uint32_t tasks_handle;
    struct rknpu_task *tasks = (struct rknpu_task *)mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);

    uint64_t regcmd_dma, regcmd_obj;
    uint32_t regcmd_handle;
    uint64_t *regcmd = (uint64_t *)(mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle));
    
    uint64_t input_dma, input_obj;
    uint32_t input_handle;
    void *input = mem_allocate(fd, type_size, &input_dma, &input_obj, 0, &input_handle);  

    uint64_t weights_dma, weights_obj;
    uint32_t weights_handle;
    void *weights = mem_allocate(fd, type_size, &weights_dma, &weights_obj, 0, &weights_handle);

    uint64_t output_dma, output_obj;
    uint32_t output_handle;
    void *output = mem_allocate(fd, type_size, &output_dma, &output_obj, 0, &output_handle);

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


    // To return input, weights, and output, you can use output parameters or a struct.
    // Example: define a struct to hold them and return it.
    // Set input, weights and output physical memory locations. Note limited to 
    // a 32 bit address size (4GB)
   
    EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
    DPU_S_POINTER_EXECUTER_PP_EN(1) |
    DPU_S_POINTER_POINTER_PP_EN(1));

    EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) |
    DPU_FEATURE_MODE_CFG_CONV_MODE(0) |
    DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2) |
    DPU_FEATURE_MODE_CFG_FLYING_MODE(1));

    EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) |
    DPU_DATA_FORMAT_IN_PRECISION(2) |
    DPU_DATA_FORMAT_PROC_PRECISION(2));

    EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_CVT_TYPE(0) |
    DPU_EW_CFG_EW_DATA_MODE(1) |
    DPU_EW_CFG_EDATA_SIZE(2) |
    DPU_EW_CFG_EW_ALU_ALGO(current_alu_algorithm) |
    DPU_EW_CFG_EW_RELU_BYPASS(0) |
    DPU_EW_CFG_EW_LUT_BYPASS(0) |
    DPU_EW_CFG_EW_OP_SRC(0));

    EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(7) |
    DPU_DATA_CUBE_CHANNEL_CHANNEL(7));

    EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_OD_BYPASS(1));
    EMIT(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0));

     //0x1001108202c04070
     EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(0) | DPU_BS_CFG_BS_ALU_SRC(0) |
     DPU_BS_CFG_BS_RELUX_EN(0) |
     DPU_BS_CFG_BS_RELU_BYPASS(1) |
     DPU_BS_CFG_BS_MUL_PRELU(0) |
     DPU_BS_CFG_BS_MUL_BYPASS(1) |
     DPU_BS_CFG_BS_ALU_BYPASS(1) | 
     DPU_BS_CFG_BS_BYPASS(1));

    
    EMIT(REG_DPU_WDMA_SIZE_0,
    DPU_WDMA_SIZE_0_CHANNEL_WDMA(7));

    EMIT(REG_DPU_WDMA_SIZE_1,
    DPU_WDMA_SIZE_1_HEIGHT_WDMA(0) | DPU_WDMA_SIZE_1_WIDTH_WDMA(9));

    EMIT(REG_DPU_BN_CFG,
    DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
    DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));

    EMIT(REG_DPU_BN_ALU_CFG,0);    
    EMIT(REG_DPU_BN_MUL_CFG,0);    
    EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0)

    //0x1001108202c04070
    EMIT(REG_DPU_EW_CFG,
        DPU_EW_CFG_EW_CVT_TYPE(0) | DPU_EW_CFG_EW_DATA_MODE(1) |
           DPU_EW_CFG_EDATA_SIZE(2) | DPU_EW_CFG_EW_ALU_ALGO(current_alu_algorithm) |
           DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |
           DPU_EW_CFG_EW_OP_SRC(1));


    float addition_scale = 0.0;
    float add_scale = 0.0;
    if (fabs(addition_scale - 0.090192) < 0.00001) {
       add_scale = 299.671889248;
    } else if (fabs(addition_scale - 0.399250) < 0.00001) {
       add_scale = 1326.499209406;
    } else if (fabs(addition_scale - 0.364902) < 0.00001) {
       add_scale = 780.34375;
    } else if (fabs(addition_scale - 0.422037) < 0.00001) {
       add_scale = 715.5625;
    } else if (fabs(addition_scale - 0.213016) < 0.00001) {
       add_scale = 564.6875;
    } else if (fabs(addition_scale - 0.244231) < 0.00001) {
       add_scale = 499.796875;
    } else if (fabs(addition_scale - 0.283416) < 0.00001) {
       add_scale = 488.203125;
    } else if (fabs(addition_scale - 0.171151) < 0.00001) {
       add_scale = 602.90625;
    } else if (fabs(addition_scale - 0.164588) < 0.00001) {
       add_scale = 271.921875;
    } else if (fabs(addition_scale - 0.204098) < 0.00001) {
       add_scale = 262.90625;
    } else if (fabs(addition_scale - 0.116532) < 0.00001) {
       add_scale = 450.140625;
    } else if (fabs(addition_scale - 0.134499) < 0.00001) {
       add_scale = 212.1953125;
    } else if (fabs(addition_scale - 0.220141) < 0.00001) {
       add_scale = 368.28125;
    } else if (fabs(addition_scale - 0.094560) < 0.00001) {
       add_scale = 416.421875;
    } else if (fabs(addition_scale - 0.093230) < 0.00001) {
       add_scale = 305.421875;
    } else if (fabs(addition_scale - 0.100618) < 0.00001) {
       add_scale = 313.671875;
    } else {
       add_scale = 0.0;
    }
    
    uint32_t add_scale_bits = (uint32_t)(round(add_scale));
    
    printf("add_scale_bits: %d\n", add_scale_bits);

    unsigned add_shift = 127 + 31 - 32 - (add_scale_bits >> 23) + 16;

    unsigned scale = ((add_scale_bits >> 9) & 0x7fff);
    if (scale < 1 << 14)
       scale |= 1 << 14;

    //0x1001000000014078
    EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
    EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
    EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
    EMIT(REG_DPU_OUT_CVT_OFFSET, 0);


    // 0x02000000
    // 512 << 16 = 0x02000000
    // x << 16 = 0x00010001
      
    emit_raw(&regs, DPU | 0x1, REG_DPU_OUT_CVT_SCALE, 65537);

    EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(1-1));
    EMIT(REG_DPU_EW_OP_VALUE_0, 0);
    EMIT(REG_DPU_EW_OP_VALUE_1, 0);
    EMIT(REG_DPU_EW_OP_VALUE_2, 0);
    EMIT(REG_DPU_EW_OP_VALUE_3, 0);
    EMIT(REG_DPU_EW_OP_VALUE_4, 0);
    EMIT(REG_DPU_EW_OP_VALUE_5, 0);
    EMIT(REG_DPU_EW_OP_VALUE_6, 0);
    EMIT(REG_DPU_EW_OP_VALUE_7, 0);

    EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(12))
    EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(12))

    emit_raw(&regs, DPU | 0x1, 0x40c4, 0);
    EMIT(REG_DPU_LUT_ACCESS_CFG, 0);
    EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
    EMIT(REG_DPU_LUT_CFG, 0);
    EMIT(REG_DPU_LUT_INFO, 0);
    EMIT(REG_DPU_LUT_LE_START, 0);
    EMIT(REG_DPU_LUT_LE_END, 0);
    EMIT(REG_DPU_LUT_LO_START, 0);
    EMIT(REG_DPU_LUT_LO_END, 0);
    EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0);
    EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
    EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0);
    EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);

    EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma))
    EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(input_dma))
    EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR(weights_dma))
        
    EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(9));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
         DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(0));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
         DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(7));

      EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(0));

      EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
      EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG,
         DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(2)); // 16 bit
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
         DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(12));
         //0x2001000178495044
         uint32_t rdma_feat_mode_cfg = 0x0;
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2);

         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15);
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(0);
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2);
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(0);
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1);
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(0);
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1);
          
         EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat_mode_cfg);
         EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
         EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH,
            DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(2));
         EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
         EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
            DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
               DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));
         EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
           DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(2)); 
         emit_raw(&regs, 0x00, 0x00, 0);
         EMIT(REG_PC_REGISTER_AMOUNTS, 0);

         //util_dynarray_append(regs, uint64_t, 0x0041000000000000);
         // 0x0081000000180008
         push(&regs, 0x0101000000000014);
         emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(12) | PC_OPERATION_ENABLE_OP_EN(0));           
/**
0x1001000100014084 = 0x00010001
0x1001000002004084 = 0x00000200
*/
   
      uint64_t npu_regs_a[regs.size];
      memcpy(npu_regs_a, regs.data, regs.size * sizeof(uint64_t));  // Copy elements to array

    memcpy(regcmd,npu_regs_a,sizeof(npu_regs_a));

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300; // wait for DPU to finish
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = sizeof(npu_regs_a)/sizeof(uint64_t); //nInstrs - 1;
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

int submitTask(int fd, uint64_t tasks_obj)
{
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

/**
 * @brief Float16 operation with specified ALU algorithm
 * @param a First float16 operand
 * @param b Second float16 operand
 * @param alu_algorithm The ALU algorithm to use (0-10)
 * @return Result of operation in float16 format
 */
__fp16* float16_alu_op(__fp16* a, __fp16* b, uint32_t alu_algorithm)
{
    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;

    struct MemHandles handles = createRegCmd(fd, get_type_size(dtype), alu_algorithm);
    __fp16 *weights_fp16 = (__fp16*)(handles.weights);
    __fp16 *feature_data_fp16 = (__fp16*)(handles.input);
    __fp16 *output_data = (__fp16*)(handles.output);
    
    memcpy(weights_fp16, a, get_type_size(dtype));
    memcpy(feature_data_fp16, b, get_type_size(dtype));

    int ret = submitTask(fd, handles.tasks_obj);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return NULL;
    }
    return output_data;
}

/**
 * @brief Float16 addition operation (ALU algorithm 2)
 * @param a First float16 operand
 * @param b Second float16 operand
 * @return Sum of a and b in float16 format
 */
__fp16* float16_add_op(__fp16* a, __fp16* b)
{
    return float16_alu_op(a, b, 2); // ALU algorithm 2 = Add
}


/**
 * @brief Int16 operation with specified ALU algorithm
 * @param a First int16 operand
 * @param b Second int16 operand
 * @param alu_algorithm The ALU algorithm to use (0-10)
 * @return Result of operation in int16 format
 */
 int16_t* int16_alu_op(int16_t* a, int16_t* b, uint32_t alu_algorithm)
 {
     int fd = getDeviceFd();
     rknn_tensor_type dtype = RKNN_TENSOR_INT16;

     struct MemHandles handles = createRegCmd(fd, get_type_size(dtype), alu_algorithm);
     int16_t *weights_int16 = (int16_t*)(handles.weights);
     int16_t *feature_data_int16 = (int16_t*)(handles.input);
     int16_t *output_data = (int16_t*)(handles.output);

     memcpy(weights_int16, a, get_type_size(dtype));
     memcpy(feature_data_int16, b, get_type_size(dtype));
 
     int ret = submitTask(fd, handles.tasks_obj);
     if(ret < 0) {
         printf("RKNPU_SUBMIT failed %d\n",ret);
         return NULL;
     }
     return output_data;
}

/**
 * @brief Int16 addition operation (ALU algorithm 2)
 * @param a First int16 operand
 * @param b Second int16 operand
 * @return Sum of a and b in int16 format
 */
 int16_t* int16_add_op(int16_t* a, int16_t* b)
 {
     return int16_alu_op(a, b, 2); // ALU algorithm 2 = Add
}


/**
 * @brief Int8 operation with specified ALU algorithm
 * @param a First int8 operand
 * @param b Second int8 operand
 * @param alu_algorithm The ALU algorithm to use (0-10)
 * @return Result of operation in int8 format
 */
 int8_t* int8_alu_op(int8_t* a, int8_t* b, uint32_t alu_algorithm)
 {

     int fd = getDeviceFd();
     rknn_tensor_type dtype = RKNN_TENSOR_INT8;
     initArray(&regs, 1024);

     struct MemHandles handles = createRegCmd(fd, get_type_size(dtype), alu_algorithm);
     int8_t *weights_int8 = (int8_t*)(handles.weights);
     int8_t *feature_data_int8 = (int8_t*)(handles.input);
     int8_t *output_data = (int8_t*)(handles.output);

     memcpy(weights_int8, a, get_type_size(dtype));
     memcpy(feature_data_int8, b, get_type_size(dtype));
 
     int ret = submitTask(fd, handles.tasks_obj);
     if(ret < 0) {
         printf("RKNPU_SUBMIT failed %d\n",ret);
         return NULL;
     }
     return output_data;
}

/**
 * @brief Int8 addition operation (ALU algorithm 2)
 * @param a First int8 operand
 * @param b Second int8 operand
 * @return Sum of a and b in int8 format
 */
 int8_t* int8_add_op(int8_t* a, int8_t* b)
 {

     return int8_alu_op(a, b, 2); // ALU algorithm 2 = Add
}


#ifdef __cplusplus
}
#endif

#endif /* RKNNOPS_H */