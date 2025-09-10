/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <unistd.h>
 #include <sys/ioctl.h>
 #include <stdlib.h>
 #include <string.h>
 #include <fcntl.h>
 #include <errno.h>
 #include <sys/mman.h>
 
 #include <libdrm/drm.h>
 
 #include "rknpu-ioctl.h"
 
   // Test currently runs against kernel 5.10 haven't tested 6.1 kernel.
 

 
   // Hand crafted register definitions for a simple fp 16 convolution which
   // can be done with single NPU task because the input cube and weights are
   // small. Feature data is 4x1x40 and weights 1x1x40x16, output is 4x1x16.
   // Note: numerous registers require changes if the input cube or weight
   // dimensions are altered.

   // Updated npu_regs[] to match the values from the latest npu_regs_map2 dump
  
   uint64_t npu_regs[] = {
    0x10010000000e4004,
    0x20010000000e5004,
    0x1001000001e5400c,
    0x1001900000044010,
    0x1001000000004014,
    0x1001ffed02004020,
    0x1001000000c04024,
    0x1001000000094030,
    0x1001000000004034,
    0x1001000000004038,
    0x100100030003403c,
    0x1001000000534040,
    0x1001000000004044,
    0x1001000000004048,
    0x100100000000404c,
    0x1001000000024050,
    0x1001000000004054,
    0x1001000000034058,
    0x100100000009405c,
    0x1001000000534060,
    0x1001000000004064,
    0x1001000000004068,
    0x100100000000406c,
    0x100110c202c04070,
    0x1001000000004074,
    0x1001000000014078,
    0x100100000000407c,
    0x1001000000004080,
    0x1001000000014084,
    0x1001000000004088,
    0x1001000000004090,
    0x1001000000004094,
    0x1001000000004098,
    0x100100000000409c,
    0x10010000000040a0,
    0x10010000000040a4,
    0x10010000000040a8,
    0x10010000000040ac,
    0x1001000000c040c0,
    0x10010000000040c4,
    0x1001000000004100,
    0x1001000000004104,
    0x1001000000004108,
    0x100100000000410c,
    0x1001000000004110,
    0x1001000000004114,
    0x1001000000004118,
    0x100100000000411c,
    0x1001000000004120,
    0x1001000000004124,
    0x1001000000004128,
    0x100100000000412c,
    0x200100000009500c,
    0x2001000000005010,
    0x2001000000035014,
    0x2001ffed00805018,
    0x200100000000501c,
    0x2001000000005020,
    0x2001000000005028,
    0x200100000000502c,
    0x20014000000c5034,
    0x2001ffed01405038,
    0x2001000000c05040,
    0x2001000278815044,
    0x2001000000005048,
    0x200100000020504c,
    0x2001000000005064,
    0x2001010101015068,
    0x200100000020506c,
    0x0000000000000000,
    0x0101000000000014,
    0x0041000000000000,
    0x0081000000180008
  
   };
 
 int feature_data(int C, int H, int W, int C2, int c, int h, int w) {
 
   int plane = (c-1)/C2;
   int src = plane * H * W * C2;
   int offset = (c-1) % C2;
   int pos = src + C2 * ((h-1) * W + (w-1)) + offset;
   return pos;
 }
 
 
 int weight_data(int K, int C, int k, int c) {
 
   // fp16 format
 
   int cpg=32;  
   int kgs = (C/cpg)+1;  
   int gi = ((c-1)/cpg)+1;
   int C2_gs = ((C-1)/8)+1;
   int c2_gs = ((c-1)/8)+1;
   int c1_gs = ((c2_gs-1)/4);
   int dst = c1_gs * 32 * K;
   int rgs = (C2_gs)-(c1_gs*4);
   int r=(c-1)%cpg;
   if (gi == kgs) {
     dst = dst + (rgs*8*(k-1));
   } else {
     dst = dst + (cpg*(k-1));
   }
   dst = dst + r;
   return dst;
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
   *handle = mem_create.handle;
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
 
 int main(int argc, char **argv) {
 
   char buf1[256], buf2[256], buf3[256];
   memset(buf1, 0, sizeof(buf1));
   memset(buf2, 0, sizeof(buf2));
   memset(buf3, 0, sizeof(buf3));
 
   int ret;
   int M=10;
   int K=1;
   int N=1;
 
   // Open DRI called "rknpu"
   int fd = open("/dev/dri/card1", O_RDWR);
   if(fd<0) {
     printf("Failed to open /dev/dri/card1 %d\n",errno);
     exit(1);
   }
 
   struct drm_version dv;
   memset(&dv, 0, sizeof(dv));
   dv.name = buf1;
   dv.name_len = sizeof(buf1);
   dv.date = buf2;
   dv.date_len = sizeof(buf2);
   dv.desc = buf3;
   dv.desc_len = sizeof(buf3);
 
   ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
   if (ret <0) {
     printf("DRM_IOCTL_VERISON failed %d\n",ret);
     exit(1);
   }
   printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
 
   
   uint64_t tasks_dma, tasks_obj;
   uint32_t tasks_handle;
   struct rknpu_task *tasks = static_cast<struct rknpu_task*>(mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle));
 
  
   uint64_t regcmd_dma, regcmd_obj;
   uint32_t regcmd_handle;
   uint64_t *regcmd = static_cast<uint64_t*>(mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle));

   uint64_t input_dma, input_obj;
   uint32_t input_handle;
   void *input = mem_allocate(fd, 4194304, &input_dma, &input_obj, 0, &input_handle);
 
   uint64_t weights_dma, weights_obj;
   uint32_t weights_handle;
   void *weights = mem_allocate(fd, 4194304, &weights_dma, &weights_obj, 0, &weights_handle);
 
   uint64_t output_dma, output_obj;
   uint32_t output_handle;
   void *output = mem_allocate(fd, 4194304, &output_dma, &output_obj, 0, &output_handle);
 
   printf("input dma is %lx, output dma is %lx, weights dma is %lx\n", input_dma, output_dma, weights_dma);
   if ((regcmd == NULL) || (tasks == NULL) || (input == NULL) || (weights == NULL) || (output == NULL)) {
     printf("Failed to allocate memory \n");
     exit(1);
   }
 
 
   // Set input, weights and output physical memory locations. Note limited to 
   // a 32 bit address size (4GB)
   npu_regs[55] = npu_regs[55] | ((input_dma & 0xFFFFFFFF) <<16);
   npu_regs[61] = npu_regs[61] | ((weights_dma & 0xFFFFFFFF)  <<16);
   npu_regs[5] = npu_regs[5] | ((output_dma & 0xFFFFFFFF) <<16);
   printf("input_dma %lx\n", input_dma);
   printf("weights_dma %lx\n", weights_dma);
   printf("output_dma %lx\n", output_dma);
   printf("npu_regs[79] %lx\n", npu_regs[79]);
   printf("npu_regs[85] %lx\n", npu_regs[85]);
   printf("npu_regs[29] %lx\n", npu_regs[29]);
   printf("Size of npu_regs %d\n",(int)(sizeof(npu_regs)/sizeof(uint64_t)));
   memcpy(regcmd,npu_regs,sizeof(npu_regs));
 
   tasks[0].flags  = 0;
   tasks[0].op_idx = 4;
   tasks[0].enable_mask = 0x18;
   tasks[0].int_mask = 0x300; // wait for DPU to finish
   tasks[0].int_clear = 0x1ffff;
   tasks[0].int_status = 0;
   tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t); //nInstrs - 1;
   tasks[0].regcfg_offset = 0;
   tasks[0].regcmd_addr = regcmd_dma;
 
   memset((void *)input,0,104*sizeof(int));
   memset((void *)weights,0,104*sizeof(int));
   memset((void *)output,0,104*sizeof(int));
 
   int *weights_fp16 = static_cast<int*>(weights);
   for (int i = 0; i < 104; ++i) {
       weights_fp16[i] =6;
   }

   int *feature_data_fp16 = static_cast<int*>(input);
   for (int i = 0; i < 104; ++i) {
       feature_data_fp16[i] = 7;
   }
 
   munmap(input,4194304);
   munmap(weights,4194304);

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
   for (int i = 0; i < 1; i++) {
    ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
    printf("RKNPU_SUBMIT returned %d\n", ret);
   }
  
 
   printf("=========================================================================================================\n");
   int *output_data = static_cast<int*>(output);
   printf("Output data:\n");
   for (size_t i = 0; i < 50; ++i) {
       printf("%d ", output_data[i]);
   }
 
   
   printf("\033[0;37m");
   printf("=========================================================================================================\n");
 
   
   void *map = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, 3, 0x100000000);


   if (map == MAP_FAILED) {
       perror("mmap failed");
   } else {
       uint64_t map_data[128]; // 1024 bytes / 8 bytes per uint64_t = 128
       memcpy(map_data, map, 1024);
 
       printf("map_data[0]: 0x%016lx\n", map_data[0]);
       printf("map_data[1]: 0x%016lx\n", map_data[1]);
       printf("map_data[2]: 0x%016lx\n", map_data[2]);
       printf("map_data[3]: 0x%016lx\n", map_data[3]);
   }
 
   
   struct rknpu_task *task_map = (struct rknpu_task *)map;
 
   printf("tasks[0] map: %p\n", (void*)task_map);
   printf("struct rknpu_task fields:\n");
   printf("  flags:         %u\n", task_map->flags);
   printf("  op_idx:        %u\n", task_map->op_idx);
   printf("  enable_mask:   %u\n", task_map->enable_mask);
   printf("  int_mask:      %u\n", task_map->int_mask);
   printf("  int_clear:     %u\n", task_map->int_clear);
   printf("  int_status:    %u\n", task_map->int_status);
   printf("  regcfg_amount: %u\n", task_map->regcfg_amount);
   printf("  regcfg_offset: %u\n", task_map->regcfg_offset);
   printf("  regcmd_addr:   0x%016lx\n", (unsigned long)task_map->regcmd_addr);

   void *regmap2 = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, 3, 0x100004000);
   printf("regmap2: %p\n", regmap2);
   if (regmap2 == MAP_FAILED) {
       perror("mmap regmap2 failed");
   } else {
       int64_t npu_regs_map2[1024 / sizeof(int64_t)];
       memcpy(npu_regs_map2, regmap2, 1024);
       for (int i = 0; i < 20; i++) {
           printf("npu_regs_map2[%d]: 0x%016lx\n", i, npu_regs_map2[i]);
       }
       // It is good practice to unmap when done
       munmap(regmap2, 1024);
   }

   printf("input: %p\n", input);



   munmap(regcmd,1024);
   munmap(tasks,1024);
   munmap(input,4194304);
   munmap(weights,4194304);
   munmap(output,4194304);
 
   mem_destroy(fd, regcmd_handle, regcmd_obj);
   mem_destroy(fd, tasks_handle, tasks_obj );
   mem_destroy(fd, input_handle, input_obj);
   mem_destroy(fd, weights_handle, weights_obj);
   mem_destroy(fd, output_handle, output_obj);
 
   close(fd);
   return 0;
 }