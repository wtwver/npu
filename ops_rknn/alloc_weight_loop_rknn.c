// gcc -o alloc_weight_loop_rknn alloc_weight_loop_rknn.c -I../include -lrknn_api

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "rknn_matmul_api.h"

static int align_up(int value, int align) {
  if (align <= 0) {
    return value;
  }
  return ((value + align - 1) / align) * align;
}

static void log_timestamp(int iter) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  printf("[iter %d] time %lld.%09ld\n",
         iter,
         (long long)ts.tv_sec,
         ts.tv_nsec);
}

int main(void) {
  const int loops = 50;
  const int dim = 8165;
  const int align_k = 32;
  const int align_n = 32;
  const int aligned_k = align_up(dim, align_k);
  const int aligned_n = align_up(dim, align_n);
  const size_t weight_elems = (size_t)dim * (size_t)dim;
  const size_t weight_bytes = weight_elems * sizeof(__fp16);

  rknn_matmul_ctx ctx = 0;
  rknn_matmul_info info;
  memset(&info, 0, sizeof(info));
  info.M = 1;
  info.K = aligned_k;
  info.N = aligned_n;
  info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
  info.B_layout = 0;
  info.AC_layout = 0;
  info.B_quant_type = 0;
  info.AC_quant_type = 0;
  info.iommu_domain_id = 0;
  info.group_size = 0;

  rknn_matmul_io_attr attr;
  memset(&attr, 0, sizeof(attr));
  int ret = rknn_matmul_create(&ctx, &info, &attr);
  if (ret != 0) {
    printf("rknn_matmul_create failed: %d\n", ret);
    return 1;
  }

  printf("alloc_weight_loop_rknn: dim=%d elems=%zu bytes=%zu aligned_k=%d aligned_n=%d attr.B.size=%u\n",
         dim,
         weight_elems,
         weight_bytes,
         aligned_k,
         aligned_n,
         attr.B.size);

  for (int i = 0; i < loops; ++i) {
    log_timestamp(i);
    rknn_tensor_mem* mem = rknn_create_mem(ctx, attr.B.size);
    if (mem == NULL) {
      printf("[iter %d] rknn_create_mem failed\n", i);
      break;
    }
    if (mem->virt_addr != NULL) {
      memset(mem->virt_addr, 0, mem->size);
    }
    printf("[iter %d] size=%u phys=0x%llx fd=%d\n",
           i,
           mem->size,
           (unsigned long long)mem->phys_addr,
           mem->fd);
    rknn_destroy_mem(ctx, mem);
  }

  rknn_matmul_destroy(ctx);
  return 0;
}
