// gcc -o alloc_weight_loop alloc_weight_loop.c -I../include -ldrm -lm

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "rknnops.h"

static void log_timestamp(int iter) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  printf("[iter %d] time %lld.%09ld\n",
         iter,
         (long long)ts.tv_sec,
         ts.tv_nsec);
}

int main(void) {
  const int loops = 500;
  const int dim = 8165;
  size_t weight_elems = (size_t)dim * (size_t)dim;
  size_t weight_bytes = weight_elems * sizeof(__fp16);
  size_t weight_aligned = (weight_bytes + 0x3f) & ~((size_t)0x3f);
  size_t alloc_bytes = REGCMD_RESERVED + weight_aligned;
  size_t map_bytes = page_align_size(alloc_bytes);

  int fd = getDeviceFd();
  printf("alloc_weight_loop: dim=%d elems=%zu bytes=%zu alloc_bytes=%zu map_bytes=%zu\n",
         dim, weight_elems, weight_bytes, alloc_bytes, map_bytes);

  for (int i = 0; i < loops; i++) {
    log_timestamp(i);
    uint64_t dma = 0;
    uint64_t obj = 0;
    uint32_t handle = 0;
    void *map = mem_allocate(fd, alloc_bytes, &dma, &obj,
                             RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE |
                                 RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT,
                             &handle);
    if (!map) {
      printf("[iter %d] allocation failed\n", i);
      break;
    }
    printf("[iter %d] handle=%u dma=0x%llx obj=0x%llx\n",
           i,
           handle,
           (unsigned long long)dma,
           (unsigned long long)obj);

    munmap(map, map_bytes);
    if (handle) {
      mem_destroy(fd, handle, obj);
    }
  }

  close(fd);
  return 0;
}
