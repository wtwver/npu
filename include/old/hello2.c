#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <libdrm/drm.h>
#include <sys/mman.h>

#include "rknpu-ioctl.h"

// Function to create a flink name for a GEM object
int create_flink_name(int fd, uint32_t handle, uint32_t *flink_name) {
    struct drm_gem_flink flink_req = {
        .handle = handle,
        .name = 0
    };

    int ret = ioctl(fd, DRM_IOCTL_GEM_FLINK, &flink_req);
    if (ret < 0) {
        printf("DRM_IOCTL_GEM_FLINK failed: %s\n", strerror(errno));
        return ret;
    }

    *flink_name = flink_req.name;
    printf("Created flink name %u for handle %u\n", *flink_name, handle);
    return 0;
}

// Function to open a GEM object by flink name
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

void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, int flags, uint32_t *handle_out) {
    int ret;

    struct rknpu_mem_create mem_create = {
        .flags = RKNPU_MEM_IOMMU | RKNPU_MEM_ZEROING | flags | RKNPU_MEM_CACHEABLE,
        .size = size,
    };

    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if(ret < 0) exit(2);
    fprintf(stderr, "mem create returned handle %08x, obj_addr %16llx dma_addr %16llx\n", mem_create.handle, mem_create.obj_addr, mem_create.dma_addr);

    struct rknpu_mem_map mem_map = { .handle = mem_create.handle };
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
    printf("memmap returned %d %llx\n", ret, mem_map.offset);
    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);
    if(map == MAP_FAILED) exit(2);
    printf("mmap returned %p\n", map);

    *dma_addr = mem_create.dma_addr;
    *obj = mem_create.obj_addr;
    if (handle_out) *handle_out = mem_create.handle;

    return map;
}

void mem_sync(int fd, uint64_t obj_addr, uint64_t offset, uint64_t size) {
    int ret;

    struct rknpu_mem_sync m_sync = {
        .obj_addr = obj_addr,
        .offset = offset,
        .size = size,
		.flags = RKNPU_MEM_SYNC_TO_DEVICE,
    };
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &m_sync);
    printf("memsync returned %d\n", ret);
}

void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr) {
    int ret;

    struct rknpu_mem_destroy mem_destroy_req = {
        .handle = handle,
        .obj_addr = obj_addr,
    };

    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &mem_destroy_req);
    if (ret < 0) {
        printf("Failed to destroy memory object with handle %u: %s\n", handle, strerror(errno));
    } else {
        printf("Successfully destroyed memory object with handle %u\n", handle);
    }
}

int main(int argc, char **argv) {
    char buf1[256], buf2[256], buf3[256];
    memset(buf1, 0, sizeof(buf1));
    memset(buf2, 0, sizeof(buf2));
    memset(buf3, 0, sizeof(buf3));

    int ret;
    // Open DRI called "rknpu"
    int fd = open("/dev/dri/card1", O_RDWR);
    if(fd<0) exit(1);
    struct drm_version dv;
    memset(&dv, 0, sizeof(dv));
    dv.name = buf1;
    dv.name_len = sizeof(buf1);
    dv.date = buf2;
    dv.date_len = sizeof(buf2);
    dv.desc = buf3;
    dv.desc_len = sizeof(buf3);

    ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
    printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
    if(ret < 0) exit(2);

    struct drm_unique du;
    du.unique = buf1;
    du.unique_len = sizeof(buf1);;
    ret = ioctl(fd, DRM_IOCTL_GET_UNIQUE, &du);
    printf("du is %s\n", du.unique);
    if(ret < 0) exit(2);

    uint64_t instr_dma, instr_obj;
    uint32_t instr_handle;
    uint64_t *instrs = mem_allocate(fd, 1024*1024, &instr_dma, &instr_obj, 0, &instr_handle);
    printf("%s %d\n", __FILE__, __LINE__);

	// Why is this a GEM?!?
    uint64_t tasks_dma, tasks_obj;
    uint32_t tasks_handle;
    struct rknpu_task *tasks = mem_allocate(fd, 1024*1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);
    printf("%s %d\n", __FILE__, __LINE__);

    uint64_t input_dma, input_obj;
    uint32_t input_handle;
    void *input = mem_allocate(fd, 1024*1024, &input_dma, &input_obj, 0, &input_handle);
    printf("%s %d\n", __FILE__, __LINE__);

    uint64_t weight_dma, weight_obj;
    uint32_t weight_handle;
    void *weight = mem_allocate(fd, 1024*1024, &weight_dma, &weight_obj, 0, &weight_handle);
    printf("%s %d\n", __FILE__, __LINE__);

    uint64_t output_dma, output_obj;
    uint32_t output_handle;
    void *output = mem_allocate(fd, 1024*1024, &output_dma, &output_obj, 0, &output_handle);
    printf("%s %d\n", __FILE__, __LINE__);

    // Create flink names for the allocated memory objects
    uint32_t instr_flink, tasks_flink, input_flink, weight_flink, output_flink;

    if (create_flink_name(fd, instr_handle, &instr_flink) < 0) {
        printf("Failed to create flink name for instr\n");
        goto cleanup;
    }

    if (create_flink_name(fd, tasks_handle, &tasks_flink) < 0) {
        printf("Failed to create flink name for tasks\n");
        goto cleanup;
    }

    if (create_flink_name(fd, input_handle, &input_flink) < 0) {
        printf("Failed to create flink name for input\n");
        goto cleanup;
    }

    if (create_flink_name(fd, weight_handle, &weight_flink) < 0) {
        printf("Failed to create flink name for weight\n");
        goto cleanup;
    }

    if (create_flink_name(fd, output_handle, &output_flink) < 0) {
        printf("Failed to create flink name for output\n");
        goto cleanup;
    }

    printf("Created flink names: instr=%u, tasks=%u, input=%u, weight=%u, output=%u\n",
           instr_flink, tasks_flink, input_flink, weight_flink, output_flink);

	struct rknpu_action act = {
		.flags = RKNPU_ACT_RESET,
	};
	ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    printf("%s %d\n", __FILE__, __LINE__);

#define IRQ_CNA_FEATURE_GROUP0	(1 << 0)
#define IRQ_CNA_FEATURE_GROUP1	(1 << 1)
#define IRQ_CNA_WEIGHT_GROUP0	(1 << 2)
#define IRQ_CNA_WEIGHT_GROUP1	(1 << 3)
#define IRQ_CNA_CSC_GROUP0		(1 << 4)
#define IRQ_CNA_CSC_GROUP1		(1 << 5)
#define IRQ_CORE_GROUP0			(1 << 6)
#define IRQ_CORE_GROUP1			(1 << 7)
#define IRQ_DPU_GROUP0			(1 << 8)
#define IRQ_DPU_GROUP1			(1 << 9)
#define IRQ_PPU_GROUP0			(1 << 10)
#define IRQ_PPU_GROUP1			(1 << 11)
#define IRQ_DMA_READ_ERROR		(1 << 12)
#define IRQ_DMA_WRITE_ERROR		(1 << 13)

#define RKNPU_JOB_DONE (1 << 0)
#define RKNPU_JOB_ASYNC (1 << 1)
#define RKNPU_JOB_DETACHED (1 << 2)

#define RKNPU_CORE_AUTO_MASK 0x00
#define RKNPU_CORE0_MASK 0x01
#define RKNPU_CORE1_MASK 0x02
#define RKNPU_CORE2_MASK 0x04


#if 0
	for(int i=0; i<10; i++) {
		instrs[0 + 4 * i] = (0x0101 << 48) | 0x14 | (instr_dma << 8); // Jump to xxx
		instrs[1 + 4 * i] = 0x0101000000000014; // Write 0 instructions left (Write 0 to pc_register_amounts 0x14)
		instrs[2 + 4 * i] = 0x0041000000000000; // Documentation says that this is needed...
		instrs[3 + 4 * i] = 0x0101000000000014; // Write 0 instructions left (Write 0 to pc_register_amounts 0x14)
		//instrs[3 + 4 * i] = 0x00810000000d0008; // Set all block's op_en to true
	}
#else
#define INSTR(TGT, value, reg) (((uint64_t)TGT)<< 48) | ( ((uint64_t)value) << 16) | (uint64_t)reg
    int nInstrs = 0;

#include "instrs.h"
#endif
    printf("%s %d\n", __FILE__, __LINE__);

	tasks[0].flags  = 0;
	tasks[0].op_idx = 1;
	tasks[0].enable_mask = 0x7f; //unused?!?
	//tasks[0].int_mask = 0x1ffff; // Ask for any interrupt at all...?
	tasks[0].int_mask = 0xf; // Ask for any interrupt at all...?
	//tasks[0].int_mask = 0x30c;
	tasks[0].int_clear = 0x1ffff;
	tasks[0].regcfg_amount = nInstrs - 0;
	tasks[0].regcfg_offset = 0;
	tasks[0].regcmd_addr = instr_dma;
    printf("%s %d\n", __FILE__, __LINE__);

	mem_sync(fd, tasks_obj, 0, 1024*1024);
    printf("%s %d\n", __FILE__, __LINE__);
	mem_sync(fd, instr_obj, 0, 1024*1024);
    printf("%s %d\n", __FILE__, __LINE__);

	struct rknpu_submit submit = {
		.flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK /*| RKNPU_JOB_PINGPONG*/,
		.timeout = 1000,
		.task_start = 0,
		.task_number = 1,
		.task_counter = 0,
		.priority = 0,
		.task_obj_addr = tasks_obj,
		.regcfg_obj_addr = 0, // unused?
		.task_base_addr = instr_dma,
		.user_data = 0, //unused
		.core_mask = 1, // = auto
		.fence_fd = 0, //unused because flags didn't ask for a fence in .flags
		.subcore_task = {
			// Only use core 1, nothing for core 2/3
			{
				.task_start = 0,
				.task_number = 1,
			}, { 0, 0}, {0, 0},
		},
	};

    printf("%s %d\n", __FILE__, __LINE__);
	ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
	printf("Submit returned %d\n", ret);

    // Demonstrate how to use flink names to access the GEM objects
    printf("\n=== Demonstrating GEM FLINK functionality ===\n");
    printf("These flink names can be used by other processes to access the same memory:\n");
    printf("instr: %u, tasks: %u, input: %u, weight: %u, output: %u\n",
           instr_flink, tasks_flink, input_flink, weight_flink, output_flink);

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
                printf("First few bytes from reopened output: %02x %02x %02x %02x\n",
                       ((uint8_t*)reopened_map)[0], ((uint8_t*)reopened_map)[1],
                       ((uint8_t*)reopened_map)[2], ((uint8_t*)reopened_map)[3]);

                munmap(reopened_map, reopened_size);
            }
        }
    }

cleanup:
    // Cleanup memory objects
    printf("Cleaning up memory objects...\n");

    // Properly destroy the memory objects
    mem_destroy(fd, instr_handle, instr_obj);
    mem_destroy(fd, tasks_handle, tasks_obj);
    mem_destroy(fd, input_handle, input_obj);
    mem_destroy(fd, weight_handle, weight_obj);
    mem_destroy(fd, output_handle, output_obj);

    close(fd);
    return ret;
}
