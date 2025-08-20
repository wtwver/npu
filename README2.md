fix matmul result is wrong

# How to dump GEM

run gdb --args ./matmul_api_demo
break rknn_destroy_mem


when break, run ./hello to dump gem to a file
try a different input format like int8 to fp16 in ./matal_api_demo
diff the difference in dump gem

find out which gem represent instruction, which for input A and inputB and outputC


# How to decode dump gem1

python3 decode.py --xml registers.xml --dump gem1_regdump.bin


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
