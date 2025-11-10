################################################################################
Conv2D Multi-Test Suite
Testing 1 different convolution shapes
################################################################################

================================================================================
TEST: conv2d with input shape (1,3,5,7), weight shape (6,3,2,3)
  Weight shape: (6,3,2,3)
================================================================================
I RKNN: [22:45:08.787] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [22:45:08.788] RKNN Driver Information, version: 0.9.2
I RKNN: [22:45:08.788] RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 (@2025-04-03T08:26:16)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
D RKNN: [22:45:08.788] allocated memory, name: task, virt addr: 0x7ff7ff6000, dma addr: 0xffec6000, obj addr: 0xffffff8274765400, size: 440, aligned size: 4096, fd: 4, handle: 1, flags: 0x40b, gem name: 1, iommu domain id: 0
D RKNN: [22:45:08.789] allocated memory, name: weight, virt addr: 0x7ff7ff3000, dma addr: 0xffec8000, obj addr: 0xffffff8274763c00, size: 9728, aligned size: 12288, fd: 5, handle: 2, flags: 0x403, gem name: 2, iommu domain id: 0
D RKNN: [22:45:08.789] subgraph0 regcmdbuffer size: 6656, taskbuffer size: 440
D RKNN: [22:45:08.789] allocated memory, name: internal, virt addr: 0x7ff7ff2000, dma addr: 0xffec7000, obj addr: 0xffffff8274766c00, size: 896, aligned size: 4096, fd: 6, handle: 3, flags: 0x403, gem name: 3, iommu domain id: 0
D RKNN: [22:45:08.789] ---------------------------------------------------------------------------------------------------------------
D RKNN: [22:45:08.789]                                      Feature Tensor Information Table                       
D RKNN: [22:45:08.789] -----------------------------------------------------------------------------+---------------------------------
D RKNN: [22:45:08.789] ID  User           Tensor   DataType  DataFormat   OrigShape    NativeShape  |     [Start       End)       Size
D RKNN: [22:45:08.789] -----------------------------------------------------------------------------+---------------------------------
D RKNN: [22:45:08.789] 1   Conv           input    FLOAT16   NC1HWC2      (1,3,5,7)    (1,4,5,7,3)  | 0xffec7000 0xffec70f0 0x000000f0
D RKNN: [22:45:08.789] 2   OutputOperator output   FLOAT16   NC1HWC2      (1,6,4,5)    (1,8,4,5,8)  | 0xffec7100*0xffec7380 0x00000280
D RKNN: [22:45:08.789] -----------------------------------------------------------------------------+---------------------------------
D RKNN: [22:45:08.789] ------------------------------------------------------------------------------
D RKNN: [22:45:08.789]                           Const Tensor Information Table         
D RKNN: [22:45:08.789] --------------------------------------------+---------------------------------
D RKNN: [22:45:08.789] ID  User Tensor      DataType  OrigShape    |     [Start       End)       Size
D RKNN: [22:45:08.789] --------------------------------------------+---------------------------------
D RKNN: [22:45:08.789] 1   Conv conv.weight FLOAT16   (6,3,2,3)    | 0xffec8000 0xffec8240 0x00000240
D RKNN: [22:45:08.789] --------------------------------------------+---------------------------------
D RKNN: [22:45:08.789] ----------------------------------------
D RKNN: [22:45:08.789] Total Internal Memory Size: 0.875KB
D RKNN: [22:45:08.789] Total Weight Memory Size: 2.5625KB
D RKNN: [22:45:08.789] ----------------------------------------
D RKNN: [22:45:08.790] The RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU is not set and without GPU op in Graphs, OpenCL will not be initialized
  Native input type: FP16, format: NHWC
  Native input dims: 1 5 7 3
  Native input size: 210, size_with_stride: 240, w_stride: 8
  Native output type: FP16, format: NC1HWC2
  Native output dims: 1 2 4 5 8
  Native output size: 640, size_with_stride: 640, w_stride: 0
D RKNN: [22:45:08.791] allocated memory, name: external, virt addr: 0x7ff7ff1000, dma addr: 0xffecf000, obj addr: 0xffffff8274765800, size: 640, aligned size: 4096, fd: 7, handle: 4, flags: 0x403, gem name: 4, iommu domain id: 0
D RKNN: [22:45:08.791] import memory, fd: 7, refcount: 2
  Input tensor (NCHW)
    n=0
      c=0
        h=0: 0.609863 1.24023 2.28906 3.75977 5.64844 7.96094 10.6875 
        h=1: 1.50977 2.14062 3.18945 4.66016 6.55078 8.85938 11.5938 
        h=2: 3.00977 3.64062 4.69141 6.16016 8.04688 10.3594 13.0938 
        h=3: 5.10938 5.73828 6.78906 8.25781 10.1484 12.4609 15.1875 
        h=4: 7.80859 8.4375 9.49219 10.9609 12.8516 15.1562 17.8906 
      c=1
        h=0: 1.12012 2.38086 4.48047 7.42188 11.2031 15.8203 21.2812 
        h=1: 2.91992 4.17969 6.28125 9.21875 13 17.625 23.0781 
        h=2: 5.92188 7.17969 9.28125 12.2188 16 20.625 26.0781 
        h=3: 10.1172 11.3828 13.4766 16.4219 20.2031 24.8125 30.2812 
        h=4: 15.5234 16.7812 18.875 21.8125 25.5938 30.2188 35.6875 
      c=2
        h=0: 1.62988 3.51953 6.67188 11.0781 16.75 23.6875 31.875 
        h=1: 4.32812 6.21875 9.36719 13.7812 19.4531 26.375 34.5625 
        h=2: 8.82812 10.7188 13.8672 18.2812 23.9531 30.875 39.0625 
        h=3: 15.1328 17.0156 20.1719 24.5781 30.25 37.1875 45.375 
        h=4: 23.2344 25.125 28.2656 32.6875 38.3438 45.2812 53.4688 

Breakpoint 1, 0x0000007ff790c060 in rknn_run () from /lib/librknnrt.so
"============"drm name is rknpu - 20231018 - RKNPU driver
du is 
Dumping specified GEM objects...

=== Processing GEM 1 ===

==================================================
Processing GEM Flink 1
==================================================
gem flink 1: ret=0 handle=1 size=4096
DRM_IOCTL_RKNPU_MEM_MAP returned 0 0x100000000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294967296>

==================================================
Decoded rknpu_task entries for GEM 1
==================================================
Task 0 @ offset 0x0000
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x0000000d
  int_mask     : 0x00000300
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 108 entries (0x0000006c)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec8a40

Task 1 @ offset 0x0028
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x0000000d
  int_mask     : 0x00000300
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 104 entries (0x00000068)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec8dc0

Task 2 @ offset 0x0050
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x00000060
  int_mask     : 0x00000c00
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 26 entries (0x0000001a)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec9140

Task 3 @ offset 0x0078
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x0000000d
  int_mask     : 0x00000300
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 104 entries (0x00000068)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec9240

Task 4 @ offset 0x00a0
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x00000060
  int_mask     : 0x00000c00
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 26 entries (0x0000001a)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec95c0

Task 5 @ offset 0x00c8
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x0000000d
  int_mask     : 0x00000300
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 104 entries (0x00000068)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec96c0

Task 6 @ offset 0x00f0
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x00000060
  int_mask     : 0x00000c00
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 26 entries (0x0000001a)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec9a40

Task 7 @ offset 0x0118
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x0000000d
  int_mask     : 0x00000300
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 104 entries (0x00000068)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec9b40

Task 8 @ offset 0x0140
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x00000060
  int_mask     : 0x00000c00
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 26 entries (0x0000001a)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec9ec0

Task 9 @ offset 0x0168
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x0000000d
  int_mask     : 0x00000300
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 104 entries (0x00000068)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffec9fc0

Task 10 @ offset 0x0190
  flags        : 0x00000000
  op_idx       : 1 (0x00000001)
  enable_mask  : 0x00000060
  int_mask     : 0x00000c00
  int_clear    : 0x0001ffff
  int_status   : 0x00000000
  regcfg_amount: 26 entries (0x0000001a)
  regcfg_offset: 0x00000000
  regcmd_addr  : 0x00000000ffeca340

Decoded 11 task entries to dump/gem1_tasks.txt
drm name is rknpu - 20231018 - RKNPU driver
du is 
Dumping specified GEM objects...

=== Processing GEM 2 ===

==================================================
Processing GEM Flink 2
==================================================
gem flink 2: ret=0 handle=1 size=12288
DRM_IOCTL_RKNPU_MEM_MAP returned 0 0x100001000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=12288, pos=0, offset=4294971392>

==================================================
Processing GEM Flink 2 for Register Decode
==================================================
Successfully opened GEM via flink 2
memmap returned 0 0x100001000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=12288, pos=0, offset=4294971392> 4294971392
DEBUG: Found 199 registers in XML
DEBUG: Loaded 199 register definitions
DEBUG: base selection for GEM 2: dma_base=4293689344 phys_base=0x100001000 task_base=None chosen=0xffec8000 commands=1013
Successfully created dump/gem2_regdump.bin
[0xffec8000] lsb 00003c003c003c00 - noone Unknown
DEBUG: Looking for offset 0x3c00, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0xffec8010] lsb 0000bc00bc00bc00 - noone Unknown
DEBUG: Looking for offset 0xbc00, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0xffec8030] lsb 00003c003c003c00 - noone Unknown
[0xffec8040] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8060] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8080] lsb 00003c003c003c00 - noone Unknown
[0xffec8090] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec80b0] lsb 00003c003c003c00 - noone Unknown
[0xffec80d0] lsb 00003c003c003c00 - noone Unknown
[0xffec80e0] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8100] lsb 00003c003c003c00 - noone Unknown
[0xffec8110] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8120] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8140] lsb 00003c003c003c00 - noone Unknown
[0xffec8150] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8170] lsb 00003c003c003c00 - noone Unknown
[0xffec8190] lsb 00003c003c003c00 - noone Unknown
[0xffec81a0] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec81c0] lsb 00003c003c003c00 - noone Unknown
[0xffec81d0] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec81e0] lsb 00003c003c003c00 - noone Unknown
[0xffec81f0] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8210] lsb 00003c003c003c00 - noone Unknown
[0xffec8220] lsb 0000bc00bc00bc00 - noone Unknown
[0xffec8240] lsb 0101010101010101 - PC Unknown
[0xffec8248] lsb 0101010101010101 - PC Unknown
[0xffec8250] lsb 0101010101010101 - PC Unknown
[0xffec8258] lsb 0101010101010101 - PC Unknown
[0xffec8260] lsb 00000055967894c8 - noone Unknown
[0xffec82e0] lsb 00000055967894a0 - noone Unknown
[0xffec82e8] lsb 0000007f71be7290 - noone Unknown
[0xffec82f0] lsb 0000007f71be7290 - noone Unknown
[0xffec82f8] lsb 0000007f71be7290 - noone Unknown
[0xffec8328] lsb 0000000700000000 - noone         EMIT(REG_PC_VERSION, 0x00070000);
[0xffec8330] lsb 7468676965772e01 - DPU Unknown
[0xffec8338] lsb 0000000000000441 - noone Unknown
[0xffec8340] lsb 000000559677ea40 - noone Unknown
[0xffec8348] lsb 0000007fad430ab0 - noone Unknown
[0xffec8368] lsb 0000000400000004 - noone         EMIT(REG_PC_VERSION_NUM, 0x00040000);
[0xffec8370] lsb 0000005596789478 - noone Unknown
[0xffec83f0] lsb 0000005596789450 - noone Unknown
[0xffec83f8] lsb 0000007f71be7290 - noone Unknown
[0xffec8400] lsb 0000007f71be7290 - noone Unknown
[0xffec8408] lsb 0000007f71be7290 - noone Unknown
[0xffec8438] lsb 0000000700000000 - noone         EMIT(REG_PC_VERSION, 0x00070000);
[0xffec8448] lsb 0000000000000331 - noone Unknown
[0xffec8450] lsb 000000559677ea40 - noone Unknown
[0xffec8458] lsb 0000007fad430ab0 - noone Unknown
[0xffec8460] lsb 0000000000000401 - noone Unknown
[0xffec8478] lsb 0000000400000002 - noone Unknown
[0xffec8480] lsb 00000055967893f8 - noone Unknown
[0xffec8500] lsb 00000055967893d0 - noone Unknown
[0xffec8508] lsb 0000007f71be7290 - noone Unknown
[0xffec8510] lsb 0000007f71be7290 - noone Unknown
[0xffec8518] lsb 0000007f71be7290 - noone Unknown
[0xffec8548] lsb 0000000700000000 - noone         EMIT(REG_PC_VERSION, 0x00070000);
[0xffec8558] lsb 0000000000000221 - noone Unknown
[0xffec8560] lsb 000000559677ea40 - noone Unknown
[0xffec8568] lsb 0000007fad430ab0 - noone Unknown
[0xffec8570] lsb 0000000000000501 - noone Unknown
[0xffec8610] lsb 0000005596789380 - noone Unknown
[0xffec8618] lsb 0000007f71be7290 - noone Unknown
[0xffec8620] lsb 0000007f71be7290 - noone Unknown
[0xffec8628] lsb 0000007f71be7290 - noone Unknown
[0xffec8640] lsb 0101010101010101 - PC Unknown
[0xffec8648] lsb 0101010101010101 - PC Unknown
[0xffec8650] lsb 0101010101010101 - PC Unknown
[0xffec8658] lsb 0101010101010101 - PC Unknown
[0xffec8660] lsb 00000000ffffffff - noone Unknown
[0xffec8668] lsb 715f74757074756f - PC Unknown
[0xffec8670] lsb 78616d5f746e6175 - CORE Unknown
[0xffec8680] lsb 000000559673dc80 - noone Unknown
[0xffec8688] lsb 0000000000000241 - noone Unknown
[0xffec8690] lsb 0000007fad430ce0 - noone Unknown
[0xffec8698] lsb 0000005596772a00 - noone Unknown
[0xffec86b0] lsb 00000055932e4d6d - noone Unknown
[0xffec86c0] lsb 0000000000000003 - noone Unknown
[0xffec86c8] lsb 000000559673df30 - noone Unknown
[0xffec86d0] lsb 000000559673df50 - noone Unknown
[0xffec86d8] lsb 00000000000001f1 - noone Unknown
[0xffec86e0] lsb 00000055967726d0 - noone Unknown
[0xffec86e8] lsb 0000007fad430ab0 - noone Unknown
[0xffec86f0] lsb 0000000000000003 - noone Unknown
[0xffec86f8] lsb 0000005596772cc0 - noone Unknown
[0xffec8700] lsb 0000005596772b00 - noone Unknown
[0xffec8708] lsb 0000000000000041 - noone Unknown
[0xffec8710] lsb 000000559673dd60 - noone Unknown
[0xffec8718] lsb 000000559673dc70 - noone Unknown
[0xffec8728] lsb 0000000000000021 - noone Unknown
[0xffec8730] lsb 000000559673dd60 - noone Unknown
[0xffec8738] lsb 0000007fad430ab0 - noone Unknown
[0xffec8740] lsb 0000000000000070 - noone Unknown
[0xffec8750] lsb 00000055932abbad - noone Unknown
[0xffec8760] lsb 0000000000000003 - noone Unknown
[0xffec8768] lsb 000000559673df70 - noone Unknown
[0xffec8770] lsb 000000559673dde0 - noone Unknown
[0xffec8778] lsb 0000000000000151 - noone Unknown
[0xffec8780] lsb 00000055967726d0 - noone Unknown
[0xffec8788] lsb 0000007fad430ab0 - noone Unknown
[0xffec8798] lsb 0000000000000021 - noone Unknown
[0xffec87a0] lsb 0000005596772c90 - noone Unknown
[0xffec87a8] lsb 0000007fad430ab0 - noone Unknown
[0xffec87b0] lsb 0000000000000040 - noone Unknown
[0xffec87c0] lsb 00000055932aba0d - noone Unknown
[0xffec87d0] lsb 0000000000000003 - noone Unknown
[0xffec87d8] lsb 0000005596772ce0 - noone Unknown
[0xffec87e0] lsb 000000559673dc20 - noone Unknown
[0xffec87e8] lsb 00000000000000e1 - noone Unknown
[0xffec87f0] lsb 000000559673dcc0 - noone Unknown
[0xffec87f8] lsb 00000055967726d0 - noone Unknown
[0xffec8810] lsb 000000559321630d - noone Unknown
[0xffec8820] lsb 0000000000000040 - noone Unknown
[0xffec8830] lsb 00000055932e41ed - noone Unknown
[0xffec8840] lsb 0000000000000003 - noone Unknown
[0xffec8848] lsb 000000559673dd70 - noone Unknown
[0xffec8850] lsb 000000559673dd90 - noone Unknown
[0xffec8858] lsb 0000000000000071 - noone Unknown
[0xffec8860] lsb 0000005596772750 - noone Unknown
[0xffec8868] lsb 00000055967728a0 - noone Unknown
[0xffec8878] lsb 0000000000000021 - noone Unknown
[0xffec8880] lsb 000000559673df00 - noone Unknown
[0xffec8888] lsb 0000007fad430ab0 - noone Unknown
[0xffec8890] lsb 0000000000000040 - noone Unknown
[0xffec88a0] lsb 00000055932e4e2d - noone Unknown
[0xffec88b0] lsb 0000000000000003 - noone Unknown
[0xffec88b8] lsb 0000005596772ab0 - noone Unknown
[0xffec88c0] lsb 00000000000002a0 - noone Unknown
[0xffec88c8] lsb 0000000000000050 - noone Unknown
[0xffec88d0] lsb 00000055932ddc9d - noone Unknown
[0xffec88d8] lsb 00000055967758d0 - noone Unknown
[0xffec88e0] lsb 0000005596780550 - noone Unknown
[0xffec88e8] lsb 00000055967804a0 - noone Unknown
[0xffec88f0] lsb 0000005596780410 - noone Unknown
[0xffec88f8] lsb 000000559674b770 - noone Unknown
[0xffec8900] lsb 0000005596775530 - noone Unknown
[0xffec8908] lsb 0000005596749970 - noone Unknown
[0xffec8910] lsb 000000559661ddb0 - noone Unknown
[0xffec8918] lsb 00000000000001a1 - noone Unknown
[0xffec8920] lsb 0000007fad430c40 - noone Unknown
[0xffec8928] lsb 0000007fad430c40 - noone Unknown
[0xffec8938] lsb 0000000000000181 - noone Unknown
[0xffec8940] lsb 000000559673ddd0 - noone Unknown
[0xffec8948] lsb 0000007fad430ab0 - noone Unknown
[0xffec8958] lsb 0000000000000161 - noone Unknown
[0xffec8960] lsb 000000559673ddd0 - noone Unknown
[0xffec8968] lsb 0000007fad430ab0 - noone Unknown
[0xffec8978] lsb 0000000000000141 - noone Unknown
[0xffec8980] lsb 000000559673ddd0 - noone Unknown
[0xffec8988] lsb 0000007fad430ab0 - noone Unknown
[0xffec8998] lsb 0000000000000121 - noone Unknown
[0xffec89a0] lsb 0000005596781930 - noone Unknown
[0xffec89a8] lsb 000000559675d6c0 - noone Unknown
[0xffec89b0] lsb 0000000000000401 - noone Unknown
[0xffec89c8] lsb 0000000400000002 - noone Unknown
[0xffec89d0] lsb 000000559662a428 - noone Unknown
[0xffec8a40] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffec8a58] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec8a60] lsb 10010000000e4004 - DPU           EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
[0xffec8a68] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec8a70] lsb 0201000000701010 - CNA           EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(7));
[0xffec8a78] lsb 0201000000091014 - CNA           EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
[0xffec8a80] lsb 0201000800051020 - CNA           EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(8) | CNA_DATA_SIZE0_DATAIN_HEIGHT(5));
[0xffec8a88] lsb 0201000200081024 - CNA           EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(2) | CNA_DATA_SIZE1_DATAIN_CHANNEL(8));
[0xffec8a90] lsb 0201000000051028 - CNA           EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(5));
[0xffec8a98] lsb 020100000014102c - CNA           EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(20));
[0xffec8aa0] lsb 0201000002401030 - CNA           EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000240);
[0xffec8aa8] lsb 0201000000601034 - CNA           EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(96));
[0xffec8ab0] lsb 0201030200061038 - CNA           EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(3) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(2) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(6));
[0xffec8ab8] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffec8ac0] lsb 0201000000281044 - CNA           EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(40));
[0xffec8ac8] lsb 020100000001104c - CNA           EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_BYPASS(1));
[0xffec8ad0] lsb 0201000100001050 - CNA           EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
[0xffec8ad8] lsb 0201000100001054 - CNA           EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
[0xffec8ae0] lsb 0201000100001058 - CNA           EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
[0xffec8ae8] lsb 020100010000105c - CNA           EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
[0xffec8b08] lsb 0201ffec50001070 - CNA           EMIT(REG_CNA_FEATURE_DATA_ADDR, 0xffec5000);
[0xffec8b18] lsb 0201000f000f1078 - CNA           EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
[0xffec8b20] lsb 020100000008107c - CNA           EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(8));
[0xffec8b28] lsb 0201000000201080 - CNA           EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(32));
[0xffec8b30] lsb 0201000700051084 - CNA           EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(7) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(5));
[0xffec8b38] lsb 0201000000081088 - CNA           EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
[0xffec8b50] lsb 0201ffec80001110 - CNA           EMIT(REG_CNA_DCOMP_ADDR0, 0xffec8000);
[0xffec8bd8] lsb 020100000fff1180 - CNA           EMIT(REG_CNA_CVT_CON5, 0x00000fff);
[0xffec8be8] lsb 0801000002003010 - CORE          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2));
[0xffec8bf0] lsb 0801000300043014 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(3) | CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(4));
[0xffec8bf8] lsb 08010000000f3018 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
[0xffec8c08] lsb 0801000000003030 - CORE Unknown
[0xffec8c10] lsb 1001000001e4400c - DPU           EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
[0xffec8c18] lsb 1001480000024010 - DPU           EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
[0xffec8c28] lsb 1001ffec40004020 - DPU           EMIT(REG_DPU_DST_BASE_ADDR, 0xffec4000);
[0xffec8c30] lsb 1001000001404024 - DPU           EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(20));
[0xffec8c38] lsb 1001000000044030 - DPU           EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(4));
[0xffec8c40] lsb 1001000000034034 - DPU           EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(3));
[0xffec8c50] lsb 10010005000f403c - DPU           EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(5) | DPU_DATA_CUBE_CHANNEL_CHANNEL(15));
[0xffec8c58] lsb 1001000000534040 - DPU           EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
[0xffec8c78] lsb 1001000001264050 - DPU           EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1) | DPU_BS_OW_CFG_OD_BYPASS(1));
[0xffec8c88] lsb 10010000000f4058 - DPU           EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(15));
[0xffec8c90] lsb 100100030004405c - DPU           EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(3) | DPU_WDMA_SIZE_1_WIDTH_WDMA(4));
[0xffec8c98] lsb 1001000000534060 - DPU           EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
[0xffec8cb8] lsb 1001000003834070 - DPU           EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
[0xffec8cc8] lsb 1001000000014078 - DPU           EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
[0xffec8ce0] lsb 1001000100014084 - DPU           EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
[0xffec8d30] lsb 10010000028040c0 - DPU           EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(40));
[0xffec8d38] lsb 10010000000040c4 - DPU Unknown
[0xffec8db8] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));


[0xffec8dc0] lsb 10010000000e4004 - DPU           EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
[0xffec8dc8] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec8dd0] lsb 0201100000501010 - CNA           EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_RESERVED_0(16) | CNA_CONV_CON2_FEATURE_GRAINS(5));
[0xffec8dd8] lsb 0201000000091014 - CNA           EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
[0xffec8de0] lsb 0201000800031020 - CNA           EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(8) | CNA_DATA_SIZE0_DATAIN_HEIGHT(3));
[0xffec8de8] lsb 0201000200081024 - CNA           EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(2) | CNA_DATA_SIZE1_DATAIN_CHANNEL(8));
[0xffec8df0] lsb 0201000000051028 - CNA           EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(5));
[0xffec8df8] lsb 02010000000a102c - CNA           EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(10));
[0xffec8e00] lsb 0201000002401030 - CNA           EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000240);
[0xffec8e08] lsb 0201000000601034 - CNA           EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(96));
[0xffec8e10] lsb 0201030200061038 - CNA           EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(3) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(2) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(6));
[0xffec8e18] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffec8e20] lsb 0201000000181044 - CNA           EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(24));
[0xffec8e28] lsb 020100000001104c - CNA           EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_BYPASS(1));
[0xffec8e30] lsb 0201000100001050 - CNA           EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
[0xffec8e38] lsb 0201000100001054 - CNA           EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
[0xffec8e40] lsb 0201000100001058 - CNA           EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
[0xffec8e48] lsb 020100010000105c - CNA           EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
[0xffec8e68] lsb 0201ffec50001070 - CNA           EMIT(REG_CNA_FEATURE_DATA_ADDR, 0xffec5000);
[0xffec8e78] lsb 0201000f000f1078 - CNA           EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
[0xffec8e80] lsb 020100000008107c - CNA           EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(8));
[0xffec8e88] lsb 0201000000201080 - CNA           EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(32));
[0xffec8e90] lsb 0201000700031084 - CNA           EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(7) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(3));
[0xffec8e98] lsb 0201000000081088 - CNA           EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
[0xffec8eb0] lsb 0201ffec80001110 - CNA           EMIT(REG_CNA_DCOMP_ADDR0, 0xffec8000);
[0xffec8f38] lsb 020100000fff1180 - CNA           EMIT(REG_CNA_CVT_CON5, 0x00000fff);
[0xffec8f48] lsb 0801000002003010 - CORE          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2));
[0xffec8f50] lsb 0801000100043014 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(1) | CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(4));
[0xffec8f58] lsb 08010000000f3018 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
[0xffec8f68] lsb 0801000000003030 - CORE Unknown
[0xffec8f70] lsb 1001000001e4400c - DPU           EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
[0xffec8f78] lsb 1001480000024010 - DPU           EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
[0xffec8f88] lsb 1001ffec40004020 - DPU           EMIT(REG_DPU_DST_BASE_ADDR, 0xffec4000);
[0xffec8f90] lsb 1001000001404024 - DPU           EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(20));
[0xffec8f98] lsb 1001000000044030 - DPU           EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(4));
[0xffec8fa0] lsb 1001000000014034 - DPU           EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(1));
[0xffec8fb0] lsb 10010005000f403c - DPU           EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(5) | DPU_DATA_CUBE_CHANNEL_CHANNEL(15));
[0xffec8fb8] lsb 1001000000534040 - DPU           EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
[0xffec8fd8] lsb 1001000001264050 - DPU           EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1) | DPU_BS_OW_CFG_OD_BYPASS(1));
[0xffec8fe8] lsb 10010000000f4058 - DPU           EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(15));
[0xffec8ff0] lsb 100100010004405c - DPU           EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(1) | DPU_WDMA_SIZE_1_WIDTH_WDMA(4));
[0xffec8ff8] lsb 1001000000534060 - DPU           EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
[0xffec9018] lsb 1001000003834070 - DPU           EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
[0xffec9028] lsb 1001000000014078 - DPU           EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
[0xffec9040] lsb 1001000100014084 - DPU           EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
[0xffec9090] lsb 10010000028040c0 - DPU           EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(40));
[0xffec9098] lsb 10010000000040c4 - DPU Unknown
[0xffec9100] lsb 0101ffec91400010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268355860));
[0xffec9108] lsb 01010001000e0014 - PC            EMIT(REG_PC_REGISTER_AMOUNTS, PC_REGISTER_AMOUNTS_RESERVED_0(1) | PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(14));
[0xffec9118] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));


[0xffec9140] lsb 40010000000e6004 - PPU           EMIT(REG_PPU_S_POINTER, PPU_S_POINTER_POINTER_PP_MODE(1) | PPU_S_POINTER_EXECUTER_PP_EN(1) | PPU_S_POINTER_POINTER_PP_EN(1));
[0xffec9148] lsb 80010000000e7004 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
[0xffec9160] lsb 40010000001f6014 - PPU           EMIT(REG_PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec9178] lsb 40010000001f6020 - PPU           EMIT(REG_PPU_DATA_CUBE_OUT_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(31));
[0xffec9180] lsb 4001000000116024 - PPU           EMIT(REG_PPU_OPERATION_MODE_CFG, PPU_OPERATION_MODE_CFG_FLYING_MODE(1) | PPU_OPERATION_MODE_CFG_POOLING_METHOD(1));
[0xffec91b8] lsb 4001ffec82406070 - PPU           EMIT(REG_PPU_DST_BASE_ADDR, PPU_DST_BASE_ADDR_DST_BASE_ADDR(268355620));
[0xffec91c0] lsb 400100000010607c - PPU           EMIT(REG_PPU_DST_SURF_STRIDE, PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
[0xffec91c8] lsb 4001000000106084 - PPU           EMIT(REG_PPU_DATA_FORMAT, PPU_DATA_FORMAT_INDEX_ADD(1));
[0xffec91d0] lsb 40010000000360dc - PPU           EMIT(REG_PPU_MISC_CTRL, PPU_MISC_CTRL_BURST_LEN(3));
[0xffec91e8] lsb 80010000001f7014 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec91f0] lsb 8001ffec8640701c - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0xffec8640);
[0xffec91f8] lsb 8001000000107024 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(1));
[0xffec9200] lsb 8001000000107028 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(1));
[0xffec9208] lsb 8001000000017030 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(1));
[0xffec9228] lsb 0081000000600008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(48));


[0xffec9240] lsb 10010000000e4004 - DPU           EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
[0xffec9248] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec9250] lsb 0201100000501010 - CNA           EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_RESERVED_0(16) | CNA_CONV_CON2_FEATURE_GRAINS(5));
[0xffec9258] lsb 0201000000091014 - CNA           EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
[0xffec9260] lsb 0201000800031020 - CNA           EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(8) | CNA_DATA_SIZE0_DATAIN_HEIGHT(3));
[0xffec9268] lsb 0201000200081024 - CNA           EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(2) | CNA_DATA_SIZE1_DATAIN_CHANNEL(8));
[0xffec9270] lsb 0201000000051028 - CNA           EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(5));
[0xffec9278] lsb 02010000000a102c - CNA           EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(10));
[0xffec9280] lsb 0201000002401030 - CNA           EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000240);
[0xffec9288] lsb 0201000000601034 - CNA           EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(96));
[0xffec9290] lsb 0201030200061038 - CNA           EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(3) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(2) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(6));
[0xffec9298] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffec92a0] lsb 0201000000181044 - CNA           EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(24));
[0xffec92a8] lsb 020100000001104c - CNA           EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_BYPASS(1));
[0xffec92b0] lsb 0201000100001050 - CNA           EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
[0xffec92b8] lsb 0201000100001054 - CNA           EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
[0xffec92c0] lsb 0201000100001058 - CNA           EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
[0xffec92c8] lsb 020100010000105c - CNA           EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
[0xffec92e8] lsb 0201ffec50601070 - CNA           EMIT(REG_CNA_FEATURE_DATA_ADDR, 0xffec5060);
[0xffec92f8] lsb 0201000f000f1078 - CNA           EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
[0xffec9300] lsb 020100000008107c - CNA           EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(8));
[0xffec9308] lsb 0201000000201080 - CNA           EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(32));
[0xffec9310] lsb 0201000700031084 - CNA           EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(7) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(3));
[0xffec9318] lsb 0201000000081088 - CNA           EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
[0xffec9330] lsb 0201ffec80001110 - CNA           EMIT(REG_CNA_DCOMP_ADDR0, 0xffec8000);
[0xffec93b8] lsb 020100000fff1180 - CNA           EMIT(REG_CNA_CVT_CON5, 0x00000fff);
[0xffec93c8] lsb 0801000002003010 - CORE          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2));
[0xffec93d0] lsb 0801000100043014 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(1) | CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(4));
[0xffec93d8] lsb 08010000000f3018 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
[0xffec93e8] lsb 0801000000003030 - CORE Unknown
[0xffec93f0] lsb 1001000001e4400c - DPU           EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
[0xffec93f8] lsb 1001480000024010 - DPU           EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
[0xffec9408] lsb 1001ffec40a04020 - DPU           EMIT(REG_DPU_DST_BASE_ADDR, 0xffec40a0);
[0xffec9410] lsb 1001000001404024 - DPU           EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(20));
[0xffec9418] lsb 1001000000044030 - DPU           EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(4));
[0xffec9420] lsb 1001000000014034 - DPU           EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(1));
[0xffec9430] lsb 10010005000f403c - DPU           EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(5) | DPU_DATA_CUBE_CHANNEL_CHANNEL(15));
[0xffec9438] lsb 1001000000534040 - DPU           EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
[0xffec9458] lsb 1001000001264050 - DPU           EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1) | DPU_BS_OW_CFG_OD_BYPASS(1));
[0xffec9468] lsb 10010000000f4058 - DPU           EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(15));
[0xffec9470] lsb 100100010004405c - DPU           EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(1) | DPU_WDMA_SIZE_1_WIDTH_WDMA(4));
[0xffec9478] lsb 1001000000534060 - DPU           EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
[0xffec9498] lsb 1001000003834070 - DPU           EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
[0xffec94a8] lsb 1001000000014078 - DPU           EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
[0xffec94c0] lsb 1001000100014084 - DPU           EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
[0xffec9510] lsb 10010000028040c0 - DPU           EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(40));
[0xffec9518] lsb 10010000000040c4 - DPU Unknown
[0xffec9580] lsb 0101ffec95c00010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268355932));
[0xffec9588] lsb 01010001000e0014 - PC            EMIT(REG_PC_REGISTER_AMOUNTS, PC_REGISTER_AMOUNTS_RESERVED_0(1) | PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(14));
[0xffec9598] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));


[0xffec95c0] lsb 40010000000e6004 - PPU           EMIT(REG_PPU_S_POINTER, PPU_S_POINTER_POINTER_PP_MODE(1) | PPU_S_POINTER_EXECUTER_PP_EN(1) | PPU_S_POINTER_POINTER_PP_EN(1));
[0xffec95c8] lsb 80010000000e7004 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
[0xffec95e0] lsb 40010000001f6014 - PPU           EMIT(REG_PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec95f8] lsb 40010000001f6020 - PPU           EMIT(REG_PPU_DATA_CUBE_OUT_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(31));
[0xffec9600] lsb 4001000000116024 - PPU           EMIT(REG_PPU_OPERATION_MODE_CFG, PPU_OPERATION_MODE_CFG_FLYING_MODE(1) | PPU_OPERATION_MODE_CFG_POOLING_METHOD(1));
[0xffec9638] lsb 4001ffec82406070 - PPU           EMIT(REG_PPU_DST_BASE_ADDR, PPU_DST_BASE_ADDR_DST_BASE_ADDR(268355620));
[0xffec9640] lsb 400100000010607c - PPU           EMIT(REG_PPU_DST_SURF_STRIDE, PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
[0xffec9648] lsb 4001000000106084 - PPU           EMIT(REG_PPU_DATA_FORMAT, PPU_DATA_FORMAT_INDEX_ADD(1));
[0xffec9650] lsb 40010000000360dc - PPU           EMIT(REG_PPU_MISC_CTRL, PPU_MISC_CTRL_BURST_LEN(3));
[0xffec9668] lsb 80010000001f7014 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec9670] lsb 8001ffec8640701c - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0xffec8640);
[0xffec9678] lsb 8001000000107024 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(1));
[0xffec9680] lsb 8001000000107028 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(1));
[0xffec9688] lsb 8001000000017030 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(1));
[0xffec96a8] lsb 0081000000600008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(48));
[0xffec96c0] lsb 10010000000e4004 - DPU           EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
[0xffec96c8] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec96d0] lsb 0201200000501010 - CNA           EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_RESERVED_0(32) | CNA_CONV_CON2_FEATURE_GRAINS(5));
[0xffec96d8] lsb 0201000000091014 - CNA           EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
[0xffec96e0] lsb 0201000800031020 - CNA           EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(8) | CNA_DATA_SIZE0_DATAIN_HEIGHT(3));
[0xffec96e8] lsb 0201000200081024 - CNA           EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(2) | CNA_DATA_SIZE1_DATAIN_CHANNEL(8));
[0xffec96f0] lsb 0201000000051028 - CNA           EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(5));
[0xffec96f8] lsb 02010000000a102c - CNA           EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(10));
[0xffec9700] lsb 0201000002401030 - CNA           EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000240);
[0xffec9708] lsb 0201000000601034 - CNA           EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(96));
[0xffec9710] lsb 0201030200061038 - CNA           EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(3) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(2) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(6));
[0xffec9718] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffec9720] lsb 0201000000181044 - CNA           EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(24));
[0xffec9728] lsb 020100000001104c - CNA           EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_BYPASS(1));
[0xffec9730] lsb 0201000100001050 - CNA           EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
[0xffec9738] lsb 0201000100001054 - CNA           EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
[0xffec9740] lsb 0201000100001058 - CNA           EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
[0xffec9748] lsb 020100010000105c - CNA           EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
[0xffec9768] lsb 0201ffec50001070 - CNA           EMIT(REG_CNA_FEATURE_DATA_ADDR, 0xffec5000);
[0xffec9778] lsb 0201000f000f1078 - CNA           EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
[0xffec9780] lsb 020100000008107c - CNA           EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(8));
[0xffec9788] lsb 0201000000201080 - CNA           EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(32));
[0xffec9790] lsb 0201000700031084 - CNA           EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(7) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(3));
[0xffec9798] lsb 0201000000081088 - CNA           EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
[0xffec97b0] lsb 0201ffec80001110 - CNA           EMIT(REG_CNA_DCOMP_ADDR0, 0xffec8000);
[0xffec9838] lsb 020100000fff1180 - CNA           EMIT(REG_CNA_CVT_CON5, 0x00000fff);
[0xffec9848] lsb 0801000002003010 - CORE          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2));
[0xffec9850] lsb 0801000100043014 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(1) | CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(4));
[0xffec9858] lsb 08010000000f3018 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
[0xffec9868] lsb 0801000000003030 - CORE Unknown
[0xffec9870] lsb 1001000001e4400c - DPU           EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
[0xffec9878] lsb 1001480000024010 - DPU           EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
[0xffec9888] lsb 1001ffec40004020 - DPU           EMIT(REG_DPU_DST_BASE_ADDR, 0xffec4000);
[0xffec9890] lsb 1001000001404024 - DPU           EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(20));
[0xffec9898] lsb 1001000000044030 - DPU           EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(4));
[0xffec98a0] lsb 1001000000014034 - DPU           EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(1));
[0xffec98b0] lsb 10010005000f403c - DPU           EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(5) | DPU_DATA_CUBE_CHANNEL_CHANNEL(15));
[0xffec98b8] lsb 1001000000534040 - DPU           EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
[0xffec98d8] lsb 1001000001264050 - DPU           EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1) | DPU_BS_OW_CFG_OD_BYPASS(1));
[0xffec98e8] lsb 10010000000f4058 - DPU           EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(15));
[0xffec98f0] lsb 100100010004405c - DPU           EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(1) | DPU_WDMA_SIZE_1_WIDTH_WDMA(4));
[0xffec98f8] lsb 1001000000534060 - DPU           EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
[0xffec9918] lsb 1001000003834070 - DPU           EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
[0xffec9928] lsb 1001000000014078 - DPU           EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
[0xffec9940] lsb 1001000100014084 - DPU           EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
[0xffec9990] lsb 10010000028040c0 - DPU           EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(40));
[0xffec9998] lsb 10010000000040c4 - DPU Unknown
[0xffec9a00] lsb 0101ffec9a400010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268356004));
[0xffec9a08] lsb 01010002000e0014 - PC            EMIT(REG_PC_REGISTER_AMOUNTS, PC_REGISTER_AMOUNTS_RESERVED_0(2) | PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(14));
[0xffec9a18] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffec9a40] lsb 40010000000e6004 - PPU           EMIT(REG_PPU_S_POINTER, PPU_S_POINTER_POINTER_PP_MODE(1) | PPU_S_POINTER_EXECUTER_PP_EN(1) | PPU_S_POINTER_POINTER_PP_EN(1));
[0xffec9a48] lsb 80010000000e7004 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
[0xffec9a60] lsb 40010000001f6014 - PPU           EMIT(REG_PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec9a78] lsb 40010000001f6020 - PPU           EMIT(REG_PPU_DATA_CUBE_OUT_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(31));
[0xffec9a80] lsb 4001000000116024 - PPU           EMIT(REG_PPU_OPERATION_MODE_CFG, PPU_OPERATION_MODE_CFG_FLYING_MODE(1) | PPU_OPERATION_MODE_CFG_POOLING_METHOD(1));
[0xffec9ab8] lsb 4001ffec82406070 - PPU           EMIT(REG_PPU_DST_BASE_ADDR, PPU_DST_BASE_ADDR_DST_BASE_ADDR(268355620));
[0xffec9ac0] lsb 400100000010607c - PPU           EMIT(REG_PPU_DST_SURF_STRIDE, PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
[0xffec9ac8] lsb 4001000000106084 - PPU           EMIT(REG_PPU_DATA_FORMAT, PPU_DATA_FORMAT_INDEX_ADD(1));
[0xffec9ad0] lsb 40010000000360dc - PPU           EMIT(REG_PPU_MISC_CTRL, PPU_MISC_CTRL_BURST_LEN(3));
[0xffec9ae8] lsb 80010000001f7014 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec9af0] lsb 8001ffec8640701c - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0xffec8640);
[0xffec9af8] lsb 8001000000107024 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(1));
[0xffec9b00] lsb 8001000000107028 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(1));
[0xffec9b08] lsb 8001000000017030 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(1));
[0xffec9b28] lsb 0081000000600008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(48));
[0xffec9b40] lsb 10010000000e4004 - DPU           EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
[0xffec9b48] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec9b50] lsb 0201200000401010 - CNA           EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_RESERVED_0(32) | CNA_CONV_CON2_FEATURE_GRAINS(4));
[0xffec9b58] lsb 0201000000091014 - CNA           EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
[0xffec9b60] lsb 0201000800021020 - CNA           EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(8) | CNA_DATA_SIZE0_DATAIN_HEIGHT(2));
[0xffec9b68] lsb 0201000200081024 - CNA           EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(2) | CNA_DATA_SIZE1_DATAIN_CHANNEL(8));
[0xffec9b70] lsb 0201000000051028 - CNA           EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(5));
[0xffec9b78] lsb 020100000005102c - CNA           EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(5));
[0xffec9b80] lsb 0201000002401030 - CNA           EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000240);
[0xffec9b88] lsb 0201000000601034 - CNA           EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(96));
[0xffec9b90] lsb 0201030200061038 - CNA           EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(3) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(2) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(6));
[0xffec9b98] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffec9ba0] lsb 0201000000101044 - CNA           EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(16));
[0xffec9ba8] lsb 020100000001104c - CNA           EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_BYPASS(1));
[0xffec9bb0] lsb 0201000100001050 - CNA           EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
[0xffec9bb8] lsb 0201000100001054 - CNA           EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
[0xffec9bc0] lsb 0201000100001058 - CNA           EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
[0xffec9bc8] lsb 020100010000105c - CNA           EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
[0xffec9be8] lsb 0201ffec50601070 - CNA           EMIT(REG_CNA_FEATURE_DATA_ADDR, 0xffec5060);
[0xffec9bf8] lsb 0201000f000f1078 - CNA           EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
[0xffec9c00] lsb 020100000008107c - CNA           EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(8));
[0xffec9c08] lsb 0201000000201080 - CNA           EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(32));
[0xffec9c10] lsb 0201000700021084 - CNA           EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(7) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(2));
[0xffec9c18] lsb 0201000000081088 - CNA           EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
[0xffec9c30] lsb 0201ffec80001110 - CNA           EMIT(REG_CNA_DCOMP_ADDR0, 0xffec8000);
[0xffec9cb8] lsb 020100000fff1180 - CNA           EMIT(REG_CNA_CVT_CON5, 0x00000fff);
[0xffec9cc8] lsb 0801000002003010 - CORE          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2));
[0xffec9cd0] lsb 0801000000043014 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(4));
[0xffec9cd8] lsb 08010000000f3018 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
[0xffec9ce8] lsb 0801000000003030 - CORE Unknown
[0xffec9cf0] lsb 1001000001e4400c - DPU           EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
[0xffec9cf8] lsb 1001480000024010 - DPU           EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
[0xffec9d08] lsb 1001ffec40a04020 - DPU           EMIT(REG_DPU_DST_BASE_ADDR, 0xffec40a0);
[0xffec9d10] lsb 1001000001404024 - DPU           EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(20));
[0xffec9d18] lsb 1001000000044030 - DPU           EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(4));
[0xffec9d30] lsb 10010005000f403c - DPU           EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(5) | DPU_DATA_CUBE_CHANNEL_CHANNEL(15));
[0xffec9d38] lsb 1001000000534040 - DPU           EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
[0xffec9d58] lsb 1001000001264050 - DPU           EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1) | DPU_BS_OW_CFG_OD_BYPASS(1));
[0xffec9d68] lsb 10010000000f4058 - DPU           EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(15));
[0xffec9d70] lsb 100100000004405c - DPU           EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_WIDTH_WDMA(4));
[0xffec9d78] lsb 1001000000534060 - DPU           EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
[0xffec9d98] lsb 1001000003834070 - DPU           EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
[0xffec9da8] lsb 1001000000014078 - DPU           EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
[0xffec9dc0] lsb 1001000100014084 - DPU           EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
[0xffec9e10] lsb 10010000028040c0 - DPU           EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(40));
[0xffec9e18] lsb 10010000000040c4 - DPU Unknown
[0xffec9e80] lsb 0101ffec9ec00010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268356076));
[0xffec9e88] lsb 01010002000e0014 - PC            EMIT(REG_PC_REGISTER_AMOUNTS, PC_REGISTER_AMOUNTS_RESERVED_0(2) | PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(14));
[0xffec9e98] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffec9ec0] lsb 40010000000e6004 - PPU           EMIT(REG_PPU_S_POINTER, PPU_S_POINTER_POINTER_PP_MODE(1) | PPU_S_POINTER_EXECUTER_PP_EN(1) | PPU_S_POINTER_POINTER_PP_EN(1));
[0xffec9ec8] lsb 80010000000e7004 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
[0xffec9ee0] lsb 40010000001f6014 - PPU           EMIT(REG_PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec9ef8] lsb 40010000001f6020 - PPU           EMIT(REG_PPU_DATA_CUBE_OUT_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(31));
[0xffec9f00] lsb 4001000000116024 - PPU           EMIT(REG_PPU_OPERATION_MODE_CFG, PPU_OPERATION_MODE_CFG_FLYING_MODE(1) | PPU_OPERATION_MODE_CFG_POOLING_METHOD(1));
[0xffec9f38] lsb 4001ffec82406070 - PPU           EMIT(REG_PPU_DST_BASE_ADDR, PPU_DST_BASE_ADDR_DST_BASE_ADDR(268355620));
[0xffec9f40] lsb 400100000010607c - PPU           EMIT(REG_PPU_DST_SURF_STRIDE, PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
[0xffec9f48] lsb 4001000000106084 - PPU           EMIT(REG_PPU_DATA_FORMAT, PPU_DATA_FORMAT_INDEX_ADD(1));
[0xffec9f50] lsb 40010000000360dc - PPU           EMIT(REG_PPU_MISC_CTRL, PPU_MISC_CTRL_BURST_LEN(3));
[0xffec9f68] lsb 80010000001f7014 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffec9f70] lsb 8001ffec8640701c - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0xffec8640);
[0xffec9f78] lsb 8001000000107024 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(1));
[0xffec9f80] lsb 8001000000107028 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(1));
[0xffec9f88] lsb 8001000000017030 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(1));
[0xffec9fa8] lsb 0081000000600008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(48));
[0xffec9fc0] lsb 10010000000e4004 - DPU           EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
[0xffec9fc8] lsb 02016000a120100c - CNA           EMIT(REG_CNA_CONV_CON1, CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(10) | CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2));
[0xffec9fd0] lsb 0201200000401010 - CNA           EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_RESERVED_0(32) | CNA_CONV_CON2_FEATURE_GRAINS(4));
[0xffec9fd8] lsb 0201000000091014 - CNA           EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_Y_STRIDE(1) | CNA_CONV_CON3_CONV_X_STRIDE(1));
[0xffec9fe0] lsb 0201000800021020 - CNA           EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(8) | CNA_DATA_SIZE0_DATAIN_HEIGHT(2));
[0xffec9fe8] lsb 0201000200081024 - CNA           EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(2) | CNA_DATA_SIZE1_DATAIN_CHANNEL(8));
[0xffec9ff0] lsb 0201000000051028 - CNA           EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(5));
[0xffec9ff8] lsb 020100000005102c - CNA           EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(5));
[0xffeca000] lsb 0201000002401030 - CNA           EMIT(REG_CNA_WEIGHT_SIZE0, 0x00000240);
[0xffeca008] lsb 0201000000601034 - CNA           EMIT(REG_CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(96));
[0xffeca010] lsb 0201030200061038 - CNA           EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(3) | CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(2) | CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(6));
[0xffeca018] lsb 0201000000b11040 - CNA           EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(11) | CNA_CBUF_CON0_DATA_BANK(1));
[0xffeca020] lsb 0201000000101044 - CNA           EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(16));
[0xffeca028] lsb 020100000001104c - CNA           EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_BYPASS(1));
[0xffeca030] lsb 0201000100001050 - CNA           EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
[0xffeca038] lsb 0201000100001054 - CNA           EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
[0xffeca040] lsb 0201000100001058 - CNA           EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
[0xffeca048] lsb 020100010000105c - CNA           EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
[0xffeca068] lsb 0201ffec50901070 - CNA           EMIT(REG_CNA_FEATURE_DATA_ADDR, 0xffec5090);
[0xffeca078] lsb 0201000f000f1078 - CNA           EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
[0xffeca080] lsb 020100000008107c - CNA           EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(8));
[0xffeca088] lsb 0201000000201080 - CNA           EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(32));
[0xffeca090] lsb 0201000700021084 - CNA           EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(7) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(2));
[0xffeca098] lsb 0201000000081088 - CNA           EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
[0xffeca0b0] lsb 0201ffec80001110 - CNA           EMIT(REG_CNA_DCOMP_ADDR0, 0xffec8000);
[0xffeca138] lsb 020100000fff1180 - CNA           EMIT(REG_CNA_CVT_CON5, 0x00000fff);
[0xffeca148] lsb 0801000002003010 - CORE          EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_PROC_PRECISION(2));
[0xffeca150] lsb 0801000000043014 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(4));
[0xffeca158] lsb 08010000000f3018 - CORE          EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
[0xffeca168] lsb 0801000000003030 - CORE Unknown
[0xffeca170] lsb 1001000001e4400c - DPU           EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
[0xffeca178] lsb 1001480000024010 - DPU           EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(2) | DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_PROC_PRECISION(2));
[0xffeca188] lsb 1001ffec40f04020 - DPU           EMIT(REG_DPU_DST_BASE_ADDR, 0xffec40f0);
[0xffeca190] lsb 1001000001404024 - DPU           EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(20));
[0xffeca198] lsb 1001000000044030 - DPU           EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(4));
[0xffeca1b0] lsb 10010005000f403c - DPU           EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(5) | DPU_DATA_CUBE_CHANNEL_CHANNEL(15));
[0xffeca1b8] lsb 1001000000534040 - DPU           EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1));
[0xffeca1d8] lsb 1001000001264050 - DPU           EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1) | DPU_BS_OW_CFG_OD_BYPASS(1));
[0xffeca1e8] lsb 10010000000f4058 - DPU           EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(15));
[0xffeca1f0] lsb 100100000004405c - DPU           EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_WIDTH_WDMA(4));
[0xffeca1f8] lsb 1001000000534060 - DPU           EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) | DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
[0xffeca218] lsb 1001000003834070 - DPU           EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) | DPU_EW_CFG_EW_BYPASS(1));
[0xffeca228] lsb 1001000000014078 - DPU           EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
[0xffeca240] lsb 1001000100014084 - DPU           EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) | DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
[0xffeca290] lsb 10010000028040c0 - DPU           EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(40));
[0xffeca298] lsb 10010000000040c4 - DPU Unknown
[0xffeca300] lsb 0101ffeca3400010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268356148));
[0xffeca308] lsb 01010002000e0014 - PC            EMIT(REG_PC_REGISTER_AMOUNTS, PC_REGISTER_AMOUNTS_RESERVED_0(2) | PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(14));
[0xffeca318] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffeca340] lsb 40010000000e6004 - PPU           EMIT(REG_PPU_S_POINTER, PPU_S_POINTER_POINTER_PP_MODE(1) | PPU_S_POINTER_EXECUTER_PP_EN(1) | PPU_S_POINTER_POINTER_PP_EN(1));
[0xffeca348] lsb 80010000000e7004 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) | PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) | PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
[0xffeca360] lsb 40010000001f6014 - PPU           EMIT(REG_PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffeca378] lsb 40010000001f6020 - PPU           EMIT(REG_PPU_DATA_CUBE_OUT_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(31));
[0xffeca380] lsb 4001000000116024 - PPU           EMIT(REG_PPU_OPERATION_MODE_CFG, PPU_OPERATION_MODE_CFG_FLYING_MODE(1) | PPU_OPERATION_MODE_CFG_POOLING_METHOD(1));
[0xffeca3b8] lsb 4001ffec82406070 - PPU           EMIT(REG_PPU_DST_BASE_ADDR, PPU_DST_BASE_ADDR_DST_BASE_ADDR(268355620));
[0xffeca3c0] lsb 400100000010607c - PPU           EMIT(REG_PPU_DST_SURF_STRIDE, PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(1));
[0xffeca3c8] lsb 4001000000106084 - PPU           EMIT(REG_PPU_DATA_FORMAT, PPU_DATA_FORMAT_INDEX_ADD(1));
[0xffeca3d0] lsb 40010000000360dc - PPU           EMIT(REG_PPU_MISC_CTRL, PPU_MISC_CTRL_BURST_LEN(3));
[0xffeca3e8] lsb 80010000001f7014 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(31));
[0xffeca3f0] lsb 8001ffec8640701c - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0xffec8640);
[0xffeca3f8] lsb 8001000000107024 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(1));
[0xffeca400] lsb 8001000000107028 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(1));
[0xffeca408] lsb 8001000000017030 - PPU_RDMA      EMIT(REG_PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(1));
[0xffeca428] lsb 0081000000600008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(48));
[0xffeca440] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca448] lsb 000003000000000d - noone Unknown
[0xffeca450] lsb 000000000001ffff - noone Unknown
[0xffeca458] lsb 000000000000006c - noone Unknown
[0xffeca468] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca470] lsb 000003000000000d - noone Unknown
[0xffeca478] lsb 000000000001ffff - noone Unknown
[0xffeca480] lsb 0000000000000068 - noone Unknown
[0xffeca488] lsb 0000000000000380 - noone Unknown
[0xffeca490] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca498] lsb 00000c0000000060 - noone Unknown
[0xffeca4a0] lsb 000000000001ffff - noone Unknown
[0xffeca4a8] lsb 000000000000001a - noone Unknown
[0xffeca4b0] lsb 0000000000000700 - noone Unknown
[0xffeca4b8] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca4c0] lsb 000003000000000d - noone Unknown
[0xffeca4c8] lsb 000000000001ffff - noone Unknown
[0xffeca4d0] lsb 0000000000000068 - noone Unknown
[0xffeca4d8] lsb 0000000000000800 - noone Unknown
[0xffeca4e0] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca4e8] lsb 00000c0000000060 - noone Unknown
[0xffeca4f0] lsb 000000000001ffff - noone Unknown
[0xffeca4f8] lsb 000000000000001a - noone Unknown
[0xffeca500] lsb 0000000000000b80 - noone Unknown
[0xffeca508] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca510] lsb 000003000000000d - noone Unknown
[0xffeca518] lsb 000000000001ffff - noone Unknown
[0xffeca520] lsb 0000000000000068 - noone Unknown
[0xffeca528] lsb 0000000000000c80 - noone Unknown
[0xffeca530] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca538] lsb 00000c0000000060 - noone Unknown
[0xffeca540] lsb 000000000001ffff - noone Unknown
[0xffeca548] lsb 000000000000001a - noone Unknown
[0xffeca558] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca560] lsb 000003000000000d - noone Unknown
[0xffeca568] lsb 000000000001ffff - noone Unknown
[0xffeca570] lsb 0000000000000068 - noone Unknown
[0xffeca580] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca588] lsb 00000c0000000060 - noone Unknown
[0xffeca590] lsb 000000000001ffff - noone Unknown
[0xffeca598] lsb 000000000000001a - noone Unknown
[0xffeca5a0] lsb 0000000000001480 - noone Unknown
[0xffeca5a8] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca5b0] lsb 000003000000000d - noone Unknown
[0xffeca5b8] lsb 000000000001ffff - noone Unknown
[0xffeca5c0] lsb 0000000000000068 - noone Unknown
[0xffeca5c8] lsb 0000000000001580 - noone Unknown
[0xffeca5d0] lsb 0000000100000000 - noone         EMIT(REG_PC_VERSION, 0x00010000);
[0xffeca5d8] lsb 00000c0000000060 - noone Unknown
[0xffeca5e0] lsb 000000000001ffff - noone Unknown
[0xffeca5e8] lsb 000000000000001a - noone Unknown
[0xffeca5f0] lsb 0000000000001900 - noone Unknown
Dumped 1013 register commands to dump/gem2_regdump.bin
[0xffec8db8] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffec9118] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffec9598] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffec9a18] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffec9e98] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
[0xffeca318] lsb 00810000000d0008 - noone         EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(6) | PC_OPERATION_ENABLE_OP_EN(1));
6
[0xffec9100] lsb 0101ffec91400010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268355860));
[0xffec9580] lsb 0101ffec95c00010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268355932));
[0xffec9a00] lsb 0101ffec9a400010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268356004));
[0xffec9e80] lsb 0101ffec9ec00010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268356076));
[0xffeca300] lsb 0101ffeca3400010 - PC            EMIT(REG_PC_BASE_ADDRESS, PC_BASE_ADDRESS_PC_SOURCE_ADDR(268356148));
5
drm name is rknpu - 20231018 - RKNPU driver
du is 
Dumping specified GEM objects...

=== Processing GEM 3 ===

==================================================
Processing GEM Flink 3
==================================================
gem flink 3: ret=0 handle=1 size=4096
DRM_IOCTL_RKNPU_MEM_MAP returned 0 0x100004000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294983680>

==================================================
Processing GEM Flink 3 for Register Decode
==================================================
Successfully opened GEM via flink 3
memmap returned 0 0x100004000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294983680> 4294983680
DEBUG: Found 199 registers in XML
DEBUG: Loaded 199 register definitions
No register commands found in this GEM; skipping decode.
drm name is rknpu - 20231018 - RKNPU driver
du is 
Dumping specified GEM objects...

=== Processing GEM 4 ===

==================================================
Processing GEM Flink 4
==================================================
gem flink 4: ret=0 handle=1 size=4096
DRM_IOCTL_RKNPU_MEM_MAP returned 0 0x100005000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294987776>

==================================================
Processing GEM Flink 4 for Register Decode
==================================================
Successfully opened GEM via flink 4
memmap returned 0 0x100005000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294987776> 4294987776
DEBUG: Found 199 registers in XML
DEBUG: Loaded 199 register definitions
No register commands found in this GEM; skipping decode.
drm name is rknpu - 20231018 - RKNPU driver
du is 
Dumping specified GEM objects...

=== Processing GEM 5 ===

==================================================
Processing GEM Flink 5
==================================================
gem flink 5: ret=0 handle=1 size=4096
DRM_IOCTL_RKNPU_MEM_MAP returned 0 0x100006000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294991872>

==================================================
Processing GEM Flink 5 for Register Decode
==================================================
Successfully opened GEM via flink 5
memmap returned 0 0x100006000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294991872> 4294991872
DEBUG: Found 199 registers in XML
DEBUG: Loaded 199 register definitions
DEBUG: base selection for GEM 5: dma_base=None phys_base=0x100006000 task_base=None chosen=0x100006000 commands=30
Successfully created dump/gem5_regdump.bin
[0x100006000] lsb 3cf63e853c7b38e1 - CORE Unknown
DEBUG: Looking for offset 0x38e1, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0x100006008] lsb 447b4094430a40c3 - PPU Unknown
DEBUG: Looking for offset 0x40c3, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0x100006010] lsb 498a476c438546ac - PC Unknown
DEBUG: Looking for offset 0x46ac, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0x100006018] lsb 47f64c30499a45a6 - PC Unknown
DEBUG: Looking for offset 0x45a6, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0x100006020] lsb 4d5249584dec4be9 - PC Unknown
DEBUG: Looking for offset 0x4be9, available offsets: [0, 4, 8, 16, 20, 32, 36, 40, 44, 48]
[0x100006028] lsb 0000000000004ff8 - noone Unknown
[0x100006030] lsb 4048445441d73e0a - PPU Unknown
[0x100006038] lsb 464842614638442e - CNA Unknown
[0x100006040] lsb 4ae4489c44a948af - CNA Unknown
[0x100006048] lsb 486e4cdd4a80468d - CORE Unknown
[0x100006050] lsb 4dc549cc4e984c68 - PC Unknown
[0x100006058] lsb 0000000000005052 - noone Unknown
[0x100006060] lsb 4348486a45ec4205 - PC Unknown
[0x100006068] lsb 48a444b1495c472e - CORE Unknown
[0x100006070] lsb 4c924a1c46294aef - CORE Unknown
[0x100006078] lsb 492e4dfd4c004806 - PC Unknown
[0x100006080] lsb 4e854a8c4fb84d28 - CNA Unknown
[0x100006088] lsb 00000000000050e2 - noone Unknown
[0x100006090] lsb 45bd4b91490f451c - PC Unknown
[0x100006098] lsb 4abd46ca4c4149b1 - CNA Unknown
[0x1000060a0] lsb 4e254c1b48214d0b - CNA Unknown
[0x1000060a8] lsb 4a3b4f904d0d4913 - CNA Unknown
[0x1000060b0] lsb 4f924b9850a64e34 - PC Unknown
[0x1000060b8] lsb 00000000000051ac - noone Unknown
[0x1000060c0] lsb 48384dcf4bc347cf - CORE Unknown
[0x1000060c8] lsb 4cb848bf4e484c32 - CORE Unknown
[0x1000060d0] lsb 50164d74497b4f11 - DPU Unknown
[0x1000060d8] lsb 4b9450cb4e664a6d - PC Unknown
[0x1000060e0] lsb 50764c7951a94f8e - DPU Unknown
[0x1000060e8] lsb 00000000000052af - noone Unknown
Dumped 30 register commands to dump/gem5_regdump.bin
drm name is rknpu - 20231018 - RKNPU driver
du is 
Dumping specified GEM objects...

=== Processing GEM 6 ===

==================================================
Processing GEM Flink 6
==================================================
gem flink 6: ret=0 handle=1 size=4096
DRM_IOCTL_RKNPU_MEM_MAP returned 0 0x100007000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294995968>

==================================================
Processing GEM Flink 6 for Register Decode
==================================================
Successfully opened GEM via flink 6
memmap returned 0 0x100007000
mmap returned <mmap.mmap closed=False, access=ACCESS_DEFAULT, length=4096, pos=0, offset=4294995968> 4294995968
DEBUG: Found 199 registers in XML
DEBUG: Loaded 199 register definitions
No register commands found in this GEM; skipping decode.
============
D RKNN: [22:45:08.792] allocated memory, name: input, virt addr: 0x7ff7ff0000, dma addr: 0xffec5000, obj addr: 0xffffff8274763000, size: 240, aligned size: 4096, fd: 8, handle: 5, flags: 0x403, gem name: 5, iommu domain id: 0
D RKNN: [22:45:08.792] allocated memory, name: output, virt addr: 0x7ff7fef000, dma addr: 0xffec4000, obj addr: 0xffffff8274760400, size: 640, aligned size: 4096, fd: 9, handle: 6, flags: 0x403, gem name: 6, iommu domain id: 0
D RKNN: [22:45:08.792] enable argb mode, dtype: FLOAT16, channel: 3
D RKNN: [22:45:08.792] update cvt sign: 0
D RKNN: [22:45:10.429] Output(output): size_with_stride larger than model origin size, if need run OutputOperator in NPU, please call rknn_create_memory using size_with_stride.
  Output shape: 1x6x4x5

  Expected Output (CPU computed):
    Output Channel 0:
      6.29932 8.82031 11.3477 13.8574 16.3633 
      6.29883 8.82227 11.3379 13.8555 16.3789 
      6.29883 8.82031 11.3438 13.8633 16.3828 
      6.28906 8.81641 11.3359 13.8516 16.3984 
    Output Channel 1:
      3.78271 6.29688 8.81641 11.3535 13.8672 
      3.7793 6.30078 8.82617 11.3398 13.8555 
      3.7793 6.30078 8.81641 11.3398 13.8516 
      3.78906 6.29297 8.83594 11.3359 13.8516 
    Output Channel 2:
      -10.082 -15.1172 -20.1641 -25.2109 -30.2305 
      -10.0781 -15.123 -20.1641 -25.1953 -30.2344 
      -10.0781 -15.1211 -20.1602 -25.2031 -30.2344 
      -10.0781 -15.1094 -20.1719 -25.1875 -30.25 
    Output Channel 3:
      6.29932 8.82031 11.3477 13.8574 16.3633 
      6.29883 8.82227 11.3379 13.8555 16.3789 
      6.29883 8.82031 11.3438 13.8633 16.3828 
      6.28906 8.81641 11.3359 13.8516 16.3984 
    Output Channel 4:
      3.78271 6.29688 8.81641 11.3535 13.8672 
      3.7793 6.30078 8.82617 11.3398 13.8555 
      3.7793 6.30078 8.81641 11.3398 13.8516 
      3.78906 6.29297 8.83594 11.3359 13.8516 
    Output Channel 5:
      -10.082 -15.1172 -20.1641 -25.2109 -30.2305 
      -10.0781 -15.123 -20.1641 -25.1953 -30.2344 
      -10.0781 -15.1211 -20.1602 -25.2031 -30.2344 
      -10.0781 -15.1094 -20.1719 -25.1875 -30.25 

  Actual Output (RKNN):
    Output Channel 0:
      6.30078 8.82031 11.3438 13.8594 16.3594 
      6.29688 8.82031 11.3359 13.8594 16.375 
      6.29688 8.82031 11.3438 13.8594 16.375 
      6.28906 8.8125 11.3359 13.8516 16.4062 
    Output Channel 1:
      3.7832 6.29688 8.8125 11.3516 13.8672 
      3.7793 6.30078 8.82812 11.3438 13.8594 
      3.7793 6.30078 8.8125 11.3438 13.8516 
      3.78906 6.29297 8.83594 11.3359 13.8516 
    Output Channel 2:
      -10.0781 -15.1172 -20.1562 -25.2188 -30.2344 
      -10.0781 -15.125 -20.1562 -25.1875 -30.2344 
      -10.0781 -15.125 -20.1562 -25.2031 -30.2344 
      -10.0781 -15.1094 -20.1719 -25.1875 -30.25 
    Output Channel 3:
      6.30078 8.82031 11.3438 13.8594 16.3594 
      6.29688 8.82031 11.3359 13.8594 16.375 
      6.29688 8.82031 11.3438 13.8594 16.375 
      6.28906 8.8125 11.3359 13.8516 16.4062 
    Output Channel 4:
      3.7832 6.29688 8.8125 11.3516 13.8672 
      3.7793 6.30078 8.82812 11.3438 13.8594 
      3.7793 6.30078 8.8125 11.3438 13.8516 
      3.78906 6.29297 8.83594 11.3359 13.8516 
    Output Channel 5:
      -10.0781 -15.1172 -20.1562 -25.2188 -30.2344 
      -10.0781 -15.125 -20.1562 -25.1875 -30.2344 
      -10.0781 -15.125 -20.1562 -25.2031 -30.2344 
      -10.0781 -15.1094 -20.1719 -25.1875 -30.25 
D RKNN: [22:45:10.431] free memory, name: external, virt addr: 0x7ff7ff1000, dma addr: 0xffecf000, obj addr: 0xffffff8274765800, size: 640, aligned size: 4096, fd: 7, handle: 4, flags: 0x403, gem name: 4, iommu domain id: 0
D RKNN: [22:45:10.431] free memory, name: internal, virt addr: 0x7ff7ff2000, dma addr: 0xffec7000, obj addr: 0xffffff8274766c00, size: 896, aligned size: 4096, fd: 6, handle: 3, flags: 0x403, gem name: 3, iommu domain id: 0
D RKNN: [22:45:10.431] free memory, name: weight, virt addr: 0x7ff7ff3000, dma addr: 0xffec8000, obj addr: 0xffffff8274763c00, size: 9728, aligned size: 12288, fd: 5, handle: 2, flags: 0x403, gem name: 2, iommu domain id: 0
D RKNN: [22:45:10.432] free memory, name: task, virt addr: 0x7ff7ff6000, dma addr: 0xffec6000, obj addr: 0xffffff8274765400, size: 440, aligned size: 4096, fd: 4, handle: 1, flags: 0x40b, gem name: 1, iommu domain id: 0
D RKNN: [22:45:10.432] free memory, name: input, virt addr: 0x7ff7ff0000, dma addr: 0xffec5000, obj addr: 0xffffff8274763000, size: 240, aligned size: 4096, fd: 8, handle: 5, flags: 0x403, gem name: 5, iommu domain id: 0
D RKNN: [22:45:10.432] free memory, name: output, virt addr: 0x7ff7fef000, dma addr: 0xffec4000, obj addr: 0xffffff8274760400, size: 640, aligned size: 4096, fd: 9, handle: 6, flags: 0x403, gem name: 6, iommu domain id: 0

  ✓ PASSED (max error: 0.0078125)

################################################################################
TEST SUMMARY
################################################################################
Total tests: 1
Passed: 1
Failed: 0
################################################################################
[Inferior 1 (process 416462) exited normally]