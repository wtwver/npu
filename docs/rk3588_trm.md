RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

7

RW  0x0

6

RW  0x0

5

RW  0x0

4

RW  0x0

3

RW  0x0

2

RW  0x0

1

RW  0x0

0

RW  0x0

WRCEIEN
AXI write channel error interrupt enable.
1'b0: Disable
1'b1: Enable
DISEIEN
INFLATE huffman distance error interrupt enable.
1'b0: Disable
1'b1: Enable
LENEIEN
INFLATE huffman length error interrupt enable.
1'b0: Disable
1'b1: Enable
LITEIEN
INFLATE huffman literal error interrupt enable.
1'b0: Disable
1'b1: Enable
SQMEIEN
INFLATE SQ match error interrupt enable.
1'b0: Disable
1'b1: Enable
SLCIEN
INFLATE store block length check error interrupt enable.
1'b0: Disable
1'b1: Enable
HDEIEN
INFLATE file header error interrupt enable.
1'b0: Disable
1'b1: Enable
DSIEN
DECOM stop interrupt enable.
1'b0: Disable
1'b1: Enable

DECOM_AXI_STAT
Address: Operational Base + offset (0x0030)

Bit  Attr  Reset Value

31:5  RO  0x0000000

4

RO  0x1

3:2  RO  0x0

1:0  RO  0x0

Description

reserved
AXI_IDLE
AXI master idle state register.
1'b0: AXI master is busy
1'b1: AXI master is idle
When the decompression is aborted, DECOM will wait for AXI
master's current read/write transfer to complete, that is, wait for
AXI_IDLE=1, then reset DECOM, otherwise the uncompleted AXI
transmission will cause AXI bus exception.
It should be noted that after the decompression is abnormal, the
AXI master's unfinished write operation will continue, but the
output value of wstrb[15:0] will become 16'b0.
RRESP
AXI read channel response state.
BRESP
AXI write channel response state.

DECOM_TSIZEL
Address: Operational Base + offset (0x0034)

Copyright 2022 © Rockchip Electronics Co., Ltd.

1980

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:0  RO  0x00000000

TSIZEL
The total size of the data after decompression.
TSIZE[63:0]. (Unit:byte)

DECOM_TSIZEH
Address: Operational Base + offset (0x0038)

Bit  Attr  Reset Value
31:0  RO  0x00000000  TSIZEH

Description

The total size of the data after decompression.

DECOM_MGNUM
Address: Operational Base + offset (0x003C)

Bit  Attr  Reset Value

Description

31:0  RO  0x00000000

MGNUM
Magic number of LZ4 is 0x184D2204.
If the MGNUM is not 0x184D2204, a magic number error(mn_err)
will be generated. And DECOM will stop decompressing.

DECOM_FRAME
Address: Operational Base + offset (0x0040)

Bit  Attr  Reset Value

31:24 RO  0x00

23:16 RO  0x00

15:8  RO  0x00

7:6  RO  0x0

5

RO  0x0

Description

reserved
HCC
Header checksum byte.
One-byte checksum of combined descriptor fields, including
optional ones. The value is the second byte of xxh32() using zero
as a see, and the full frame descriptor as an input (including
optional fields when they are present). A wrong checksum
indicates an error in the descriptor. Header checksum is
informational and can be skipped.
The header checksum check function can be enabled by
controlling the HCEN enable signal in the CTRL register. After
being turned on, DECOM calculates the hash32 check value based
on the data of the frame descriptor part of the LZ4 compressed
file, and performs the header checksum byte in the file. In
contrast, if they are not equal, an HCC error will be generated
and an interrupt will be generated.
BD
BD byte.
Including Block Maximum Size information. This information is
useful to help the decoder allocate memory. Size here refers to
the original(uncompressed) data size.
VN_NUM
Version number.
2-bits field, must be set to 2'b01. Any other value cannot be
decoded by this version of the specification. If the version
number is error, Version number error(VNE) will be generated.
BCC_FLG
Block checksum flag.
If this flag is set, each data block will be followed by a 4-bytes
checksum, calculated by using the xxHash-32 algorithm on the
raw (compressed) data block. The intention is to detect data
corruption (storage or transmission errors) immediately, before
decoding. Block checksum usage is optional.

Copyright 2022 © Rockchip Electronics Co., Ltd.

1981

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

4

RO  0x0

3

RO  0x0

2

1

RO  0x0

RO  0x0

0

RO  0x0

BIND_FLG
Block independence flag.
If this flag is set to "1", blocks are independent. If this flag is set
to "0", each block depends on previous ones (up to LZ4 window
size, which is 64 KB). In such case, it's necessary to decode all
blocks in sequence. Block dependency improves compression
ratio, especially for small blocks. On the other hand, it makes
random access or multi-threaded decoding impossible. For
debugging.
CS_FLG
Content size flag.
If this flag is set, the uncompressed size of data included within
the frame will be present as an 8 bytes unsigned little endian
value, after the flags. Content size usage is optional.
CCC_FLG
Content checksum flag.
If this flag is set, a 32-bits content checksum will be appended
after the EndMark.
reserved
DICT_FLG
Dictionary ID flag.
If this flag is set, a 4-bytes Dict-ID field will be present, after the
descriptor flags and the Content Size. If this flag is set, Dictionary
ID error will be generated. DECOM will stop decompressing.

DECOM_DICTID
Address: Operational Base + offset (0x0044)

Bit  Attr  Reset Value

Description

31:0  RO  0x00000000

DICTID
Dictionary ID in the compressed file header.
Dictionary ID is only present if the DID_FLG is set. It works as a
kind of "known prefix" which is used by both the compressor and
the decompressor to "warm-up" reference tables.

DECOM_CSL
Address: Operational Base + offset (0x0048)

Bit  Attr  Reset Value

Description

31:0  RO  0x00000000

CSL
Content size(CS) lower 32-bits. (Unit:byte)
CS[63:0] is the original(uncompressed) size. This information is
optional in LZ4 file header, and only present if CS_FLG is set.
CS[63:0]={CSH, CSL};

DECOM_CSH
Address: Operational Base + offset (0x004C)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RO  0x0

CSH
Content size(CS) upper 32-bits.

DECOM_LMTSL
Address: Operational Base + offset (0x0050)

Copyright 2022 © Rockchip Electronics Co., Ltd.

1982

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:0  RW  0xffffffff

LMTSL
Limit size of decompressed data lower 32bits. (Unit:byte)
When the amount of decompressed data is greater than the
LMTS, DSOLI interrupt is generated and the decompression
process is stopped.

DECOM_LMTSH
Address: Operational Base + offset (0x0054)

Bit  Attr  Reset Value

31:0  RW  0xffffffff

LMTSH
Limit size of decompressed data higher 32bits.

Description

DECOM_GZFHD
Address: Operational Base + offset (0x0058)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RO  0x0

GZFHD
GZIP/ZLIB file header information.
When GZIPM = 1, GZFHD is GZIP file header information.
When ZLIBM = 1, GZFHD is ZLIB file header information.

DECOM_VERSION
Address: Operational Base + offset (0x00F0)

Bit  Attr  Reset Value

31:0  RO  0x00000926

VERSION
Version number = 32'h0000_0926.

Description

35.5 Application Notes

⚫  After configuring the decompression mode and other decompression parameters,

configure the decompression enable register ENR

⚫  When the decompression is not completed, setting ENR from 1 to 0 will force the
decompression process to stop and DECOM will return to the IDLE state. Need to
reconfigure and enable DECOM to start the next decompression

⚫  When an error is encountered in the decompression, DECOM will immediately stop the

decompression process, set the relevant status register, and generate the corresponding
interrupt

⚫  After the decompression process is stopped, ENR will automatically change to 0
⚫  After the decompression process is stopped, DSI will be set to 1. If DSIEN is 1, an

interrupt will be generated; if DSIEN is 0, no interrupt will be generated

⚫  After the decompression process is stopped, COMPLETE (STAT[0]) is not set to 1 if there

is an error in the decompression process; COMPLETE is set to 1 only when the
decompression is complete and there are no errors

Copyright 2022 © Rockchip Electronics Co., Ltd.

1983

RKRK3588 TRM-Part1

Chapter 36 RKNN

36

36.1 Overview

Include triple NPU CORE

RKNN is the process unit which is dedicated to neural network. It is designed to accelerate
the neural network arithmetic in field of AI (artificial intelligence) such as machine vision and
natural language processing. The variety of applications for AI is expanding, and currently
provides functionality in a variety of areas, including face tracking as well as gesture and
body tracking, image classification, video surveillance, automatic speech recognition (ASR)
and advanced driver assistance systems (ADAS).
RKNN supports the following features:
⚫
⚫  Support triple core co-work, dual core co-work, and work independently
⚫  AHB interface used for configuration only support single
⚫  AXI interface used to fetch data from memory
⚫  Support integer 4, integer 8, integer 16, float 16, Bfloat 16 and tf32 operation
⚫  1024x3 integer 8 MAC operations per cycle
⚫  512x3 integer 16 MAC operations per cycle
⚫  512x3 float 16 MAC operations per cycle
⚫  512x3 bfloat 16 MAC operations per cycle
⚫  256x3 tf32 MAC operation per cycle
⚫  2048x3 integer 4 MAC operation per cycle
⚫  384KBx3 internal buffer
⚫

Inference Engine: TensorFlow, Caffe, Tflite, Pytorch, Onnx NN, Android NN, etc.

36.2 Block Diagram

Fig. 36-1 RKNN Triple Core Architecture

Copyright 2022 © Rockchip Electronics Co., Ltd.

1984

NPU CORE 0NPU CORE 1NPU CORE 2

RKRK3588 TRM-Part1

Fig. 36-2 RKNN Single Core Architecture

36.3 Function Description

36.3.1 1.3.1 AHB/AXI Interface

The AXI master interface is used to fetch data from memory that is attached to the Soc AXI
interconnect. The AHB slave interface is used to access the registers for configuration, debug
and test.

36.3.2 1.3.2 Neural Network Accelerating Engine

As the unit name, this engine is the main process unit for Neural Network arithmetic. This
unit include convolution pre-process controller, internal buffer, mac array, accumulator. It
provides parallel convolution MAC for recognition functions and int4, int8, int16, fp16,
bfloat16 and tf16 are supported.

36.3.3 1.3.3 Data Processing Unit

Data Processing Unit mainly process the single data calculate, such as leaky_relu, relu,
relux, sigmoid, tanh…etc. It also provides function: softmax, transpose, data format
conversion, …etc.

36.3.4 1.3.4 Planar Processing Unit

Planar Processing Unit mainly provide planar function followed by output data from Data
Processing Unit, such as average pooling, max pooling, min pooling… are supported.

Copyright 2022 © Rockchip Electronics Co., Ltd.

1985

NPU CORECNAFeature DataLoadWeight DataLoad384KBBufferSequence ControllerMAC ArrayAccumulatorPPU(Planar Process Unit)Register File/Interrupt ControlConfig Interface(AHB)DPU(Data Process Unit)Memory InterfaceAXIWeight DecompressZero-SkippingRegister FileFetch
RKRK3588 TRM-Part1

36.3.5 1.3.5 Register File Fetch Unit

Register File Fetch Unit fetch register configuration from external system memory through
AXI
interface.

36.4 Register Description

36.4.1 Internal Address Mapping

Slave address can be divided into different length for different usage, which is shown as
follows.

Base Address[15:12]

Device

Address
Length

Offset Address Range

Table 1- 1 RKNN Address Mapping

4’h0
4’h1
4’h3
4’h4
4’h5
4’h6
4’h7
4’h8
4’h9
4’hf

4K BYTE
PC
4K BYTE
CNA
4K BYTE
CORE
DPU
4K BYTE
DPU_RDMA  4K BYTE
PPU
4K BYTE
PPU_RDMA  4K BYTE
4K BYTE
DDMA
4K BYTE
SDMA
4    BYTE
GLOBAL

0x0000 ~ 0x0fff
0x1000 ~ 0x1fff
0x3000 ~ 0x3fff
0x4000 ~ 0x4fff
0x5000 ~ 0x5fff
0x6000 ~ 0x6fff
0x7000 ~ 0x7fff
0x8000 ~ 0x8fff
0x9000 ~ 0x9fff
0xf000 ~ 0xf004

36.4.2 Registers Summary

Name

Offset  Size

Reset
Value

Description

0x0014  W

0x0008  W

0x0028  W

RKNN_pc_operation_enab
le
RKNN_pc_base_address  0x0010  W
RKNN_pc_register_amoun
ts
RKNN_pc_interrupt_mask  0x0020  W
RKNN_pc_interrupt_clear  0x0024  W
RKNN_pc_interrupt_statu
s
RKNN_pc_interrupt_raw_s
tatus
RKNN_pc_task_con
RKNN_pc_task_dma_base
_addr
RKNN_pc_task_status
RKNN_cna_s_status
RKNN_cna_s_pointer
RKNN_cna_operation_ena
ble
RKNN_cna_conv_con1

0x003C  W
0x1000  W
0x1004  W

0x100C  W

0x002C  W

0x0030  W

0x0034  W

0x1008  W

0x00000000  Operation Enable

0x00000000  PC address register

0x00000000  Register amount for each task

0x0001FFFF  Interrupt Mask
0x00000000  Interrupt clear

0x00000000  Interrupt status

0x00000000  Interrupt raw status

0x00000000  Task control register

0x00000000  Task Base address

0x00000000  Task status register
0x00000000  Single register group status
0x00000000  Single register group pointer

0x00000000  Operation Enable

0x00000000  Convolution control register1

Copyright 2022 © Rockchip Electronics Co., Ltd.

1986

RKRK3588 TRM-Part1

Name

Offset  Size

0x1010  W
RKNN_cna_conv_con2
0x1014  W
RKNN_cna_conv_con3
0x1020  W
RKNN_cna_data_size0
0x1024  W
RKNN_cna_data_size1
0x1028  W
RKNN_cna_data_size2
0x102C  W
RKNN_cna_data_size3
RKNN_cna_weight_size0  0x1030  W
RKNN_cna_weight_size1  0x1034  W
RKNN_cna_weight_size2  0x1038  W
0x1040  W
RKNN_cna_cbuf_con0
0x1044  W
RKNN_cna_cbuf_con1
0x104C  W
RKNN_cna_cvt_con0
0x1050  W
RKNN_cna_cvt_con1
0x1054  W
RKNN_cna_cvt_con2
0x1058  W
RKNN_cna_cvt_con3
0x105C  W
RKNN_cna_cvt_con4
0x1060  W
RKNN_cna_fc_con0
0x1064  W
RKNN_cna_fc_con1
0x1068  W
RKNN_cna_pad_con0
RKNN_cna_feature_data_
addr
RKNN_cna_fc_con2
RKNN_cna_dma_con0
RKNN_cna_dma_con1
RKNN_cna_dma_con2

0x1074  W
0x1078  W
0x107C  W
0x1080  W

0x1070  W

Reset
Value

Description

0x00000000  Convolution control register2
0x00000000  Convolution control register3
0x00000000  Feature data size control register0
0x00000000  Feature data size control register1
0x00000000  Feature data size control register2
0x00000000  Feature data size control register3
0x00000000  Weight size control 0
0x00000000  Weight size control 1
0x00000000  Weight size control 2
0x00000000  CBUF control register 0
0x00000000  CBUF control register 1
0x00000000  Input convert control register0
0x00000000  Input convert control register1
0x00000000  Input convert control register2
0x00000000  Input convert control register3
0x00000000  Input convert control register4
0x00000000  Full connected control register0
0x00000000  Full connected control register1
0x00000000  Pad control register0

0x00000000

Base address for input feature
data

0x00000000  Full connected control register2
0x00000000  AXI control register 0
0x00000000  AXI control register 1
0x00000000  AXI control register 2

RKNN_cna_fc_data_size0  0x1084  W

0x00000000

RKNN_cna_fc_data_size1  0x1088  W

0x00000000

Full connected data size control
register0
Full connected data size control
register1

RKNN_cna_clk_gate

0x1090  W

RKNN_cna_dcomp_ctrl

0x1100  W

0x1140  W

0x1104  W

RKNN_cna_dcomp_regnu
m
RKNN_cna_dcomp_addr0  0x1110  W
RKNN_cna_dcomp_amoun
t0
RKNN_cna_dcomp_amoun
t1
RKNN_cna_dcomp_amoun
t2
RKNN_cna_dcomp_amoun
t3

0x1144  W

0x114C  W

0x1148  W

0x00000000

0x00000000  Clock gating control register
Weight decompress control
register
Weight decompress register
number

0x00000000

0x00000000  Base address of the weight

0x00000000

0x00000000

0x00000000

0x00000000

Amount of the weight decompress
for the 0 decompress
Amount of the weight decompress
for the 1 decompress
Amount of the weight decompress
for the 2 decompress
Amount of the weight decompress
for the 3 decompress

Copyright 2022 © Rockchip Electronics Co., Ltd.

1987

RKRK3588 TRM-Part1

Name

Offset  Size

0x1164  W

0x1150  W

0x115C  W

0x116C  W

0x1154  W

0x1174  W

0x1168  W

0x1158  W

0x1160  W

0x1170  W

RKNN_cna_dcomp_amoun
t4
RKNN_cna_dcomp_amoun
t5
RKNN_cna_dcomp_amoun
t6
RKNN_cna_dcomp_amoun
t7
RKNN_cna_dcomp_amoun
t8
RKNN_cna_dcomp_amoun
t9
RKNN_cna_dcomp_amoun
t10
RKNN_cna_dcomp_amoun
t11
RKNN_cna_dcomp_amoun
t12
RKNN_cna_dcomp_amoun
t13
RKNN_cna_dcomp_amoun
t14
RKNN_cna_dcomp_amoun
t15
RKNN_cna_cvt_con5
RKNN_cna_pad_con1
RKNN_core_s_status
RKNN_core_s_pointer
RKNN_core_operation_en
able
RKNN_core_mac_gating  0x300C  W
0x3010  W
RKNN_core_misc_cfg
RKNN_core_dataout_size_
0
RKNN_core_dataout_size_
1
RKNN_core_clip_truncate  0x301C  W
0x4000  W
RKNN_dpu_s_status
0x4004  W
RKNN_dpu_s_pointer
RKNN_dpu_operation_ena
ble
RKNN_dpu_feature_mode
_cfg

0x1180  W
0x1184  W
0x3000  W
0x3004  W

0x3008  W

0x117C  W

0x4008  W

0x3018  W

0x400C  W

0x1178  W

0x3014  W

Reset
Value

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

Description

Amount of the weight decompress
for the 4 decompress
Amount of the weight decompress
for the 5 decompress
Amount of the weight decompress
for the 6 decompress
Amount of the weight decompress
for the 7 decompress
Amount of the weight decompress
for the 8 decompress
Amount of the weight decompress
for the 9 decompress
Amount of the weight decompress
for the 10 decompress
Amount of the weight decompress
for the 11 decompress
Amount of the weight decompress
for the 12 decompress
Amount of the weight decompress
for the 13 decompress
Amount of the weight decompress
for the 14 decompress
Amount of the weight decompress
for the 15 decompress

0x00000000  Input convert control register5
0x00000000  Pad controller register1
0x00000000  Single register group status
0x00000000  Single register group pointer

0x00000000  Operation Enable

0x07800800  MAC gating register
0x00000000  Misc configuration register

0x00000000  Feature size register 0 of output

0x00000000  Feature size register 1 of output

0x00000000  Shift value register
0x00000000  Single register group status
0x00000000  Single register group pointer

0x00000000  Operation Enable

0x00000000  Configuration of the feature mode

Copyright 2022 © Rockchip Electronics Co., Ltd.

1988

RKRK3588 TRM-Part1

Name

Offset  Size

0x403C  W

0x4038  W

0x4034  W

0x404C  W

0x4030  W

0x4024  W

0x4020  W

0x4040  W
0x4044  W
0x4048  W

RKNN_dpu_data_format  0x4010  W
0x4014  W
RKNN_dpu_offset_pend
RKNN_dpu_dst_base_add
r
RKNN_dpu_dst_surf_strid
e
RKNN_dpu_data_cube_wi
dth
RKNN_dpu_data_cube_he
ight
RKNN_dpu_data_cube_no
tch_addr
RKNN_dpu_data_cube_ch
annel
RKNN_dpu_bs_cfg
RKNN_dpu_bs_alu_cfg
RKNN_dpu_bs_mul_cfg
RKNN_dpu_bs_relux_cmp
_value
0x4050  W
RKNN_dpu_bs_ow_cfg
RKNN_dpu_bs_ow_op
0x4054  W
RKNN_dpu_wdma_size_0  0x4058  W
RKNN_dpu_wdma_size_1  0x405C  W
0x4060  W
RKNN_dpu_bn_cfg
0x4064  W
RKNN_dpu_bn_alu_cfg
0x4068  W
RKNN_dpu_bn_mul_cfg
RKNN_dpu_bn_relux_cmp
_value
RKNN_dpu_ew_cfg
RKNN_dpu_ew_cvt_offset
_value
RKNN_dpu_ew_cvt_scale_
value
RKNN_dpu_ew_relux_cmp
_value
RKNN_dpu_out_cvt_offset  0x4080  W
RKNN_dpu_out_cvt_scale  0x4084  W
RKNN_dpu_out_cvt_shift  0x4088  W
RKNN_dpu_ew_op_value_
0
RKNN_dpu_ew_op_value_
1

0x4074  W

0x4070  W

0x4090  W

0x4094  W

0x406C  W

0x4078  W

0x407C  W

Reset
Value

Description

0x00000000  Configuration of the data format
0x00000000  Value of the offset pend

0x00000000  Destination base address

0x00000000  Destination surface size

0x00000000  Width of the input cube

0x00000000  Height of the input cube

0x00000000  Notch signal of the input cube

0x00000000  Channel of the input cube

0x00000000  Configuration of the BS
0x00000000  Configuration of the BS ALU
0x00000000  Configuration of the BS MUL

0x00000000  Value of the RELUX compare with

0x00000000  Configuration of the BS OW
0x00000000  Ow op of the BS OW
0x00000000  Size 0 of the WDMA
0x00000000  Size 1 of the WDMA
0x00000000  Configuration of BN
0x00000000  Configuration of the BN ALU
0x00000000  Configuration of the BN MUL

0x00000000  Value of the RELUX compare with

0x00000000  Configuration of EW

0x00000000  Offset of the EW input convert

0x00000000  Scale of the EW input convert

0x00000000  Value of the RELUX compare with

0x00000000  Offset of the output converter
0x00000000  Scale of the output converter
0x00000000  Shift of the output converter

0x00000000  Configure operand0 of the EW

0x00000000  Configure operand1 of the EW

Copyright 2022 © Rockchip Electronics Co., Ltd.

1989

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

0x40A0  W

0x40A4  W

0x40A8  W

0x4098  W

0x4104  W

0x40AC  W

0x409C  W

0x4108  W
0x410C  W
0x4110  W
0x4114  W
0x4118  W
0x411C  W

RKNN_dpu_ew_op_value_
2
RKNN_dpu_ew_op_value_
3
RKNN_dpu_ew_op_value_
4
RKNN_dpu_ew_op_value_
5
RKNN_dpu_ew_op_value_
6
RKNN_dpu_ew_op_value_
7
RKNN_dpu_surface_add
0x40C0  W
RKNN_dpu_lut_access_cfg 0x4100  W
RKNN_dpu_lut_access_da
ta
RKNN_dpu_lut_cfg
RKNN_dpu_lut_info
RKNN_dpu_lut_le_start
RKNN_dpu_lut_le_end
RKNN_dpu_lut_lo_start
RKNN_dpu_lut_lo_end
RKNN_dpu_lut_le_slope_s
cale
RKNN_dpu_lut_le_slope_s
hift
RKNN_dpu_lut_lo_slope_s
cale
RKNN_dpu_lut_lo_slope_s
hift
RKNN_dpu_rdma_s_statu
s
RKNN_dpu_rdma_s_point
er
RKNN_dpu_rdma_operati
on_enable
RKNN_dpu_rdma_data_cu
be_width
RKNN_dpu_rdma_data_cu
be_height
RKNN_dpu_rdma_data_cu
be_channel

0x5000  W

0x4124  W

0x5010  W

0x412C  W

0x5008  W

0x4128  W

0x5014  W

0x500C  W

0x4120  W

0x5004  W

0x00000000  Configure operand2 of the EW

0x00000000  Configure operand3 of the EW

0x00000000  Configure operand4 of the EW

0x00000000  Configure operand5 of the EW

0x00000000  Configure operand6 of the EW

0x00000000  Configure operand7 of the EW

0x00000000  Value of the surface adder
0x00000000  LUT access address and type

0x00000000  Configuration of LUT access data

0x00000000  Configuration of the LUT
0x00000000  LUT information register
0x00000000  LE LUT start point
0x00000000  LE LUT end point
0x00000000  LO LUT start point
0x00000000  LO LUT end point

0x00000000  LE LUT slope scale

0x00000000  LE LUT slope shift

0x00000000  LO LUT slope scale

0x00000000  LO LUT slope shift

0x00000000  Single register group status

0x00000000  Single register group pointer

0x00000000  Operation Enable

0x00000000  Input cube width

0x00000000  Input cube height

0x00000000  Input cube channel

Copyright 2022 © Rockchip Electronics Co., Ltd.

1990

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

0x5018  W

0x501C  W

0x5028  W

0x5020  W

0x5040  W

0x5034  W

0x5048  W

0x5038  W

0x5044  W

0x502C  W

RKNN_dpu_rdma_src_bas
e_addr
RKNN_dpu_rdma_brdma_
cfg
RKNN_dpu_rdma_bs_bas
e_addr
RKNN_dpu_rdma_nrdma_
cfg
RKNN_dpu_rdma_bn_bas
e_addr
RKNN_dpu_rdma_erdma_
cfg
RKNN_dpu_rdma_ew_bas
e_addr
RKNN_dpu_rdma_ew_surf
_stride
RKNN_dpu_rdma_feature
_mode_cfg
RKNN_dpu_rdma_src_dm
a_cfg
RKNN_dpu_rdma_surf_no
tch
RKNN_dpu_rdma_pad_cfg  0x5064  W
RKNN_dpu_rdma_weight  0x5068  W
RKNN_dpu_rdma_ew_surf
_notch
RKNN_ppu_s_status
RKNN_ppu_s_pointer
RKNN_ppu_operation_ena
ble
RKNN_ppu_data_cube_in
_width
RKNN_ppu_data_cube_in
_height
RKNN_ppu_data_cube_in
_channel
RKNN_ppu_data_cube_ou
t_width
RKNN_ppu_data_cube_ou
t_height
RKNN_ppu_data_cube_ou
t_channel

0x6000  W
0x6004  W

0x600C  W

0x504C  W

0x6020  W

0x6018  W

0x6014  W

0x6008  W

0x506C  W

0x6010  W

0x601C  W

0x00000000  Base address of the input cube

0x00000000  Configurations of BRDMA

0x00000000  Source base address of BRDMA

0x00000000  Configurations of NRDMA

0x00000000  Source base address of NRDMA

0x00000000  Configurations of ERDMA

0x00000000  Source base address of ERDMA

0x00000000

Surface size of the cube that the
ERDMA read

0x00000000  Configuration of the feature mode

0x00000000

Configuration of the source read
DMA

0x00000000  Surface notch

0x00000000  Configuration of the pad
0x00000000  Weight of the arbiter

0x00000000  Surface notch

0x00000000  Single register group status
0x00000000  Single register group pointer

0x00000000  Operation Enable

0x00000000  Width of the input cube

0x00000000  Height of the input cube

0x00000000  Channel of the input cube

0x00000000  Width of the output cube

0x00000000  Height of the output cube

0x00000000  Channel of the output cube

Copyright 2022 © Rockchip Electronics Co., Ltd.

1991

RKRK3588 TRM-Part1

Name

Offset  Size

0x6070  W

0x6044  W

0x6040  W

0x607C  W

0x6048  W

0x6034  W

0x6024  W

0x6038  W

0x603C  W

RKNN_ppu_operation_mo
de_cfg
RKNN_ppu_pooling_kerne
l_cfg
RKNN_ppu_recip_kernel_
width
RKNN_ppu_recip_kernel_
height
RKNN_ppu_pooling_paddi
ng_cfg
RKNN_ppu_padding_value
_1_cfg
RKNN_ppu_padding_value
_2_cfg
RKNN_ppu_dst_base_add
r
RKNN_ppu_dst_surf_strid
e
RKNN_ppu_data_format  0x6084  W
0x60DC  W
RKNN_ppu_misc_ctrl
RKNN_ppu_rdma_s_statu
s
RKNN_ppu_rdma_s_point
er
RKNN_ppu_rdma_operati
on_enable
RKNN_ppu_rdma_cube_in
_width
RKNN_ppu_rdma_cube_in
_height
RKNN_ppu_rdma_cube_in
_channel
RKNN_ppu_rdma_src_bas
e_addr
RKNN_ppu_rdma_src_line
_stride
RKNN_ppu_rdma_src_surf
_stride
RKNN_ppu_rdma_data_fo
rmat
RKNN_ddma_cfg_outstan
ding
RKNN_ddma_rd_weight_0  0x8004  W

0x701C  W

0x7030  W

0x7024  W

0x7028  W

0x7010  W

0x700C  W

0x8000  W

0x7014  W

0x7000  W

0x7008  W

0x7004  W

Reset
Value

Description

0x00000000

0x00000000

Configuration of the operation
mode
Configuration of the pooling
kernel size

0x00000000  The reciprocal of the kernel width

0x00000000  The reciprocal of the kernel height

0x00000000  Configuration of the pooling pad

0x00000000  Pad Value register0

0x00000000  Pad Value register1

0x00000000

Destination address of the output
cube

0x00000000  Destination surface size

0x00000000  Configuration of the data format
0x00000000  Misc configuration

0x00000000  Single register group status

0x00000000  Single register group pointer

0x00000000  Operation Enable

0x00000000  Input cube width

0x00000000  Input cube height

0x00000000  Input cube channel

0x00000000

0x00000000

0x00000000

Source base address for input
feature
Source line number including
Virtual box
Source surface size including
Virtual box

0x00000000  Data format for the input feature

0x00000000  Outstanding config register

0x00000000  Weight of read arbiter

Copyright 2022 © Rockchip Electronics Co., Ltd.

1992

RKRK3588 TRM-Part1

Name

Offset  Size

0x8024  W

0x8008  W

0x802C  W

0x8030  W

0x8028  W

0x8018  W

0x8014  W

0x8020  W

RKNN_ddma_wr_weight_
0
RKNN_ddma_cfg_id_error  0x800C  W
RKNN_ddma_rd_weight_1  0x8010  W
RKNN_ddma_cfg_dma_fif
o_clr
RKNN_ddma_cfg_dma_ar
b
RKNN_ddma_cfg_dma_rd
_qos
RKNN_ddma_cfg_dma_rd
_cfg
RKNN_ddma_cfg_dma_wr
_cfg
RKNN_ddma_cfg_dma_ws
trb
RKNN_ddma_cfg_status
RKNN_sdma_cfg_outstan
ding
RKNN_sdma_rd_weight_0  0x9004  W
RKNN_sdma_wr_weight_0 0x9008  W
RKNN_sdma_cfg_id_error  0x900C  W
RKNN_sdma_rd_weight_1  0x9010  W
RKNN_sdma_cfg_dma_fif
o_clr
RKNN_sdma_cfg_dma_ar
b
RKNN_sdma_cfg_dma_rd
_qos
RKNN_sdma_cfg_dma_rd
_cfg
RKNN_sdma_cfg_dma_wr
_cfg
RKNN_sdma_cfg_dma_ws
trb
RKNN_sdma_cfg_status
RKNN_global_operation_e
nable

0x9000  W

0x9014  W

0xF008  W

0x902C  W

0x9024  W

0x9028  W

0x9018  W

0x9030  W

0x9020  W

Reset
Value

Description

0x00000000  Weight of write arbiter

0x00000000  Id where error happened
0x00000000  Weight of read arbiter register1

0x00000000  Clear DMA FIFO

0x00000000  DMA arbiter configuration register

0x00000000  Read Qos for DMA

0x00000000  Read configuration for AXI signals

0x00000000  Write configuration for AXI signals

0x00000000  Write strobe signal for AXI

0x00000000  AXI status signal

0x00000000  Outstanding configuration register

0x00000000  Weight of read arbiter
0x00000000  Weight of write arbiter
0x00000000  Id where error happened
0x00000000  Weight of read arbiter register1

0x00000000  Clear DMA FIFO

0x00000000  DMA arbiter configuration register

0x00000000  Read Qos for DMA

0x00000000  Read configuration for AXI signals

0x00000000  Write configuration for AXI signals

0x00000000  Write strobe signal for AXI

0x00000000  AXI status signal

0x00000000  Combine Operation Enable

Notes:Size:B- Byte (8 bits) access, HW- Half WORD (16 bits) access, W-WORD (32 bits) access, DW-
Double WORD (64 bits) access

36.4.3 Detail Registers Description

RKNN_pc_operation_enable
Address: Operational Base + offset (0x0008)

Copyright 2022 © Rockchip Electronics Co., Ltd.

1993

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
PC operation enable.
1'd0: Disable PC module;
1'd1: Enable PC module to fetch register for each task.

RKNN_pc_base_address
Address: Operational Base + offset (0x0010)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:1  RO  0x0

0

RW  0x0

pc_source_addr
PC base address.
This is the address of DMA instruction where it located.
reserved
pc_sel
PC mode enable.
1'd0: PC mode, use AXI DMA to fetch register config;
1'd1: Slave mode, use AHB to set register.

RKNN_pc_register_amounts
Address: Operational Base + offset (0x0014)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:0  RW  0x0000

Description

reserved
pc_data_amount
Data amount.
The register number need to be fetched of one task. Each register
takes 64 bits, it is combined as following:
bit[63:48] indicates which block the register forward to.
bit[47:16] the register's value
bit[15: 0] the register's offset address in each block.
bit[56]= 1 means this register is for pc block.
bit[57]= 1 CNA
bit[59]= 1 CORE
bit[60]= 1 DPU
bit[61]= 1 DPU_RDMA
bit[62]= 1 PPU
bit[63]= 1 PPU_RDMA
bit[55]= 1 to set each block's op_en
eg. 64'h0081_0000_007f_0008 will set each block's op_en(CNA,
CORE, ..., PPU_RDMA).
note: op_en is strongly recommended set at the end of register
list.
before op_en, 64'h0041_xxxx_xxxx_xxxx must be set.

RKNN_pc_interrupt_mask
Address: Operational Base + offset (0x0020)

Copyright 2022 © Rockchip Electronics Co., Ltd.

1994

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0  RW  0x1ffff

Description

reserved
int_mask
Interrupt mask.
mask[0 ]: CNA feature group 0 interrupt mask. set 1 to enable
interrupt;
mask[1 ]: CNA feature group 1;
mask[2 ]: CNA weight group 0;
mask[3 ]: CNA weight group 1;
mask[4 ]: CNA csc group 0;
mask[5 ]: CNA csc group 1;
mask[6 ]: CORE group 0;
mask[7 ]: CORE group 1;
mask[8 ]: DPU group 0;
mask[9 ]: DPU group 1;
mask[10]: PPU group 0;
mask[11]: PPU group 1;
mask[12]: DMA read error;
mask[13]: DMA write error;
Note: In pc mode, int mask set the last one task's interrupt
masking.

RKNN_pc_interrupt_clear
Address: Operational Base + offset (0x0024)

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0

W1
C

0x00000

Description

reserved
int_clr
Interrupt clear.
done_clr[0 ]: CNA feature group 0 interrupt clear;
done_clr[1 ]: CNA feature group 1;
done_clr[2 ]: CNA weight group 0;
done_clr[3 ]: CNA weight group 1;
done_clr[4 ]: CNA csc group 0;
done_clr[5 ]: CNA csc group 1;
done_clr[6 ]: CORE group 0;
done_clr[7 ]: CORE group 1;
done_clr[8 ]: DPU group 0;
done_clr[9 ]: DPU group 1;
done_clr[10]: PPU group 0;
done_clr[11]: PPU group 1;
done_clr[12]: DMA read error;
done_clr[13]: DMA write error.

RKNN_pc_interrupt_status
Address: Operational Base + offset (0x0028)

Copyright 2022 © Rockchip Electronics Co., Ltd.

1995

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0

W1
C

0x00000

Description

reserved
int_st
Interrupt status.
int_st[0 ]: CNA feature group 0 interrupt status, which and with
mask bit;
int_st[1 ]: CNA feature group 1;
int_st[2 ]: CNA weight group 0;
int_st[3 ]: CNA weight group 1;
int_st[4 ]: CNA csc group 0;
int_st[5 ]: CNA csc group 1;
int_st[6 ]: CORE group 0;
int_st[7 ]: CORE group 1;
int_st[8 ]: DPU group 0;
int_st[9 ]: DPU group 1;
int_st[10]: PPU group 0;
int_st[11]: PPU group 1;
int_st[12]: DMA read error;
int_st[13]: DMA write error.

RKNN_pc_interrupt_raw_status
Address: Operational Base + offset (0x002C)

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0

W1
C

0x00000

Description

reserved
int_raw_st
Interrupt raw status.
int_st[0 ]: CNA feature group 0 interrupt raw status;
int_st[1 ]: CNA feature group 1;
int_st[2 ]: CNA weight group 0;
int_st[3 ]: CNA weight group 1;
int_st[4 ]: CNA csc group 0;
int_st[5 ]: CNA csc group 1;
int_st[6 ]: CORE group 0;
int_st[7 ]: CORE group 1;
int_st[8 ]: DPU group 0;
int_st[9 ]: DPU group 1;
int_st[10]: PPU group 0;
int_st[11]: PPU group 1;
int_st[12]: DMA read error;
int_st[13]: DMA write error.

RKNN_pc_task_con
Address: Operational Base + offset (0x0030)

Bit  Attr  Reset Value

31:14 RO  0x00000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

1996

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

13

W1
C

0x0

12

RW  0x0

11:0  RW  0x000

task_count_clear
Task counter clear register.
Clear the counter that counting current task. Before task started,
it is suggested to clear.
task_pp_en
PC task ping-pong mode enable.
1'd0: Ping-pong mode off. The second group register setting is
fetched after first group task operation is finished;
1'd1: Tasks' registers are fetched in ping-pong mode. The second
group register setting is fetched immediately after first group's
register fetching is finished.
task_number
PC task number.
Set the total task number to be executed.

RKNN_pc_task_dma_base_addr
Address: Operational Base + offset (0x0034)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

dma_base_addr
Task base address.
This is for each DMA's base address.
For feature DMA, weight DMA, DPU DMA, PPU DMA, the address
is set as offset address. Final address appear on AXI bus is base
address + offset address.
reserved

RKNN_pc_task_status
Address: Operational Base + offset (0x003C)

Bit  Attr  Reset Value

31:28 RO  0x0

27:0  RW  0x0000000

Description

reserved
task_status
Task status.
[11:0]: Current task counter value;
[12]: Indicate the first task is operating;
[13]: Indicate the last task is operating;
[12]: Indicate the first task's register is fetching;
[13]: Indicate the last task's register is fetching.

RKNN_cna_s_status
Address: Operational Base + offset (0x1000)

Bit  Attr  Reset Value

31:18 RO  0x0000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

1997

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

17:16 RO  0x0

15:2  RO  0x0000

1:0  RO  0x0

status_1
Executer 1 status.
2'd0: Executer 1 is in idle state;
2'd1: Executer 1 is operating;
2'd2: Executer 1 is operating, executer 1 is waiting to operate;
2'd3: Reserved.
reserved
status_0
Executer 0 status.
2'd0: Executer 0 is in idle state;
2'd1: Executer 0 is operating;
2'd2: Executer 0 is operating, executer 1 is waiting to operate;
2'd3: Reserved.

RKNN_cna_s_pointer
Address: Operational Base + offset (0x1004)

Bit  Attr  Reset Value

31:17 RO  0x0000

16

RO  0x0

15:6  RO  0x000

5

4

W1
C

W1
C

0x0

0x0

3

RW  0x0

2

RW  0x0

1

RW  0x0

Description

reserved
executer
Which register group to be used.
1'd0: Executer group 0;
1'd1: Executer group 1.
reserved
executer_pp_clear
Clear executer group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_clear
Clear register group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_mode
Register group ping-pong mode.
1'd0: Pointer ping-pong by executer;
      eg. if current executer is 0, next pointer will toggle to 1;
1'd1: Pointer ping-pong by pointer;
      eg. if current pointer is 0, next pointer will toggle to 1.
executer_pp_en
Executer group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer_pp_en
Register group ping-pong enable.
1'd0: Disable;
1'd1: Enable.

Copyright 2022 © Rockchip Electronics Co., Ltd.

1998

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

0

RW  0x0

pointer
Which register group ready to be set.
1'd0: Register group 0;
1'd1: Register group 1.

RKNN_cna_operation_enable
Address: Operational Base + offset (0x1008)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
Set this register will trigger CNA block operate.
This register and after this are all shadowed for ping-pong
operation.
1'd0: Disable;
1'd1: Enable.

RKNN_cna_conv_con1
Address: Operational Base + offset (0x100C)

Bit  Attr  Reset Value

31

RO  0x0

30

RW  0x0

29

RW  0x0

28:17 RO  0x000

16

RW  0x0

15:12 RW  0x0

11:10 RO  0x0

Description

reserved
nonalign_dma
CNA DMA non-align mode.
1'd0: Disable;
1'd1: Enable, please enable this bit under ARGB mode.
Enable this bit will enable DMA fetching feature data
continuously.
group_line_off
Group line fetch off.
1'd0: Enable group line fetch;
1'd1: Disable.
This setting only influence line fetch efficiency.
reserved
deconv
Enable deconvolution function.
1'd0: Disable;
1'd1: Enable.
argb_in
Non-align channel layer control register.
4'd8: 1 channel input mode;
4'd9: 2 channel input mode;
4'd10: 3 channel input mode;
4'd11: 4 channel input mode.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

1999

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

9:7  RW  0x0

6:4  RW  0x0

3:0  RW  0x0

proc_precision
Process precision.
3'd0: Input precision is int 8;
3'd1: Input data precision is int 16;
3'd2: Input data precision is float 16;
3'd3: Input data precision is bfloat 16;
3'd4: Reserved;
3'd5: Reserved;
3'd6: Input data precision is int4;
3'd7: Input data precision is tf32.
in_precision
Input precision.
3'd0: Input precision is int 8;
3'd1: Input data precision is int 16;
3'd2: Input data precision is float 16;
3'd3: Input data precision is bfloat 16;
3'd4: Reserved;
3'd5: Reserved;
3'd6: Input data precision is int4;
3'd7: Input data precision is tf32.
conv_mode
Convolution mode.
2'd0: Direct convolution;
2'd1: Reserved;
2'd2: Reserved;
2'd3: Depthwise convolution.

RKNN_cna_conv_con2
Address: Operational Base + offset (0x1010)

Bit  Attr  Reset Value

31:24 RO  0x00

23:16 RW  0x00

15:14 RO  0x0

13:4  RW  0x000

3

RO  0x0

Description

reserved
kernel_group
Kernels group.
In int8, 32 kernels form 1 group, in int16 or fp16, 16 kernels
form 1 group.
eg, weight kernel is 256, in int8, you can set this register to be
256/32 -1 = 15.
reserved
feature_grains
Feature data rows needs to be buffered before convolution start.
It's suggested to set this field as y_stride+weight_height+1.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

2000

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

2

RW  0x0

1

RW  0x0

0

RW  0x0

csc_wo_en
Do weight scan.
1'd0: Enable csc output weight data to core;
1'd1: Disable.
csc_do_en
Do data scan.
1'd0: Enable csc output feature data to core;
1'd1: Disable.
cmd_fifo_srst
Command FIFO soft reset.
Reserved for debug purpose.

RKNN_cna_conv_con3
Address: Operational Base + offset (0x1014)

Bit  Attr  Reset Value

31

RO  0x0

30:28 RW  0x0

27:26 RO  0x0

25:21 RW  0x00

20:16 RW  0x00

15:14 RO  0x0

13:11 RW  0x0

10:8  RW  0x0

7:6  RO  0x0

Description

reserved
nn_mode
co-work mode.
3'd0: Int8 mac array 32x32 mode;
3'd1: 64x32 mode;
3'd2: 96x32 mode;
3'd3: Reserved;
3'd4: 32x64 mode;
3'd5: 32x96 mode;
3'd6: Reserved;
3'd7: Reserved.
This register is target for multicore mode. By single core mode
keep it at 3'd0.
reserved
atrous_y_dilation
Atrous x dilation.
Pad numbers inserted in feature map column between 2 pixels.
atrous_x_dilation
Atrous x dilation.
Pad numbers inserted in feature map row between 2 pixels.
Set this register value >0 will enable atrous convolution.
reserved
deconv_y_stride
Deconvolution y stride.
Pad numbers inserted in feature map column between 2 pixels.
deconv_x_stride
Deconvolution x stride.
Pad numbers inserted in feature map row between 2 pixels.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

2001

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

5:3  RW  0x0

2:0  RW  0x0

conv_y_stride
Convolution y stride.
Stride value in y direction.
conv_x_stride
Convolution x stride.
Stride value in x direction.

Description

RKNN_cna_data_size0
Address: Operational Base + offset (0x1020)

Bit  Attr  Reset Value

31:27 RO  0x00

26:16 RW  0x000

15:11 RO  0x00

10:0  RW  0x000

reserved
datain_width
Input feature data width.
reserved
datain_height
Input feature data height.

RKNN_cna_data_size1
Address: Operational Base + offset (0x1024)

Bit  Attr  Reset Value

31:30 RO  0x0

29:16 RW  0x0000

15:0  RW  0x0000

Description

reserved
datain_channel_real
Real channel number.
If the input channel is not integer times of 8(int8) or 4(int
16/float 16), set the real channel number in this field.
datain_channel
Input feature data channel number;
Int 8 mode, this number should be integer times of 8;
Int 16/float 16 mode, should be integer times of 4.

RKNN_cna_data_size2
Address: Operational Base + offset (0x1028)

Bit  Attr  Reset Value

31:11 RO  0x000000

10:0  RW  0x000

Description

reserved
dataout_width
Data width after convolution.

RKNN_cna_data_size3
Address: Operational Base + offset (0x102C)

Bit  Attr  Reset Value

31:24 RO  0x00

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2002

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

23:22 RW  0x0

21:0  RW  0x000000

surf_mode
Surface serial mode.
2'd0: 1surf series;
2'd1: 1surf series;
2'd2: 2 surf series;
2'd3: 4 surf series.
dataout_atomics
Data atomics after convolution which is data out total pixels
number.

RKNN_cna_weight_size0
Address: Operational Base + offset (0x1030)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

weight_bytes
Weight bytes in total for this convolution.

RKNN_cna_weight_size1
Address: Operational Base + offset (0x1034)

Bit  Attr  Reset Value

31:19 RO  0x0000

18:0  RW  0x00000

Description

reserved
weight_bytes_per_kernel
Weight bytes for one kernel.

RKNN_cna_weight_size2
Address: Operational Base + offset (0x1038)

Bit  Attr  Reset Value

Description

31:29 RO  0x0

28:24 RW  0x00

23:21 RO  0x0

20:16 RW  0x00

15:14 RO  0x0

13:0  RW  0x0000

reserved
weight_width
Kernel width.
reserved
weight_height
Kernel height.
reserved
weight_kernels
Weight kernels.

RKNN_cna_cbuf_con0
Address: Operational Base + offset (0x1040)

Bit  Attr  Reset Value

31:14 RO  0x00000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2003

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

13

RW  0x0

12

RW  0x0

11

RO  0x0

10:8  RW  0x0

7:4  RW  0x0

3:0  RW  0x0

weight_reuse
Weight data reuse enable.
1'd0: Disable;
1'd1: Enable data reuse. fetching weight directly from internal
buffer.
data_reuse
Feature data reuse enable.
1'd0: Disable;
1'd1: Enable data reuse. fetching data directly from internal
buffer.
reserved
fc_data_bank
Bank numbers for fc zero-skipping feature data. In FC zero-
skipping mode, set to be 1, Otherwise, must set to be 0.
weight_bank
Bank numbers for weight data.
4'd1: Bank 7 occupied by weight data;
4'd2: Bank 6/7 occupied by weight data;
...
4'd7: Bank 1-7 occupied by weight data.
data_bank
Bank numbers for feature data.
4'd0: Bank 0 occupied by feature data;
4'd1: Bank 0 and bank 1 occupied by feature data;
4'd2: Bank 0/1/2 occupied by feature data;
...
4'd6: Bank 0-6 occupied by feature data.

RKNN_cna_cbuf_con1
Address: Operational Base + offset (0x1044)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
data_entries
How many banks space needed to store one feature map row.

RKNN_cna_cvt_con0
Address: Operational Base + offset (0x104C)

Bit  Attr  Reset Value

Description

31:28 RO  0x0

27:22 RW  0x00

21:16 RW  0x00

15:10 RW  0x00

reserved
cvt_truncate_3
CVT truncate value 3.
cvt_truncate_2
CVT truncate value 2.
cvt_truncate_1
CVT truncate value 1.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2004

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

9:4  RW  0x00

3

RW  0x0

2

RW  0x0

1

RW  0x0

0

RW  0x0

cvt_truncate_0
CVT truncate value 0.
data_sign
Feature data is signed or unsigned.
1'd0: Unsigned;
1'd1: Signed.
round_type
Rounding type of the input convert.
1'd0: Odd in, even not;
1'd1: Round-up 0.5 to 1.
cvt_type
Cal type of the input convert.
1'd0: Multiply first, then add;
1'd1: CVT function will do add first, then multiply.
cvt_bypass
Bypass input convert.
1'd0: Enable CVT function;
1'd1: Disable CVT function.

RKNN_cna_cvt_con1
Address: Operational Base + offset (0x1050)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:0  RW  0x0000

cvt_scale0
CVT scale 0.
Multiplier operand for 1st channel.
cvt_offset0
CVT offset 0.
Adder operand for 1st channel.

RKNN_cna_cvt_con2
Address: Operational Base + offset (0x1054)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:0  RW  0x0000

cvt_scale1
CVT scale 1
Multiplier operand for 2nd channel.
cvt_offset1
CVT offset 1.
Adder operand for 2nd channel.

RKNN_cna_cvt_con3
Address: Operational Base + offset (0x1058)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

cvt_scale2
CVT scale 2.
Multiplier operand for 3rd channel.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2005

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

15:0  RW  0x0000

cvt_offset2
CVT offset 2.
Adder operand for 3rd channel.

RKNN_cna_cvt_con4
Address: Operational Base + offset (0x105C)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:0  RW  0x0000

cvt_scale3
CVT scale 3.
Multiplier operand for 4th channel.
cvt_offset3
CVT offset 3.
Adder operand for 4th channel.

RKNN_cna_fc_con0
Address: Operational Base + offset (0x1060)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:1  RO  0x0000

0

RW  0x0

fc_skip_data
FC zero skipping data
Skipped feature data value, normally set to 0.
reserved
fc_skip_en
FC zero skipping enable
1'd0: Disable;
1'd1: Enable skip some feature data value, normally skip zero.
When one pixel feature data is 0, the corresponding weight data
is not fetched from system memory.

RKNN_cna_fc_con1
Address: Operational Base + offset (0x1064)

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0  RW  0x00000

Description

reserved
data_offset
FC zero skipping data offset.
Feature data offset in fc skip mode.

RKNN_cna_pad_con0
Address: Operational Base + offset (0x1068)

Bit  Attr  Reset Value

31:8  RO  0x000000

7:4  RW  0x0

Description

reserved
pad_left
Pad left
Pad numbers in left of the feature map.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2006

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

3:0  RW  0x0

pad_top
PAD top
Pad numbers in top of the feature map.

RKNN_cna_feature_data_addr
Address: Operational Base + offset (0x1070)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

feature_base_addr
Feature data address.

RKNN_cna_fc_con2
Address: Operational Base + offset (0x1074)

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0  RW  0x00000

reserved
weight_offset
Weight data address.

Description

RKNN_cna_dma_con0
Address: Operational Base + offset (0x1078)

Bit  Attr  Reset Value

Description

31

RW  0x0

30:20 RO  0x000

19:16 RW  0x0

15:4  RO  0x000

3:0  RW  0x0

ov4k_bypass
Separate the burst command of over 4k to 2 independent burst
commands.
1'd0: Enable this feature;
1'd1: Bypass this feature.
reserved
weight_burst_len
AXI burst length for weight data DMA.
4'd3: Burst length is 4;
4'd7: Burst length is 8;
4'd15: Burst length is 16.
reserved
data_burst_len
AXI burst length for feature data DMA.
4'd3: Burst length is 4;
4'd7: Burst length is 8;
4'd15: Burst length is 16.

RKNN_cna_dma_con1
Address: Operational Base + offset (0x107C)

Bit  Attr  Reset Value

31:28 RO  0x0

27:0  RW  0x0000000

Description

reserved
line_stride
Line stride
Feature width with Virtual box.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2007

RKRK3588 TRM-Part1

RKNN_cna_dma_con2
Address: Operational Base + offset (0x1080)

Bit  Attr  Reset Value

31:28 RO  0x0

27:0  RW  0x0000000

Description

reserved
surf_stride
Surface stride
Feature map actual surface area.

RKNN_cna_fc_data_size0
Address: Operational Base + offset (0x1084)

Bit  Attr  Reset Value

31:30 RO  0x0

29:16 RW  0x0000

15:11 RO  0x00

10:0  RW  0x000

Description

reserved
dma_width
Feature input width for AXI DMA.
reserved
dma_height
Feature input height for AXI DMA.

RKNN_cna_fc_data_size1
Address: Operational Base + offset (0x1088)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:0  RW  0x0000

Description

reserved
dma_channel
Feature input channel for AXI DMA.

RKNN_cna_clk_gate
Address: Operational Base + offset (0x1090)

Bit  Attr  Reset Value

31:5  RO  0x0000000

4

3

2

RW  0x0

RO  0x0

RW  0x0

1

RW  0x0

Description

reserved
cbuf_cs_disable_clkgate
Cache auto gating.
1'd0: Auto clock gate is enabled;
1'd1: Disable CBUF clock auto gate.
reserved
csc_disable_clkgate
Sequence scan auto gating.
1'd0: Auto clock gate is enabled;
1'd1: Disable csc block clock gate.
cna_weight_disable_clkgate
Weight fetch auto gating.
1'd0: Auto clock gate is enabled;
1'd1: Disable weight block clock gate.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2008

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

0

RW  0x0

cna_feature_disable_clkgate
Feature fetch auto gating.
1'd0: Auto clock gate is enabled;
1'd1: Disable feature block clock gate.

RKNN_cna_dcomp_ctrl
Address: Operational Base + offset (0x1100)

Bit  Attr  Reset Value

31:4  RO  0x0000000

3

RW  0x0

2:0  RW  0x0

Description

reserved
wt_dec_bypass
Bypass weight decompress.
decomp_control
Control register for weight decompress.

RKNN_cna_dcomp_regnum
Address: Operational Base + offset (0x1104)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_regnum
Weight decompress register number.

RKNN_cna_dcomp_addr0
Address: Operational Base + offset (0x1110)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

decompress_addr0
Base address of the weight.
reserved

RKNN_cna_dcomp_amount0
Address: Operational Base + offset (0x1140)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount0
Amount of the weight decompress for the 0 decompress.

RKNN_cna_dcomp_amount1
Address: Operational Base + offset (0x1144)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount1
Amount of the weight decompress for the 1 decompress.

RKNN_cna_dcomp_amount2
Address: Operational Base + offset (0x1148)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount2
Amount of the weight decompress for the 2 decompress.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2009

RKRK3588 TRM-Part1

RKNN_cna_dcomp_amount3
Address: Operational Base + offset (0x114C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount3
Amount of the weight decompress for the 3 decompress.

RKNN_cna_dcomp_amount4
Address: Operational Base + offset (0x1150)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount4
Amount of the weight decompress for the 4 decompress.

RKNN_cna_dcomp_amount5
Address: Operational Base + offset (0x1154)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount5
Amount of the weight decompress for the 5 decompress.

RKNN_cna_dcomp_amount6
Address: Operational Base + offset (0x1158)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount6
Amount of the weight decompress for the 6 decompress.

RKNN_cna_dcomp_amount7
Address: Operational Base + offset (0x115C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount7
Amount of the weight decompress for the 7 decompress.

RKNN_cna_dcomp_amount8
Address: Operational Base + offset (0x1160)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount8
Amount of the weight decompress for the 8 decompress.

RKNN_cna_dcomp_amount9
Address: Operational Base + offset (0x1164)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount9
Amount of the weight decompress for the 9 decompress.

RKNN_cna_dcomp_amount10
Address: Operational Base + offset (0x1168)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2010

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount10
Amount of the weight decompress for the 10 decompress.

RKNN_cna_dcomp_amount11
Address: Operational Base + offset (0x116C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount11
Amount of the weight decompress for the 11 decompress.

RKNN_cna_dcomp_amount12
Address: Operational Base + offset (0x1170)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount12
Amount of the weight decompress for the 12 decompress.

RKNN_cna_dcomp_amount13
Address: Operational Base + offset (0x1174)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount13
Amount of the weight decompress for the 13 decompress.

RKNN_cna_dcomp_amount14
Address: Operational Base + offset (0x1178)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount14
Amount of the weight decompress for the 14 decompress.

RKNN_cna_dcomp_amount15
Address: Operational Base + offset (0x117C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

dcomp_amount15
Amount of the weight decompress for the 15 decompress.

RKNN_cna_cvt_con5
Address: Operational Base + offset (0x1180)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

per_channel_cvt_en
convert enable.
Per channel enable CVT function.
Int 4 has 32 channels in total for 128 bits.
Int 8 16 channel...

RKNN_cna_pad_con1
Address: Operational Base + offset (0x1184)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2011

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

pad_value
Pad value.

RKNN_core_s_status
Address: Operational Base + offset (0x3000)

Bit  Attr  Reset Value

31:18 RO  0x0000

17:16 RO  0x0

15:2  RO  0x0000

1:0  RO  0x0

Description

reserved
status_1
Executer 1 status.
2'd0: Executer 1 is in idle state;
2'd1: Executer 1 is operating;
2'd2: Executer 1 is operating, executer 1 is waiting to operate;
2'd3: Reserved.
reserved
status_0
Executer 0 status.
2'd0: Executer 0 is in idle state;
2'd1: Executer 0 is operating;
2'd2: Executer 0 is operating, executer 1 is waiting to operate;
2'd3: Reserved.

RKNN_core_s_pointer
Address: Operational Base + offset (0x3004)

Bit  Attr  Reset Value

31:17 RO  0x0000

16

RO  0x0

15:6  RO  0x000

5

4

W1
C

W1
C

0x0

0x0

3

RW  0x0

Description

reserved
executer
Which register group to be used.
1'd0: Executer group 0;
1'd1: Executer group 1.
reserved
executer_pp_clear
Clear executer group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_clear
Clear register group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_mode
Register group ping-pong mode.
1'd0: Pointer ping-pong by executer;
      eg. if current executer is 0, next pointer will toggle to 1;
1'd1: Pointer ping-pong by pointer;
      eg. if current pointer is 0, next pointer will toggle to 1.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2012

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

2

RW  0x0

1

RW  0x0

0

RW  0x0

executer_pp_en
Executer group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer_pp_en
Register group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer
Which register group ready to be set.
1'd0: Register group 0;
1'd1: Register group 1.

RKNN_core_operation_enable
Address: Operational Base + offset (0x3008)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
Set this register will trigger core block operate.
This register and after this are all shadowed for ping-pong
operation.
1'd0: Disable;
1'd1: Enable.

RKNN_core_mac_gating
Address: Operational Base + offset (0x300C)

Bit  Attr  Reset Value

31:27 RO  0x00

26:0  RW  0x7800800

reserved
slcg_op_en
Soft clock gating signals.

Description

RKNN_core_misc_cfg
Address: Operational Base + offset (0x3010)

Bit  Attr  Reset Value

31:20 RO  0x000

19:14 RW  0x00

13:11 RO  0x0

Description

reserved
soft_gating
Accumulate soft gating signal.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

2013

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

10:8  RW  0x0

7:2  RO  0x00

1

RW  0x0

0

RW  0x0

proc_precision
Process precision
3'd0: Input precision is int 8;
3'd1: Input data precision is int 16;
3'd2: Input data precision is float 16;
3'd3: Input data precision is bfloat 16;
3'd4: Reserved;
3'd5: Reserved;
3'd6: Input data precision is int4;
3'd7: Input data precision is tf32.
reserved
dw_en
Depthwise enable
1'd0: Disable;
1'd1: Depthwise mode enable.
qd_en
Quantify feature data calculate enable
1'd0: Disable;
1'd1: Enable.

RKNN_core_dataout_size_0
Address: Operational Base + offset (0x3014)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:0  RW  0x0000

dataout_height
Data out height
Data height after activation.
dataout_width
Data out width.
Data width after activation.

RKNN_core_dataout_size_1
Address: Operational Base + offset (0x3018)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:0  RW  0x0000

Description

reserved
dataout_channel
Data out channel number
Data channel number after activation.

RKNN_core_clip_truncate
Address: Operational Base + offset (0x301C)

Bit  Attr  Reset Value

31:7  RO  0x0000000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2014

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

6

5

RW  0x0

RO  0x0

4:0  RW  0x00

round_type
Rounding type
1'b0: Odd in, even not;
1'b1: Round-up 0.5 to 1.
reserved
clip_truncate
Truncate bits number.

RKNN_dpu_s_status
Address: Operational Base + offset (0x4000)

Bit  Attr  Reset Value

31:18 RO  0x0000

17:16 RO  0x0

15:2  RO  0x0000

1:0  RO  0x0

Description

reserved
status_1
Executer 1 status.
2'd0: Executer 1 is in idle state;
2'd1: Executer 1 is operating;
2'd2: Executer 1 is operating, executer 1 is waiting to operate;
2'd3: Reserved.
reserved
status_0
Executer 0 status.
2'd0: Executer 0 is in idle state;
2'd1: Executer 0 is operating;
2'd2: Executer 0 is operating, executer 1 is waiting to operate;
2'd3: Reserved.

RKNN_dpu_s_pointer
Address: Operational Base + offset (0x4004)

Bit  Attr  Reset Value

31:17 RO  0x0000

16

RO  0x0

15:6  RO  0x000

5

4

W1
C

W1
C

0x0

0x0

Description

reserved
executer
Which register group to be used.
1'd0: Executer group 0;
1'd1: Executer group 1.
reserved
executer_pp_clear
Clear executer group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_clear
Clear register group pointer.
Set this bit to 1 to clear pointer to 0.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2015

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

3

RW  0x0

2

RW  0x0

1

RW  0x0

0

RW  0x0

pointer_pp_mode
Register group ping-pong mode.
1'd0: Pointer ping-pong by executer;
      eg. if current executer is 0, next pointer will toggle to 1;
1'd1: Pointer ping-pong by pointer;
      eg. if current pointer is 0, next pointer will toggle to 1.
executer_pp_en
Executer group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer_pp_en
Register group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer
Which register group ready to be set.
1'd0: Register group 0;
1'd1: Register group 1.

RKNN_dpu_operation_enable
Address: Operational Base + offset (0x4008)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
Set this register will trigger DPU block operate.
This register and after this are all shadowed for ping-pong
operation.
1'd0: Disable;
1'd1: Enable.

RKNN_dpu_feature_mode_cfg
Address: Operational Base + offset (0x400C)

Bit  Attr  Reset Value

Description

31

RW  0x0

30

RW  0x0

29:26 RW  0x0

comb_use
Combine use, same as DPU_RDMA comb_use[0].
tp_en
If enable transpose.
rgp_type
Regroup type.
4'd0: Cut all input (128bit);
4'd1: Cut 4bit;
4'd2: Cut 8bit;
4'd3: Cut 16bit;
4'd4: Cut 32bit;
4'd5: Cut 64bit.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2016

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

25

RW  0x0

24:9  RW  0x0000

8:5  RW  0x0

4:3  RW  0x0

2:1  RW  0x0

0

RW  0x0

nonalign
If non-align mode is enabled.
If the output data flow is the same as the input data flow, this
mode can be used.
surf_len
In non-align mode, how many 8bytes to be stored.
burst_len
Burst length.
4'd3: Burst4;
4'd7: Burst8;
4'd15: Burst16.
conv_mode
Convolution mode.
2'd0: Normal convolution mode;
2'd1: Reserved;
2'd2: Reserved;
2'd3: Depthwise convolution mode.
output_mode
Where the DPU core output goes.
[0]: If the output goes to PPU, high is active;
[1]: If the output goes to outside, high is active.
flying_mode
Flying mode enable.
1'd0: DPU core main data is from convolution output;
1'd1: DPU core main data is from MRDMA;

RKNN_dpu_data_format
Address: Operational Base + offset (0x4010)

Bit  Attr  Reset Value

Description

31:29 RW  0x0

out_precision
Output precision.
3'd0: Integer 8bit;
3'd1: Integer 16bit;
3'd2: Float point 16bit;
3'd3: Bfloat 16bit;
3'd4: Integer 32bit;
3'd5: Float point 32bit;
3'd6: Integer 4bit.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2017

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

28:26 RW  0x0

25:16 RW  0x000

15:10 RW  0x00

9:4  RW  0x00

3

RW  0x0

2:0  RW  0x0

in_precision
Input precision same with DPU_RDMA.
3'd0: Integer 8bit;
3'd1: Integer 16bit;
3'd2: Float point 16bit;
3'd3: Bfloat 16bit;
3'd4: Integer 32bit;
3'd5: Float point 32bit;
3'd6: Integer 4bit.
ew_truncate_neg
Shift value in EW core for negative data.
bn_mul_shift_value_neg
Shift value in BN core for negative data.
bs_mul_shift_value_neg
Shift value in BS core for negative data.
mc_surf_out
How many surfaces serial the DPU output.
1'd0: Output feature obey the rule of 16byte align for one pixel;
1'd1: Output feature can output 2 surface serial or 4 surf serials.
proc_precision
Proc precision.
3'd0: Integer 8bit;
3'd1: Integer 16bit;
3'd2: Float point 16bit;
3'd3: Bfloat 16bit;
3'd4: Integer 32bit;
3'd5: Float point 32bit;
3'd6: Integer 4bit.

RKNN_dpu_offset_pend
Address: Operational Base + offset (0x4014)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:0  RW  0x0000

Description

reserved
offset_pend
What value the extra channel be set.

RKNN_dpu_dst_base_addr
Address: Operational Base + offset (0x4020)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

dst_base_addr
Destination base address.
reserved

RKNN_dpu_dst_surf_stride
Address: Operational Base + offset (0x4024)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2018

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

dst_surf_stride
Output shape surface stride.
reserved

RKNN_dpu_data_cube_width
Address: Operational Base + offset (0x4030)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

reserved
width
Width of the input cube.

Description

RKNN_dpu_data_cube_height
Address: Operational Base + offset (0x4034)

Bit  Attr  Reset Value

31:25 RO  0x00

24:22 RW  0x0

21:13 RO  0x000

12:0  RW  0x0000

Description

reserved
minmax_ctl
Configuration of the minmax op.
[0]: Enable minmax op;
[1]: Minmax type;
[2]: Probability only.
reserved
height
Height of the input cube.

RKNN_dpu_data_cube_notch_addr
Address: Operational Base + offset (0x4038)

Bit  Attr  Reset Value

31:29 RO  0x0

28:16 RW  0x0000

15:13 RO  0x0

12:0  RW  0x0000

Description

reserved
notch_addr_1
How many pixels from the end of width to the end of the shape
line end.
reserved
notch_addr_0
How many pixels from the end of width to the end of the shape
line end.

RKNN_dpu_data_cube_channel
Address: Operational Base + offset (0x403C)

Bit  Attr  Reset Value

31:29 RO  0x0

28:16 RW  0x0000

15:13 RO  0x0

reserved
orig_channel
Original output channel.
reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2019

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

12:0  RW  0x0000

channel
Cube channel.

RKNN_dpu_bs_cfg
Address: Operational Base + offset (0x4040)

Bit  Attr  Reset Value

31:20 RO  0x000

19:16 RW  0x0

15:9  RO  0x00

8

RW  0x0

7

RW  0x0

6

RW  0x0

5

RW  0x0

4

RW  0x0

3:2  RO  0x0

1

RW  0x0

Description

reserved
bs_alu_algo
BS core ALU op type.
4'd0: Reserved;
4'd1: Reserved;
4'd2: Add
4'd3: Reserved;
4'd4: Minus;
4'd5: Reserved;
4'd6: Reserved;
4'd7: Reserved;
4'd8: Reserved.
reserved
bs_alu_src
Where the ALU operand from.
1'd0: From configure register;
1'd1: From outside.
bs_relux_en
If enable RELUX.
1'd0: Disable;
1'd1: Enable.
bs_relu_bypass
If bypass BS core RELU op.
1'd0: Do not bypass;
1'd1: Bypass.
bs_mul_prelu
If enable MUL PRELU.
1'd0: Disable;
1'd1: Enable.
bs_mul_bypass
If bypass BS core MUL op.
1'd0: Do not bypass;
1'd1: Bypass.
reserved
bs_alu_bypass
If bypass BS core ALU op.
1'd0: Do not bypass;
1'd1: Bypass.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2020

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

0

RW  0x0

bs_bypass
If bypass BS core.
1'd0: Do not bypass bs core;
1'd1: Bypass bs core.

RKNN_dpu_bs_alu_cfg
Address: Operational Base + offset (0x4044)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

bs_alu_operand
Bs core ALU operand.

RKNN_dpu_bs_mul_cfg
Address: Operational Base + offset (0x4048)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:14 RO  0x0

13:8  RW  0x00

7:2  RO  0x00

1

RW  0x0

0

RW  0x0

bs_mul_operand
BS core MUL operand.
reserved
bs_mul_shift_value
Shift value in BS core for positive data.
reserved
bs_truncate_src
Where the shift value from.
1'd0: From configure register;
1'd1: From outside.
bs_mul_src
Where the MUL operand from.
1'd0: From configure register;
1'd1: From outside.

RKNN_dpu_bs_relux_cmp_value
Address: Operational Base + offset (0x404C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

bs_relux_cmp_dat
Value of the RELUX compare with.

RKNN_dpu_bs_ow_cfg
Address: Operational Base + offset (0x4050)

Bit  Attr  Reset Value

Description

31:28 RW  0x0

rgp_cnter
Regroup counter.
4'd0: Select all data;
4'd1: Select 1 from every 2;
4'd2: Select 1 from every 4;
4'd3: Select 1 from every 8;
Else: Reserved.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2021

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

27

RW  0x0

26:11 RO  0x0000

10:8  RW  0x0

7:5  RW  0x0

4:2  RW  0x0

1

RW  0x0

0

RW  0x0

tp_org_en
If enable original transpose.
1'd0: Disable;
1'd1: Enable.
reserved
size_e_2
How many 8 channels in a row the last output line (minus 1).
size_e_1
How many 8 channels in a row the middle output line (minus 1).
size_e_0
How many 8 channels in a row the first output line (minus 1).
od_bypass
If bypass CPEND.
1'd0: Do not bypass;
1'd1: Bypass.
ow_src
Where the CPEND operand from.
1'd0: From configure register;
1'd1: From outside.

Description

Description

RKNN_dpu_bs_ow_op
Address: Operational Base + offset (0x4054)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:0  RW  0x0000

reserved
ow_op
CPEND operand.

RKNN_dpu_wdma_size_0
Address: Operational Base + offset (0x4058)

Bit  Attr  Reset Value

31:28 RO  0x0

27

RW  0x0

26:16 RW  0x000

15:13 RO  0x0

12:0  RW  0x0000

reserved
tp_precision
Transpose precision.
1'd0: 8bit;
1'd1: 16bit.
size_c_wdma
Size_c for DPU_WDMA.
reserved
channel_wdma
Channel for DPU_WDMA.

RKNN_dpu_wdma_size_1
Address: Operational Base + offset (0x405C)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2022

Description

Description

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:29 RO  0x0

28:16 RW  0x0000

15:13 RO  0x0

12:0  RW  0x0000

reserved
height_wdma
Height for DPU_WDMA.
reserved
width_wdma
Width for DPU_WDMA.

RKNN_dpu_bn_cfg
Address: Operational Base + offset (0x4060)

Bit  Attr  Reset Value

31:20 RO  0x000

19:16 RW  0x0

15:9  RO  0x00

8

RW  0x0

7

RW  0x0

6

RW  0x0

5

RW  0x0

4

RW  0x0

3:2  RO  0x0

reserved
bn_alu_algo
BS core ALU op type.
4'd0: Reserved;
4'd1: Reserved;
4'd2: Add
4'd3: Reserved;
4'd4: Minus;
4'd5: Reserved;
4'd6: Reserved;
4'd7: Reserved;
4'd8: Reserved.
reserved
bn_alu_src
Where the ALU operand from
1'd0: From configure register;
1'd1: From outside.
bn_relux_en
If enable RELUX.
1'd0: Disable;
1'd1: Enable.
bn_relu_bypass
If bypass BN core RELU op.
1'd0: Do not bypass;
1'd1: Bypass.
bn_mul_prelu
If enable MUL PRELU.
1'd0: Disable;
1'd1: Enable.
bn_mul_bypass
If bypass BN core MUL op.
1'd0: Do not bypass;
1'd1: Bypass.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

2023

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

1

RW  0x0

0

RW  0x0

bn_alu_bypass
If bypass BN core ALU op.
1'd0: Do not bypass;
1'd1: Bypass.
bn_bypass
If bypass BN core.
1'd0: Do not bypass BN core;
1'd1: Bypass BN core.

RKNN_dpu_bn_alu_cfg
Address: Operational Base + offset (0x4064)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

bn_alu_operand
BN core ALU operand.

RKNN_dpu_bn_mul_cfg
Address: Operational Base + offset (0x4068)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:14 RO  0x0

13:8  RW  0x00

7:2  RO  0x00

1

RW  0x0

0

RW  0x0

bn_mul_operand
BN core MUL operand.
reserved
bn_mul_shift_value
Shift value in BN core for positive data.
reserved
bn_truncate_src
Where the shift value from
1'd0: From configure register;
1'd1: From outside.
bn_mul_src
Where the MUL operand from.
1'd0: From configure register;
1'd1: From outside.

RKNN_dpu_bn_relux_cmp_value
Address: Operational Base + offset (0x406C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

bn_relux_cmp_dat
RELUX compare data in BN core.

RKNN_dpu_ew_cfg
Address: Operational Base + offset (0x4070)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2024

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31

RW  0x0

30

RW  0x0

29:28 RW  0x0

27:24 RO  0x0

23:22 RW  0x0

21

RW  0x0

20

RW  0x0

19:16 RW  0x0

15:11 RO  0x00

10

RW  0x0

9

RW  0x0

ew_cvt_type
Convert type of EW input convert when if 0.5
1'd0: Mul first;
1'd1: Add first.
ew_cvt_round
Rounding type of EW input convert when if 0.5
1'd0: If the integer is odd, carry 1;
1'd1: Carry 1 no matter what the integer is.
ew_data_mode
Data mode of the data from ERDMA.
reserved
edata_size
Data size of the cube from ERDMA.
2'd0: 4bit;
2'd1: 8bit;
2'd2: 16bit;
2'd3: 32bit.
ew_equal_en
Min max equal enable.
1'd0: Disable;
1'd1: Enable.
ew_binary_en
Min max binary enable.
1'd0: Disable;
1'd1: Enable.
ew_alu_algo
EW core ALU op type.
4'd0: Max;
4'd1: Min;
4'd2: Add;
4'd3: Div;
4'd4: Minus;
4'd5: Abs;
4'd6: Neg;
4'd7: Floor;
4'd8: Ceil.
reserved
ew_relux_en
If enable RELUX.
1'd0: Disable;
1'd1: Enable.
ew_relu_bypass
If bypass EW core RELU op.
1'd0: Do not bypass;
1'd1: Bypass.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2025

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

8

RW  0x0

7

RW  0x0

6

RW  0x0

5

RW  0x0

4:3  RO  0x0

2

RW  0x0

1

RW  0x0

0

RW  0x0

ew_op_cvt_bypass
If bypass EW input converter.
1'd0: Do not bypass;
1'd1: Bypass.
ew_lut_bypass
If bypass LUT.
1'd0: Do not bypass;
1'd1: Bypass.
ew_op_src
Where the operand from
1'd0: From configure register;
1'd1: From outside.
ew_mul_prelu
If enable MUL PRELU.
1'd0: Disable;
1'd1: Enable.
reserved
ew_op_type
Operator type.
1'd0: ALU;
1'd1: MUL.
ew_op_bypass
If bypass EW core ALU and MUL op.
1'd0: Do not bypass;
1'd1: Bypass.
ew_bypass
If bypass EW core.
1'd0: Do not bypass EW core;
1'd1: Bypass EW core.

RKNN_dpu_ew_cvt_offset_value
Address: Operational Base + offset (0x4074)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_op_cvt_offset
EW convert offset.

RKNN_dpu_ew_cvt_scale_value
Address: Operational Base + offset (0x4078)

Bit  Attr  Reset Value

Description

31:22 RW  0x000

21:16 RW  0x00

15:0  RW  0x0000

ew_truncate
EW core shift value.
ew_op_cvt_shift
EW convert shift value.
ew_op_cvt_scale
EW convert scale.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2026

RKRK3588 TRM-Part1

RKNN_dpu_ew_relux_cmp_value
Address: Operational Base + offset (0x407C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_relux_cmp_dat
EW RELUX compare data.

RKNN_dpu_out_cvt_offset
Address: Operational Base + offset (0x4080)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

out_cvt_offset
Offset of the output converter.

RKNN_dpu_out_cvt_scale
Address: Operational Base + offset (0x4084)

Bit  Attr  Reset Value

31:17 RO  0x0000

16

RW  0x0

15:0  RW  0x0000

Description

reserved
fp32tofp16_en
If enable output from fp32 to fp16.
1'd0: Disable;
1'd1: Enable.
out_cvt_scale
Scale of the output converter.

RKNN_dpu_out_cvt_shift
Address: Operational Base + offset (0x4088)

Bit  Attr  Reset Value

Description

31

RW  0x0

30

RW  0x0

29:20 RO  0x000

19:12 RW  0x00

11:0  RW  0x000

cvt_type
Convert type of out convert when if 0.5.
1'd0: MUL first;
1'd1: ALU first.
cvt_round
Rounding type of out convert when if 0.5.
1'd0: If the integer is odd, carry 1;
1'd1: Carry 1 no matter what the integer is.
reserved
minus_exp
Minus exp of out CVT.
out_cvt_shift
Shift of the output converter.

RKNN_dpu_ew_op_value_0
Address: Operational Base + offset (0x4090)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_0
The 1st EW operand for EW core op.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2027

RKRK3588 TRM-Part1

RKNN_dpu_ew_op_value_1
Address: Operational Base + offset (0x4094)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_1
The 2nd EW operand for EW core op.

RKNN_dpu_ew_op_value_2
Address: Operational Base + offset (0x4098)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_2
the 3thd EW operand for EW core op.

RKNN_dpu_ew_op_value_3
Address: Operational Base + offset (0x409C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_3
the 4th EW operand for EW core op.

RKNN_dpu_ew_op_value_4
Address: Operational Base + offset (0x40A0)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_4
the 5th EW operand for EW core op.

RKNN_dpu_ew_op_value_5
Address: Operational Base + offset (0x40A4)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_5
The 6th EW operand for EW core op.

RKNN_dpu_ew_op_value_6
Address: Operational Base + offset (0x40A8)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_6
The 7th EW operand for EW core op.

RKNN_dpu_ew_op_value_7
Address: Operational Base + offset (0x40AC)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_operand_7
The 8th EW operand for EW core op.

RKNN_dpu_surface_add
Address: Operational Base + offset (0x40C0)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2028

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

surf_add
How many surfaces in a row.
reserved

Description

RKNN_dpu_lut_access_cfg
Address: Operational Base + offset (0x4100)

Bit  Attr  Reset Value

31:18 RO  0x0000

17

RW  0x0

16

RW  0x0

15:10 RO  0x00

9:0  RW  0x000

reserved
lut_access_type
Access type.
1'd0: Read;
1'd1: Write.
lut_table_id
Access Id.
1'd0: LE LUT;
1'd1: LO LUT.
reserved
lut_addr
Access address.

RKNN_dpu_lut_access_data
Address: Operational Base + offset (0x4104)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:0  RW  0x0000

Description

reserved
lut_access_data
Configuration of LUT access data.

RKNN_dpu_lut_cfg
Address: Operational Base + offset (0x4108)

Bit  Attr  Reset Value

31:8  RO  0x000000

7

RW  0x0

6

RW  0x0

5

RW  0x0

Description

reserved
lut_cal_sel
LUT calculate sel.
Only useful when lut_expand_en is 1.
lut_hybrid_priority
LUT hybrid flow priority.
1'd0: LE LUT;
1'd1: LO LUT.
lut_oflow_priority
Priority when over flow happened.
1'd0: LE LUT;
1'd1: LO LUT.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2029

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

4

RW  0x0

3:2  RW  0x0

1

RW  0x0

0

RW  0x0

lut_uflow_priority
Priority when under flow happened.
1'd0: LE LUT;
1'd1: LO LUT.
lut_lo_le_mux
LO LUT and LE LUT mux.
lut_expand_en
If expand two small LUT to a larger LUT.
1'd0: Disable;
1'd1: Enable.
lut_road_sel
LUT road sel.
1'd0: 1st;
1'd1: 2nd.

RKNN_dpu_lut_info
Address: Operational Base + offset (0x410C)

Bit  Attr  Reset Value

31:24 RO  0x00

23:16 RW  0x00

15:8  RW  0x00

7:0  RO  0x00

Description

reserved
lut_lo_index_select
LUT LO index selected.
Used in index generator, choose some bits to be the index.
lut_le_index_select
LUT LE index selected.
Used in index generator, choose some bits to be the index.
reserved

RKNN_dpu_lut_le_start
Address: Operational Base + offset (0x4110)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

lut_le_start
LE LUT start point.

RKNN_dpu_lut_le_end
Address: Operational Base + offset (0x4114)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

lut_le_end
LE LUT end point.

RKNN_dpu_lut_lo_start
Address: Operational Base + offset (0x4118)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

lut_lo_start
LO LUT start point.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2030

RKRK3588 TRM-Part1

RKNN_dpu_lut_lo_end
Address: Operational Base + offset (0x411C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

lut_lo_end
LO LUT end point.

RKNN_dpu_lut_le_slope_scale
Address: Operational Base + offset (0x4120)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:0  RW  0x0000

lut_le_slope_oflow_scale
LE LUT slope scale if over flow.
lut_le_slope_uflow_scale
LE LUT slope scale if under flow.

RKNN_dpu_lut_le_slope_shift
Address: Operational Base + offset (0x4124)

Bit  Attr  Reset Value

31:10 RO  0x000000

9:5  RW  0x00

4:0  RW  0x00

Description

reserved
lut_le_slope_oflow_shift
LE LUT slope shift if over flow.
lut_le_slope_uflow_shift
LE LUT slope shift if under flow.

RKNN_dpu_lut_lo_slope_scale
Address: Operational Base + offset (0x4128)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:0  RW  0x0000

lut_lo_slope_oflow_scale
LO LUT slope scale if over flow.
lut_lo_slope_uflow_scale
LO LUT slope scale if under flow.

RKNN_dpu_lut_lo_slope_shift
Address: Operational Base + offset (0x412C)

Bit  Attr  Reset Value

31:10 RO  0x000000

9:5  RW  0x00

4:0  RW  0x00

Description

reserved
lut_lo_slope_oflow_shift
LO LUT slope shift if over flow.
lut_lo_slope_uflow_shift
LO LUT slope shift if under flow.

RKNN_dpu_rdma_s_status
Address: Operational Base + offset (0x5000)

Bit  Attr  Reset Value

31:18 RO  0x0000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2031

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

17:16 RO  0x0

15:2  RO  0x0000

1:0  RO  0x0

status_1
Executer 1 status.
2'd0: Executer 1 is in idle state;
2'd1: Executer 1 is operating;
2'd2: Executer 1 is operating, executer 1 is waiting to operate;
2'd3: Reserved.
reserved
status_0
Executer 0 status.
2'd0: Executer 0 is in idle state;
2'd1: Executer 0 is operating;
2'd2: Executer 0 is operating, executer 1 is waiting to operate;
2'd3: Reserved.

RKNN_dpu_rdma_s_pointer
Address: Operational Base + offset (0x5004)

Bit  Attr  Reset Value

31:17 RO  0x0000

16

RO  0x0

15:6  RO  0x000

5

4

W1
C

W1
C

0x0

0x0

3

RW  0x0

2

RW  0x0

1

RW  0x0

Description

reserved
executer
Which register group to be used.
1'd0: Executer group 0;
1'd1: Executer group 1.
reserved
executer_pp_clear
Clear executer group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_clear
Clear register group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_mode
Register group ping-pong mode.
1'd0: Pointer ping-pong by executer;
      eg. if current executer is 0, next pointer will toggle to 1;
1'd1: Pointer ping-pong by pointer;
      eg. if current pointer is 0, next pointer will toggle to 1.
executer_pp_en
Executer group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer_pp_en
Register group ping-pong enable.
1'd0: Disable;
1'd1: Enable.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2032

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

0

RW  0x0

pointer
Which register group ready to be set.
1'd0: Register group 0;
1'd1: Register group 1.

RKNN_dpu_rdma_operation_enable
Address: Operational Base + offset (0x5008)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
Set this register will trigger DPU_RDMA block operate.
This register and after this are all shadowed for ping-pong
operation.
1'd0: Disable;
1'd1: Enable.

RKNN_dpu_rdma_data_cube_width
Address: Operational Base + offset (0x500C)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
width
Input feature width (need to minus 1).

RKNN_dpu_rdma_data_cube_height
Address: Operational Base + offset (0x5010)

Bit  Attr  Reset Value

31:29 RO  0x0

28:16 RW  0x0000

15:13 RO  0x0

12:0  RW  0x0000

Description

reserved
ew_line_notch_addr
Line notch of EW.
reserved
height
Input feature height (need to minus 1).

RKNN_dpu_rdma_data_cube_channel
Address: Operational Base + offset (0x5014)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
channel
Input feature channel (need to minus 1).

RKNN_dpu_rdma_src_base_addr
Address: Operational Base + offset (0x5018)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2033

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

src_base_addr
Fly mode source address.

RKNN_dpu_rdma_brdma_cfg
Address: Operational Base + offset (0x501C)

Bit  Attr  Reset Value

31:5  RO  0x0000000

4:1  RW  0x0

0

RO  0x0

Description

reserved
brdma_data_use
How many data type need to be read.
[0]: If read ALU operand, set 1 to enable;
[1]: If read CPEND operand, set 1 to enable;
[2]: If read MUL operand, set 1 to enable;
[3]: If read TRT operand, set 1 to enable.
reserved

RKNN_dpu_rdma_bs_base_addr
Address: Operational Base + offset (0x5020)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

bs_base_addr
Base address to read BS ALU, BS CPEND, BS MUL operand.

RKNN_dpu_rdma_nrdma_cfg
Address: Operational Base + offset (0x5028)

Bit  Attr  Reset Value

31:5  RO  0x0000000

4:1  RW  0x0

0

RO  0x0

Description

reserved
nrdma_data_use
How many data type need to be read.
[0]: If read ALU operand, set 1 to enable;
[1]: If read CPEND operand, set 1 to enable, (tie to 0, cause BN
do not have CPEND);
[2]: If read MUL operand, set 1 to enable;
[3]: If read TRT operand, set 1 to enable.
reserved

RKNN_dpu_rdma_bn_base_addr
Address: Operational Base + offset (0x502C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

bn_base_addr
Base address to read BN ALU, BN MUL operand.

RKNN_dpu_rdma_erdma_cfg
Address: Operational Base + offset (0x5034)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2034

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:30 RW  0x0

29

RW  0x0

28

RW  0x0

27:4  RO  0x000000

3:2  RW  0x0

1

RW  0x0

0

RW  0x0

erdma_data_mode
If the read data is per channel or per pixel
2'd0: Per channel;
2'd1: Per pixel;
2'd2: Per channel by pixel;
2'd3: Reserved.
erdma_surf_mode
Surface mode of the EW cube.
1'd0: 1 surface series;
1'd1: 2 surface series.
erdma_nonalign
If read the EW cube in non-align mode.
1'd0: Do not use non-align mode;
1'd1: Use non-align mode.
reserved
erdma_data_size
What the precision of the cube that the ERDMA read.
2'd0: 4bit;
2'd1: 8bit;
2'd2: 16bit;
2'd3: 32bit.
ov4k_bypass
Separate the burst command of over 4k to 2 independent burst
commands.
1'd0: Enable this feature;
1'd1: Bypass this feature.
erdma_disable
If disable ERDMA.
1'd0: Do not disable ERDMA;
1'd1: Disable ERDMA.

RKNN_dpu_rdma_ew_base_addr
Address: Operational Base + offset (0x5038)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

ew_base_addr
Base address to read EW operand.

RKNN_dpu_rdma_ew_surf_stride
Address: Operational Base + offset (0x5040)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

ew_surf_stride
The surface stride of the element wise feature map;
If erdma_data_mode is per channel, it need set to be 1.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

2035

RKRK3588 TRM-Part1

RKNN_dpu_rdma_feature_mode_cfg
Address: Operational Base + offset (0x5044)

Bit  Attr  Reset Value

31:18 RO  0x0000

17:15 RW  0x0

14:11 RW  0x0

10:8  RW  0x0

7:5  RW  0x0

4

RW  0x0

3

RW  0x0

2:1  RW  0x0

Description

reserved
in_precision
Input data precision.
3'd0: Integer 8bit;
3'd1: Integer 16bit;
3'd2: Gloat point 16bit;
3'd3: Bfloat 16bit;
3'd4: Integer 32bit;
3'd5: Float point 32bit;
3'd6: Integer 4bit.
burst_len
Burst length.
4'd3: Burst4;
4'd7: Burst8;
4'd15: Burst16.
comb_use
[0]: If enable MRDMA and ERDMA to read the same data;
[1]: Read the data to MRDMA;
[2]: Read the data to ERDMA.
proc_precision
Process precision.
3'd0: Integer 8bit;
3'd1: Integer 16bit;
3'd2: Float point 16bit;
3'd3: Bfloat 16bit;
3'd4: Integer 32bit;
3'd5: Float point 32bit;
3'd6: Integer 4bit.
mrdma_disable
If disable MRDMA.
1'd0: Do not disable MRDMA;
1'd1: Disable MRDMA.
mrdma_fp16tofp32_en
If enable DPU input from fp16 to fp32.
conv_mode
Convolution mode.
2'd0: Dc;
2'd1: Reserved;
2'd2: Reserved;
2'd3: Depthwise.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2036

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

0

RW  0x0

flying_mode
Flying mode enable.
1'd0: DPU core main data is from convolution output;
1'd1: DPU core main data is from MRDMA.

RKNN_dpu_rdma_src_dma_cfg
Address: Operational Base + offset (0x5048)

Bit  Attr  Reset Value

Description

31:19 RW  0x0000

18:14 RO  0x00

13

RW  0x0

12

RW  0x0

11:9  RW  0x0

8:6  RW  0x0

5:3  RW  0x0

2:0  RW  0x0

line_notch_addr
How many pixels from the end of width end to the shape feature
line end.
reserved
pooling_method
Pooling method
1'd0: Average pooling (up sampling can use this mode);
1'd1: Min or max pooling.
unpooling_en
If enable un-pooling.
kernel_stride_height
Un-pooling kernel stride height (minus 1).
kernel_stride_width
Un-pooling kernel stride width (minus 1).
kernel_height
Un-pooling kernel height (minus 1).
kernel_width
Un-pooling kernel width (minus 1).

RKNN_dpu_rdma_surf_notch
Address: Operational Base + offset (0x504C)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

surf_notch_addr
How many pixels from the end of this process feature map to the
end of the shape feature map.
reserved

RKNN_dpu_rdma_pad_cfg
Address: Operational Base + offset (0x5064)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:7  RO  0x000

6:4  RW  0x0

3

RO  0x0

pad_value
Pad value.
reserved
pad_top
Un-pooling top pad.
reserved

Copyright 2022 © Rockchip Electronics Co., Ltd.

2037

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

2:0  RW  0x0

pad_left
Un-pooling left pad.

RKNN_dpu_rdma_weight
Address: Operational Base + offset (0x5068)

Bit  Attr  Reset Value

Description

31:24 RW  0x00

23:16 RW  0x00

15:8  RW  0x00

7:0  RW  0x00

e_weight
The arbiter weight for ERDMA.
n_weight
The arbiter weight for NRDMA.
b_weight
The arbiter weight for BRDMA.
m_weight
The arbiter weight for MRDMA.

RKNN_dpu_rdma_ew_surf_notch
Address: Operational Base + offset (0x506C)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

ew_surf_notch
Surface notch of EW.
reserved

RKNN_ppu_s_status
Address: Operational Base + offset (0x6000)

Bit  Attr  Reset Value

31:18 RO  0x0000

17:16 RO  0x0

15:2  RO  0x0000

1:0  RO  0x0

Description

reserved
status_1
Executer 1 status.
2'd0: Executer 1 is in idle state;
2'd1: Executer 1 is operating;
2'd2: Executer 1 is operating, executer 1 is waiting to operate;
2'd3: Reserved.
reserved
status_0
Executer 0 status.
2'd0: Executer 0 is in idle state;
2'd1: Executer 0 is operating;
2'd2: Executer 0 is operating, executer 1 is waiting to operate;
2'd3: Reserved.

RKNN_ppu_s_pointer
Address: Operational Base + offset (0x6004)

Bit  Attr  Reset Value

31:17 RO  0x0000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2038

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

16

RO  0x0

15:6  RO  0x000

5

4

W1
C

W1
C

0x0

0x0

3

RW  0x0

2

RW  0x0

1

RW  0x0

0

RW  0x0

executer
Which register group to be used.
1'd0: Executer group 0;
1'd1: Executer group 1.
reserved
executer_pp_clear
Clear executer group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_clear
Clear register group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_mode
Register group ping-pong mode.
1'd0: Pointer ping-pong by executer;
      eg. if current executer is 0, next pointer will toggle to 1;
1'd1: Pointer ping-pong by pointer;
      eg. if current pointer is 0, next pointer will toggle to 1.
executer_pp_en
Executer group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer_pp_en
Register group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer
Which register group ready to be set.
1'd0: Register group 0;
1'd1: Register group 1.

RKNN_ppu_operation_enable
Address: Operational Base + offset (0x6008)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
Set this register will trigger PPU block operate.
This register and after this are all shadowed for ping-pong
operation.
1'd0: Disable;
1'd1: Enable.

RKNN_ppu_data_cube_in_width
Address: Operational Base + offset (0x600C)

Bit  Attr  Reset Value

31:13 RO  0x00000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2039

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

12:0  RW  0x0000

cube_in_width
Pooling cube width (need to minus 1).

RKNN_ppu_data_cube_in_height
Address: Operational Base + offset (0x6010)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_in_height
Pooling cube height (need to minus 1).

RKNN_ppu_data_cube_in_channel
Address: Operational Base + offset (0x6014)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_in_channel
Pooling cube channel (need to minus 1).

RKNN_ppu_data_cube_out_width
Address: Operational Base + offset (0x6018)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_out_width
Pooling output cube width (need to minus 1).

RKNN_ppu_data_cube_out_height
Address: Operational Base + offset (0x601C)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_out_height
Pooling output cube height (need to minus 1).

RKNN_ppu_data_cube_out_channel
Address: Operational Base + offset (0x6020)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_out_channel
Pooling output cube channel (need to minus 1).

RKNN_ppu_operation_mode_cfg
Address: Operational Base + offset (0x6024)

Bit  Attr  Reset Value

31

RO  0x0

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2040

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

30

RW  0x0

29

RO  0x0

28:16 RW  0x0000

15:8  RO  0x00

7:5  RW  0x0

4

RW  0x0

3:2  RO  0x0

1:0  RW  0x0

index_en
If enable output the position of each kernel.
1'd0: Disable;
1'd1: Enable.
reserved
notch_addr
How many pixels from the end of the width end to the shape line
end.
reserved
use_cnt
Use_cnt.
flying_mode
Where the pooling cube from.
1'd0: DPU;
1'd1: Outside.
reserved
pooling_method
Pooling method.
2'd0: Average pooling;
2'd1: Max pooling;
2'd2: Min pooling;
2'd3: Reserved.

RKNN_ppu_pooling_kernel_cfg
Address: Operational Base + offset (0x6034)

Bit  Attr  Reset Value

Description

31:24 RO  0x00

23:20 RW  0x0

19:16 RW  0x0

15:12 RO  0x0

11:8  RW  0x0

7:4  RO  0x0

3:0  RW  0x0

reserved
kernel_stride_height
Pooling kernel stride height (need to minus 1).
kernel_stride_width
Pooling kernel stride width (need to minus 1).
reserved
kernel_height
Pooling kernel height (need to minus 1).
reserved
kernel_width
Pooling kernel width (need to minus 1).

RKNN_ppu_recip_kernel_width
Address: Operational Base + offset (0x6038)

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0  RW  0x00000

Description

reserved
recip_kernel_width
The Reciprocal of the shape kernel width multiple 2^16.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2041

RKRK3588 TRM-Part1

RKNN_ppu_recip_kernel_height
Address: Operational Base + offset (0x603C)

Bit  Attr  Reset Value

31:17 RO  0x0000

16:0  RW  0x00000

Description

reserved
recip_kernel_height
The Reciprocal of the shape kernel height multiple 2^16.

Description

RKNN_ppu_pooling_padding_cfg
Address: Operational Base + offset (0x6040)

Bit  Attr  Reset Value

31:15 RO  0x00000

14:12 RW  0x0

11

RO  0x0

10:8  RW  0x0

7

RO  0x0

6:4  RW  0x0

3

RO  0x0

2:0  RW  0x0

reserved
pad_bottom
Pooling bottom pad.
reserved
pad_right
Pooling right pad.
reserved
pad_top
Pooling top pad.
reserved
pad_left
Pooling left pad.

RKNN_ppu_padding_value_1_cfg
Address: Operational Base + offset (0x6044)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

pad_value_0
Pad_value*1 [31:0].

RKNN_ppu_padding_value_2_cfg
Address: Operational Base + offset (0x6048)

Bit  Attr  Reset Value

31:3  RO  0x00000000  reserved

2:0  RW  0x0

pad_value_1
Pad_value*1 [34:32].

Description

RKNN_ppu_dst_base_addr
Address: Operational Base + offset (0x6070)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

dst_base_addr
Base address the output cube goes.
reserved

RKNN_ppu_dst_surf_stride
Address: Operational Base + offset (0x607C)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2042

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

dst_surf_stride
Output shape area.
reserved

RKNN_ppu_data_format
Address: Operational Base + offset (0x6084)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3

RW  0x0

2:0  RW  0x0

index_add
If index_en enable, this register is dst_surface_stride x number
of cube surface (every 8bytes per surface), else it equals to
dst_surface_stride.
dpu_flyin
If the data from DPU, and DPU data is from outside, this bit set to
be 1.
proc_precision
Process precision.

RKNN_ppu_misc_ctrl
Address: Operational Base + offset (0x60DC)

Bit  Attr  Reset Value

Description

31:16 RW  0x0000

15:9  RO  0x00

8

RW  0x0

7

RW  0x0

6:4  RO  0x0

3:0  RW  0x0

surf_len
Surface count length.
reserved
mc_surf_out
If enable multiple surfaces out.
1'd0: Disable;
1'd1: Enable.
nonalign
If enable non-align mode.
1'd0: Disable;
1'd1: Enable.
reserved
burst_len
Burst length
4'd3: Burst4;
4'd7: Burst8;
4'd15: Burst16.

RKNN_ppu_rdma_s_status
Address: Operational Base + offset (0x7000)

Bit  Attr  Reset Value

31:18 RO  0x0000

reserved

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2043

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

17:16 RO  0x0

15:2  RO  0x0000

1:0  RO  0x0

status_1
Executer 1 status.
2'd0: Executer 1 is in idle state;
2'd1: Executer 1 is operating;
2'd2: Executer 1 is operating, executer 1 is waiting to operate;
2'd3: Reserved.
reserved
status_0
Executer 0 status.
2'd0: Executer 0 is in idle state;
2'd1: Executer 0 is operating;
2'd2: Executer 0 is operating, executer 1 is waiting to operate;
2'd3: Reserved.

RKNN_ppu_rdma_s_pointer
Address: Operational Base + offset (0x7004)

Bit  Attr  Reset Value

31:17 RO  0x0000

16

RO  0x0

15:6  RO  0x000

5

4

W1
C

W1
C

0x0

0x0

3

RW  0x0

2

RW  0x0

1

RW  0x0

Description

reserved
executer
Which register group to be used.
1'd0: Executer group 0;
1'd1: Executer group 1.
reserved
executer_pp_clear
Clear executer group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_clear
Clear register group pointer.
Set this bit to 1 to clear pointer to 0.
pointer_pp_mode
Register group ping-pong mode.
1'd0: Pointer ping-pong by executer;
      eg. if current executer is 0, next pointer will toggle to 1;
1'd1: Pointer ping-pong by pointer;
      eg. if current pointer is 0, next pointer will toggle to 1.
executer_pp_en
Executer group ping-pong enable.
1'd0: Disable;
1'd1: Enable.
pointer_pp_en
Register group ping-pong enable.
1'd0: Disable;
1'd1: Enable.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2044

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

0

RW  0x0

pointer
Which register group ready to be set.
1'd0: Register group 0;
1'd1: Register group 1.

RKNN_ppu_rdma_operation_enable
Address: Operational Base + offset (0x7008)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

Description

0

RW  0x0

op_en
Set this register will trigger PPU_RDMA block operate.
This register and after this are all shadowed for ping-pong
operation.
1'd0: Disable;
1'd1: Enable.

RKNN_ppu_rdma_cube_in_width
Address: Operational Base + offset (0x700C)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_in_width
Pooling cube width (need to minus 1).

RKNN_ppu_rdma_cube_in_height
Address: Operational Base + offset (0x7010)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_in_height
Pooling cube height (need to minus 1).

RKNN_ppu_rdma_cube_in_channel
Address: Operational Base + offset (0x7014)

Bit  Attr  Reset Value

31:13 RO  0x00000

12:0  RW  0x0000

Description

reserved
cube_in_channel
Pooling cube channel (need to minus 1).

RKNN_ppu_rdma_src_base_addr
Address: Operational Base + offset (0x701C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

src_base_addr
Base address of the pooling cube.

RKNN_ppu_rdma_src_line_stride
Address: Operational Base + offset (0x7024)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2045

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

src_line_stride
Pooling cube shape width.
reserved

RKNN_ppu_rdma_src_surf_stride
Address: Operational Base + offset (0x7028)

Bit  Attr  Reset Value

Description

31:4  RW  0x0000000

3:0  RO  0x0

src_surf_stride
Pooling cube shape area.
reserved

RKNN_ppu_rdma_data_format
Address: Operational Base + offset (0x7030)

Bit  Attr  Reset Value

31:2  RO  0x00000000  reserved

Description

1:0  RW  0x0

in_precision
Input precision.
2'd0: 4bit;
2'd1: 8bit;
2'd2: 16bit;
2'd3: 32bit.

RKNN_ddma_cfg_outstanding
Address: Operational Base + offset (0x8000)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:8  RW  0x00

7:0  RW  0x00

Description

reserved
wr_os_cnt
Max numbers of write outstanding.
rd_os_cnt
Max numbers of read outstanding.

RKNN_ddma_rd_weight_0
Address: Operational Base + offset (0x8004)

Bit  Attr  Reset Value

Description

31:24 RW  0x00

23:16 RW  0x00

15:8  RW  0x00

7:0  RW  0x00

rd_weight_pdp
Weight of PPU read burst.
rd_weight_dpu
Weight of DPU read burst.
rd_weight_kernel
Weight of read weight burst.
rd_weight_feature
Weight of read feature burst.

RKNN_ddma_wr_weight_0
Address: Operational Base + offset (0x8008)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2046

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:16 RO  0x0000

15:8  RW  0x00

7:0  RW  0x00

reserved
wr_weight_pdp
Write_weight_ppu.
wr_weight_dpu
Write_weight_dpu.

RKNN_ddma_cfg_id_error
Address: Operational Base + offset (0x800C)

Bit  Attr  Reset Value

31:10 RO  0x000000

9:6  RW  0x0

5

RO  0x0

4:0  RW  0x00

reserved
wr_resp_id
Error write id.
reserved
rd_resp_id
Error read id.

Description

Description

RKNN_ddma_rd_weight_1
Address: Operational Base + offset (0x8010)

Bit  Attr  Reset Value

31:8  RO  0x000000

7:0  RW  0x00

reserved
rd_weight_pc
Weight of PC read burst.

Description

Description

Description

RKNN_ddma_cfg_dma_fifo_clr
Address: Operational Base + offset (0x8014)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

0

RW  0x0

dma_fifo_clr
Clear DMA FIFO.

RKNN_ddma_cfg_dma_arb
Address: Operational Base + offset (0x8018)

Bit  Attr  Reset Value

31:10 RO  0x000000

9

8

7

RW  0x0

RW  0x0

RO  0x0

6:4  RW  0x0

3

RO  0x0

2:0  RW  0x0

reserved
wr_arbit_model
Write_arbit_model.
rd_arbit_model
Read_arbit_model.
reserved
wr_fix_arb
Write_fix_arb.
reserved
rd_fix_arb
Read_fix_arb.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2047

Description

Description

Description

RKRK3588 TRM-Part1

RKNN_ddma_cfg_dma_rd_qos
Address: Operational Base + offset (0x8020)

Bit  Attr  Reset Value

31:10 RO  0x000000

9:8  RW  0x0

7:6  RW  0x0

5:4  RW  0x0

3:2  RW  0x0

1:0  RW  0x0

reserved
rd_pc_qos
Read_pc_qos.
rd_ppu_qos
Read_ppu_qos.
rd_dpu_qos
Read_dpu_qos.
rd_kernel_qos
Read_kernel_qos.
rd_feature_qos
Read feature_qos.

RKNN_ddma_cfg_dma_rd_cfg
Address: Operational Base + offset (0x8024)

Bit  Attr  Reset Value

31:13 RO  0x00000

12

RW  0x0

11:8  RW  0x0

7:5  RW  0x0

4:3  RW  0x0

2:0  RW  0x0

reserved
rd_arlock
Read_arlock.
rd_arcache
Read_arcache.
rd_arprot
Read_arprot.
rd_arburst
Read_arburst.
rd_arsize
Read_arsize.

RKNN_ddma_cfg_dma_wr_cfg
Address: Operational Base + offset (0x8028)

Bit  Attr  Reset Value

31:13 RO  0x00000

12

RW  0x0

11:8  RW  0x0

7:5  RW  0x0

4:3  RW  0x0

2:0  RW  0x0

reserved
wr_awlock
Write_awlock.
wr_awcache
Write awcache.
wr_awprot
Write_awprot.
wr_awburst
Write_awburst.
wr_awsize
Write_awsize.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2048

RKRK3588 TRM-Part1

RKNN_ddma_cfg_dma_wstrb
Address: Operational Base + offset (0x802C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

wr_wstrb
Write_wstrb.

RKNN_ddma_cfg_status
Address: Operational Base + offset (0x8030)

Bit  Attr  Reset Value

31:9  RO  0x000000

8

RW  0x0

7:0  RO  0x00

reserved
idel
Idel.
reserved

Description

RKNN_sdma_cfg_outstanding
Address: Operational Base + offset (0x9000)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:8  RW  0x00

7:0  RW  0x00

Description

reserved
wr_os_cnt
Max numbers of write outstanding.
rd_os_cnt
Max numbers of read outstanding.

RKNN_sdma_rd_weight_0
Address: Operational Base + offset (0x9004)

Bit  Attr  Reset Value

Description

31:24 RW  0x00

23:16 RW  0x00

15:8  RW  0x00

7:0  RW  0x00

rd_weight_pdp
Weight of PPU read burst.
rd_weight_dpu
Weight of DPU read burst.
rd_weight_kernel
Weight of read weight burst.
rd_weight_feature
Weight of read feature burst.

Description

RKNN_sdma_wr_weight_0
Address: Operational Base + offset (0x9008)

Bit  Attr  Reset Value

31:16 RO  0x0000

15:8  RW  0x00

7:0  RW  0x00

reserved
wr_weight_pdp
Write_weight_ppu.
wr_weight_dpu
Write_weight_dpu.

RKNN_sdma_cfg_id_error
Address: Operational Base + offset (0x900C)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2049

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:10 RO  0x000000

9:6  RW  0x0

5

RO  0x0

4:0  RW  0x00

reserved
wr_resp_id
Error write id.
reserved
rd_resp_id
Error read id.

Description

RKNN_sdma_rd_weight_1
Address: Operational Base + offset (0x9010)

Bit  Attr  Reset Value

31:8  RO  0x000000

7:0  RW  0x00

reserved
rd_weight_pc
Weight of PC read burst.

Description

RKNN_sdma_cfg_dma_fifo_clr
Address: Operational Base + offset (0x9014)

Bit  Attr  Reset Value

31:1  RO  0x00000000  reserved

0

RW  0x0

dma_fifo_clr
Clear DMA FIFO.

RKNN_sdma_cfg_dma_arb
Address: Operational Base + offset (0x9018)

Bit  Attr  Reset Value

31:10 RO  0x000000

9

8

7

RW  0x0

RW  0x0

RO  0x0

6:4  RW  0x0

3

RO  0x0

2:0  RW  0x0

reserved
wr_arbit_model
Write_arbit_model.
rd_arbit_model
Read_arbit_model.
reserved
wr_fix_arb
Write_fix_arb.
reserved
rd_fix_arb
Read_fix_arb.

RKNN_sdma_cfg_dma_rd_qos
Address: Operational Base + offset (0x9020)

Bit  Attr  Reset Value

31:10 RO  0x000000

9:8  RW  0x0

7:6  RW  0x0

reserved
rd_pc_qos
Read_pc_qos.
rd_ppu_qos
Read_ppu_qos.

Description

Description

Description

Copyright 2022 © Rockchip Electronics Co., Ltd.

2050

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

Description

5:4  RW  0x0

3:2  RW  0x0

1:0  RW  0x0

rd_dpu_qos
Read_dpu_qos.
rd_kernel_qos
Read_kernel_qos.
rd_feature_qos
Read feature_qos.

RKNN_sdma_cfg_dma_rd_cfg
Address: Operational Base + offset (0x9024)

Bit  Attr  Reset Value

31:13 RO  0x00000

12

RW  0x0

11:8  RW  0x0

7:5  RW  0x0

4:3  RW  0x0

2:0  RW  0x0

reserved
rd_arlock
Read_arlock.
rd_arcache
Read_arcache.
rd_arprot
Read_arprot.
rd_arburst
Read_arburst.
rd_arsize
Read_arsize.

RKNN_sdma_cfg_dma_wr_cfg
Address: Operational Base + offset (0x9028)

Bit  Attr  Reset Value

31:13 RO  0x00000

12

RW  0x0

11:8  RW  0x0

7:5  RW  0x0

4:3  RW  0x0

2:0  RW  0x0

reserved
wr_awlock
Write_awlock.
wr_awcache
Write awcache.
wr_awprot
Write_awprot.
wr_awburst
Write_awburst.
wr_awsize
Write_awsize.

Description

Description

RKNN_sdma_cfg_dma_wstrb
Address: Operational Base + offset (0x902C)

Bit  Attr  Reset Value

Description

31:0  RW  0x00000000

wr_wstrb
Write_wstrb.

RKNN_sdma_cfg_status
Address: Operational Base + offset (0x9030)

Copyright 2022 © Rockchip Electronics Co., Ltd.

2051

RKRK3588 TRM-Part1

Bit  Attr  Reset Value

31:9  RO  0x000000

8

RW  0x0

7:0  RO  0x00

reserved
idel
Idel.
reserved

RKNN_global_operation_enable
Address: Operational Base + offset (0xF008)

Description

Bit  Attr  Reset Value

31:7  RO  0x0000000

6

5

4

3

2

1

0

RW  0x0

RW  0x0

RW  0x0

RW  0x0

RW  0x0

RO  0x0

RW  0x0

Description

reserved
ppu_rdma_op_en
PPU_RDMA operation enable signal.
ppu_op_en
PPU operation enable signal.
dpu_rdma_op_en
DPU_RDMA operation enable signal.
dpu_op_en
DPU operation enable signal.
core_op_en
CORE operation enable signal.
reserved
cna_op_en
CNA operation enable signal.

36.5 Application Notes

36.5.1 Ping-pong registers
In order to reduce the time of fetch registers, every Calculate Core and Control Core of
RKNN has its owner ping-pong registers. Configure the group 0 when use the group 1, and
configure the group 1 when use group 0. As a result of hiding the time of fetch registers.
Writing S_POINTER of every block can enable this function.
36.5.2 Clock and Reset
1.5.2.1 Clock Domains
RKNN has two clock domains, one is AHB clock, the other is AXI clock. AHB clock, which is
the clock for AHB interface, while AXI clock, which is the clock for AXI interface. AXI clock
also used for core clock for every Calculate Core and Control Core. Clock frequency can be
controlled by CRU, please refer to the relevant sections. Automatic localized clock gating is
employed throughout the design in order to minimize the dynamic power consumption.
Almost all of the flip-flops are clock gated in the design. Block level clock gating also
implemented in every separate block. If a block and the interface to the block are both idle,
then the clock of that block will be gated automatically. This feature can be disabled by
software.
1.5.2.2 NPU Reset
Correspond to the clock domain, there are two reset signals. Aresetn, the reset signal for
AXI interface and every Calculate Core and Control Core. Hresetn is the AHB interface reset
pin and which is synchronized to the AHB clock domain.
All the two signals must be asserted for a minimum of 32 core clock cycles, using the
slowest of the two clocks. Then two signals must be release at the same time.
36.5.3 NPU Interrupt Application

Copyright 2022 © Rockchip Electronics Co., Ltd.

2052

RKRK3588 TRM-Part1

RKNN has 3 interrupt output signal and it remains asserted until the host processor clears
the interrupt. Each bit of PC_INTERRUPT_STATUS represents one of the 17 possible events
that the RKNN can signal to the host processor. By setting the bits of the interrupt enable
register (PC_INTERRUPT_MASK) the programmer can control which of those events will
generate an interrupt.
36.5.4 NPU operate flow
RKNN has two types of work mode: slave configured mode and pc work mode. Pc work
mode need to initial register information to the system memory, then obey flowing flows as
Figure 1-3 shows.
The information of registers initial to the system memory need to generate by the op you
want to specify. And we describe the normal work flows by describe the slave configured
mode.

Fig. 36-3 PC flow

37  Convolution flow

RKNN supports variety of convolution and matrix multiplication. If use the MAC
array to calculate, the work flow should obey the Figure 1-4, and if use DPU CORE
calculate, work flow should obey the Figure 1-5.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2053

Write pc_amount/pc_addr/ pc_task_number/pc_task_dma_baseInitial register information in system memoryStartWrite pc_interrupt_clear and pc_interrupt_maskWrite pc_op_enable to start a group of tasksEnd

RKRK3588 TRM-Part1

Fig. 36-4 Convolution flow 1

In step 1, do not set PPU mask bit if don’t use PPU.

In step 3, you need to configure the size of feature map, and the input precision,
process precision, grains, entries and weight size, including weight height, weight
width, kernels and weight bytes etc.

In step 4, you need to configure process precision, conv_mode and auto gating registers of
CMAC and CNA sequence controller.
In step 5, you can specify some operators to the outputs of convolution according to
configure the DPU_CORE. DPU CORE has some fixed operators, also you can specify some
programmed operators.

In step 6, you can do some extra operators to the outputs of DPU, pooling for
example.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2054

StartWrite pc_interrupt_clear and pc_interrupt_maskWrite Pointer register of every module decide to use to enable ping-pong register or not Write CNA register to configure the information of fetch dataWrite CNA register to configure the kind , size, input precision proc_precision of CNA COREWrite DPU register to configure the extra Ops decide to do and the output modeDo some pipeline surface opsWrite PPU register to configure the extra Ops decide to do and the output modeWrite register to enable NPUEndYN

RKRK3588 TRM-Part1

Fig. 36-5 Convolution flow 2

In step 3, you need to configure the size of feature map, and the input precision,

process precision, grains, entries and weight size, including height, width, kernels
and weight bytes etc. Here we supply zero skipping switch, if there’s a lot of zero or
some number else in the feature map, you can enable zero skipping by writing
cna_fc_con0, cna_fc_con1 and cna_fc_con2, as a result of without reading the
weights correspond to the pixel in the feature map. In zero skipping mode, you can
specify the feature map size for read different from the feature map size for
calculate.

If enable zero skipping, the conv_mode of DPU must configure to be 3, and must bypass

the BS_CORE. The alu_src must set to be 0, the mul_src must set to be 1, and the
alu_algo must set be 3. We bypass BS_CORE, and use BN_CORE to calculate
convolution, so the extra operators have to achieve by EW_CORE. But we use
NRDMA to read operands to EW_CORE when ew_src is 1.

38  Pooling flow
You can use PPU in two ways, pipeline with DPU, and flying mode independently.The first

one is described above, we describe the second flow below.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2055

StartWrite pc_interrupt_clear and pc_interrupt_maskWrite Pointer register of every module decide to use to enable ping-pong register or not Write CNA register to configure the information of fetch dataWrite DPU register to configure the extra Ops decide to do and the output modeDo some pipeline surface opsWrite PPU register to configure the extra Ops decide to do and the output modeWrite register to enable NPUEndYN

RKRK3588 TRM-Part1

Fig. 36-6 Pooling flow

In step 4, we supply two type of output mode, 8byte align per pixel and non-align
mode. In some case with uncomfortable feature map size, you can enable non-align
mode by writing ppu_misc_ctrl_nonalign and ppu_misc_ctrl_surf_len.

39  Separate op flow
As for the operators that RKNN do not fixed, you can combine DPU and PPU to achieve

it. Following we describe the DPU flying mode flow.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2056

StartWrite pc_interrupt_clear and pc_interrupt_maskWrite Pointer register of every module decide to use to enable ping-pong register or not Write PPU register to configure the information of fetch dataWrite PPU register to configure the extra Ops decide to do and the output modeWrite register to enable NPUEnd

RKRK3588 TRM-Part1

Fig. 36-7 DPU flying mode flow

In step 3, you can enable MRDMA, BRDMA, NRDMA, ERDMA to fetch data. RKNN
supports 6 types of data input precision.
In step 4, you can configure BS_ALU_BYPASS, BS_MUL_BYPASS, BS_RELU_BYPASS,

BS_MUL_PRELU, BS_MUL_SHIFT, BS_RELUX_EN etc. to specify the operators you
want to achieve.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2057

StartWrite PC_interrupt_clear and PC_interrupt_maskWrite Pointer register of every module decide to use to enable ping-pong register or not Write DPU_RDMA register to configure the information of fetch dataWrite DPU register to configure the extra Ops decide to do and the output modeWrite register to enable NPUEnd
RKRK3588 TRM-Part1

Chapter 37 Video Capture (VICAP)

37

37.1 Overview

The Video Capture, receives the data from Camera via DVP/MIPI, and transfers the data into
system main memory by AXI bus.
⚫  Support BT601 YCbCr 422 8bit input, RAW 8/10/12bit input
⚫  Support BT656 YCbCr 422 8bit progressive/interlace input
⚫  Support BT1120 YCbCr 422 16bit progressive/interlace input, single/dual-edge sampling
⚫  Support 2/4 channels mixed BT656/BT1120 YCbCr 422 8/16bit progressive/interlace

input

⚫  Support YUYV sequence configurable
⚫  Support the polarity of hsync and vsync configurable
⚫  Support receiving six interfaces of MIPI CSI/DSI, up to four IDs for each interface
⚫  Support five CSI data formats: RAW8/10/12/14, YUV422
⚫  Support three modes of HDR: virtual channel mode, identification code mode, line

counter mode

⚫  Support window cropping
⚫  Support RAW data through to ISP0/1
⚫  Support four channels of 8/16/32 times down-sampling for RAW data
⚫  Support virtual stride when write to DDR
⚫  Support NV16/NV12/YUV400/YUYV output format for YUV data
⚫  Support compact/non-compact output format for RAW data
⚫  Support MMU
⚫  Support soft reset, auto-reset when DMA error

37.2 Block Diagram

Fig. 37-1 VICAP Block Diagram

VICAP comprises with:
⚫  AHB Slave
⚫  AXI Master
⚫  MMU
⚫
⚫  CROP
⚫  SCALE
⚫  TOISP0/1
⚫  DMA

INTERFACE

37.3 Function Description

37.3.1 Interface
Translate the input video data(DVP/MIPI CSI) into the requisite data format
⚫  DVP BT656/BT1120 format

Copyright 2022 © Rockchip Electronics Co., Ltd.

2058

INTERFACEMIPI_CSI_PARSE_0MIPI_CSI_PARSE_1MIPI_CSI_PARSE_2MIPI_CSI_PARSE_3MIPI_CSI_PARSE_4MIPI_CSI_PARSE_5DVP_PARSECROPMUXAHB SlaveDMAMIPI0_DMAMIPI1_DMAMIPI2_DMAMIPI3_DMAMIPI4_DMAMIPI5_DMADVP_DMASCALE_DMAAXI_MASTERAXI_MASTER_1AXI_MASTER_0MMUMMU_1MMU_0SCALESCALE_0SCALE_1SCALE_2SCALE_3TOISP0TOISP1DVPMIPI CSI x 6DVP 4 channelsMIPI CSI 4 channels x 6MIPI0MIPI2MIPI5SCALEMIPI1MIPI3MIPI4DVPRAW x 3RAW x 3

RKRK3588 TRM-Part1

Fig. 37-2 BT656/BT1120 interlace timing relationship

⚫  DVP 2/4 channels mixed BT656/BT1120 format

Fig. 37-3 BT656/BT1120 progressive timing relationship

Copyright 2022 © Rockchip Electronics Co., Ltd.

2059

RKRK3588 TRM-Part1

⚫  DVP BT601 format

Fig. 37-4 4 channels mixed BT656/BT1120

⚫  MIPI CSI RAW8 format

Fig. 37-5 BT656/BT1120 timing relationship

⚫  MIPI CSI RAW10 format

Fig. 37-6 MIPI CSI RAW8 format

⚫  MIPI CSI RAW12 format

Fig. 37-7 MIPI CSI RAW10 format

⚫  MIPI CSI RAW14 format

Fig. 37-8 MIPI CSI RAW12 format

Copyright 2022 © Rockchip Electronics Co., Ltd.

2060

VsyncHsyncVideo Data

RKRK3588 TRM-Part1

⚫  MIPI CSI YUV422 format

Fig. 37-9 MIPI CSI RAW14 format

37.3.2 Crop
Bypass or crop the source video data to a smaller size destination.

Fig. 37-10 MIPI CSI YUV422 format

37.3.3 Scale
Scale down for RAW data.

Fig. 37-11 Crop

Copyright 2022 © Rockchip Electronics Co., Ltd.

2061

crop_widthcropheightsrc_widthsrcheightstart_xstart_y

RKRK3588 TRM-Part1

Fig. 37-12 8 times Scale down

37.3.4 Toisp
The parsed video data can go straight to ISP0/1 for real-time processing.
37.3.5 DMA
The DMA is used to transfer the data from crop module to the AXI master block which will
send the data to the AXI bus.
⚫  NV16

Copyright 2022 © Rockchip Electronics Co., Ltd.

2062

1920x1080240x1358 times scale down

RKRK3588 TRM-Part1

⚫  NV12

Fig. 37-13 NV16

⚫  YUV400

Fig. 37-14 NV12

Copyright 2022 © Rockchip Electronics Co., Ltd.

2063

RKRK3588 TRM-Part1

⚫  YUYV

Fig. 37-15 YUV400

⚫  Compact RAW

  Fig. 37-16 YUYV

⚫  Noncompact RAW

Fig. 37-17 Compact RAW12

Fig. 37-18 noncompact RAW12(high align)

37.3.6 AXI Master
Transmit the data to chip memory via the AXI Master.
37.3.7 MMU
Map the virtual address to physical address.
37.3.8 AHB Slave
Host configure the registers via the AHB Slave.

Copyright 2022 © Rockchip Electronics Co., Ltd.

2064

012345678910111213141516171819202122232425262728293031012345678910110123456789101101234567PIXEL 0PIXEL 1PIXEL 2[7:0]Bits of memory wordBits of pixelpixel012345678910111213141516171819202122232425262728293031----01234567891011----01234567891011PIXEL 0PIXEL 1Bits of memory wordBits of pixelpixel

RKRK3588 TRM-Part1

37.4 VICAP Register Description

37.4.1 Internal Address Mapping

Slave address can be divided into different length for different usage, which is shown as
follows.

Table 37-1 VICAP Address Mapping

Base Address[11:8]

Device

4’b0000
4’b0001
4’b0010
4’b0011
4’b0100
4’b0101
4’b0110
4’b0111
4’b1000
4’b1001

Global/DVP
MIPI0
MIPI1
MIPI2
MIPI3
MIPI4
MIPI5
Scale/TOISP
MMU0
MMU1

Address
Length

256 BYTE
256 BYTE
256 BYTE
256 BYTE
256 BYTE
256 BYTE
256 BYTE
256 BYTE
256 BYTE
256 BYTE

Offset Address Range

0x0000 ~ 0x00ff
0x0100 ~ 0x01ff
0x0200 ~ 0x02ff
0x0300 ~ 0x03ff
0x0400 ~ 0x04ff
0x0500 ~ 0x05ff
0x0600 ~ 0x06ff
0x0700 ~ 0x07ff
0x0800 ~ 0x08ff
0x0900 ~ 0x09ff

37.4.2 Registers Summary

Name

Offset  Size

Reset
Value

Description

VICAP_GLB_CTRL
VICAP_GLB_INTEN
VICAP_GLB_INTST
VICAP_DVP_CTRL
VICAP_DVP_INTEN
VICAP_DVP_INTSTAT
VICAP_DVP_FORMAT
VICAP_DVP_MULTI_ID
VICAP_DVP_SAV_EAV

0x0000  W
0x0004  W
0x0008  W
0x0010  W
0x0014  W
0x0018  W
0x001C  W
0x0020  W
0x0024  W

0x00010001  VICAP global control
0x00000000  VICAP global interrupt enable.
0x00000000  VICAP global interrupt status.
0x00000000  DVP path control
0x00000000  DVP path interrupt status
0x00000000  DVP path interrupt status
0x00000000  DVP path format
0x00000000  Channel ID for multi-ID mode
0xFEDCBA98  SAV/EAV for ACT/BLK

VICAP_DVP_CROP_SIZE  0x0028  W

0x01E002D0

VICAP_DVP_CROP_START  0x002C  W

0x00000000

The expected width and height of
received image
The start point of DVP path
cropping

VICAP_DVP_FRM0_ADDR_
Y_ID0
VICAP_DVP_FRM0_ADDR_
UV_ID0
VICAP_DVP_FRM1_ADDR_
Y_ID0
VICAP_DVP_FRM1_ADDR_
UV_ID0
VICAP_DVP_FRM0_ADDR_
Y_ID1
VICAP_DVP_FRM0_ADDR_
UV_ID1

0x0030  W

0x00000000  DVP path frame0 y address

0x0034  W

0x00000000  DVP path frame0 uv address

0x0038  W

0x00000000  DVP path frame1 y address

0x003C  W

0x00000000  DVP path frame1 uv address

0x0040  W

0x00000000

0x0044  W

0x00000000

DVP path frame0 y address for
ID1
DVP path frame0 uv address for
id1

Copyright 2022 © Rockchip Electronics Co., Ltd.

2065

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

VICAP_DVP_FRM1_ADDR_
Y_ID1
VICAP_DVP_FRM1_ADDR_
UV_ID1
VICAP_DVP_FRM0_ADDR_
Y_ID2
VICAP_DVP_FRM0_ADDR_
UV_ID2
VICAP_DVP_FRM1_ADDR_
Y_ID2
VICAP_DVP_FRM1_ADDR_
UV_ID2
VICAP_DVP_FRM0_ADDR_
Y_ID3
VICAP_DVP_FRM0_ADDR_
UV_ID3
VICAP_DVP_FRM1_ADDR_
Y_ID3
VICAP_DVP_FRM1_ADDR_
UV_ID3
VICAP_DVP_VIR_LINE_WI
DTH
VICAP_DVP_LINE_INT_NU
M_ID0_1
VICAP_DVP_LINE_INT_NU
M_ID2_3
VICAP_DVP_LINE_CNT_ID
0_1
VICAP_DVP_LINE_CNT_ID
2_3
VICAP_DVP_PIX_NUM_ID
0
VICAP_DVP_LINE_NUM_I
D0
VICAP_DVP_PIX_NUM_ID
1
VICAP_DVP_LINE_NUM_I
D1
VICAP_DVP_PIX_NUM_ID
2
VICAP_DVP_LINE_NUM_I
D2

0x0048  W

0x00000000  DVP path frame1 y address for id1

0x004C  W

0x00000000

DVP path frame1 uv address for
id1

0x0050  W

0x00000000  DVP path frame0 y address for id2

0x0054  W

0x00000000

DVP path frame0 uv address for
id2

0x0058  W

0x00000000  DVP path frame1 y address for id2

0x005C  W

0x00000000

DVP path frame1 uv address for
id2

0x0060  W

0x00000000  DVP path frame0 y address for id3

0x0064  W

0x00000000

DVP path frame0 uv address for
id3

0x0068  W

0x00000000  DVP path frame1 y address for id3

0x006C  W

0x00000000

DVP path frame1 uv address for
id3

0x0070  W

0x00000000  DVP path virtual line width

0x0074  W

0x00400040

0x0078  W

0x00400040

DVP path id0/id1 line interrupt
number
DVP path id2/id3 line interrupt
number

0x007C  W

0x00000000  DVP path id0/id1 line count

0x0080  W

0x00000000  DVP path id2/id3 line count

0x0084  W

0x00000000  DVP path ID0 PIXEL NUMBER

0x0088  W

0x00000000  DVP path ID0 LINE NUMBER

0x008C  W

0x00000000  DVP path ID1 PIXEL NUMBER

0x0090  W

0x00000000  DVP path ID1 LINE NUMBER

0x0094  W

0x00000000  DVP path ID2 PIXEL NUMBER

0x0098  W

0x00000000  DVP path ID2 LINE NUMBER

Copyright 2022 © Rockchip Electronics Co., Ltd.

2066

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

0x009C  W

0x00A0  W

0x00A4  W

VICAP_DVP_PIX_NUM_ID
3
VICAP_DVP_LINE_NUM_I
D3
VICAP_DVP_SYNC_HEADE
R
VICAP_MIPI0_ID0_CTRL0  0x0100  W
VICAP_MIPI0_ID0_CTRL1  0x0104  W
VICAP_MIPI0_ID1_CTRL0  0x0108  W
VICAP_MIPI0_ID1_CTRL1  0x010C  W
VICAP_MIPI0_ID2_CTRL0  0x0110  W
VICAP_MIPI0_ID2_CTRL1  0x0114  W
VICAP_MIPI0_ID3_CTRL0  0x0118  W
VICAP_MIPI0_ID3_CTRL1  0x011C  W
0x0120  W
VICAP_MIPI0_CTRL
VICAP_MIPI0_FRAME0_A
DDR_Y_ID0
VICAP_MIPI0_FRAME1_A
DDR_Y_ID0
VICAP_MIPI0_FRAME0_A
DDR_UV_ID0
VICAP_MIPI0_FRAME1_A
DDR_UV_ID0
VICAP_MIPI0_VLW_ID0
VICAP_MIPI0_FRAME0_A
DDR_Y_ID1
VICAP_MIPI0_FRAME1_A
DDR_Y_ID1
VICAP_MIPI0_FRAME0_A
DDR_UV_ID1
VICAP_MIPI0_FRAME1_A
DDR_UV_ID1

0x0134  W

0x0124  W

0x012C  W

0x0130  W

0x0128  W

0x0138  W

0x0144  W

0x0140  W

0x013C  W

VICAP_MIPI0_VLW_ID1

0x0148  W

0x00000000

VICAP_MIPI0_FRAME0_A
DDR_Y_ID2
VICAP_MIPI0_FRAME1_A
DDR_Y_ID2
VICAP_MIPI0_FRAME0_A
DDR_UV_ID2
VICAP_MIPI0_FRAME1_A
DDR_UV_ID2

0x014C  W

0x00000000

0x0150  W

0x00000000

0x0154  W

0x00000000

0x0158  W

0x00000000

0x00000000  DVP path ID3 PIXEL NUMBER

0x00000000  DVP path ID3 LINE NUMBER

0x00000000

DVP SYNC HEADER FOR
BT656/BT1120

0x00000000  MIPI0 path id0 control0
0x00000000  MIPI0 path id0 control1
0x00000000  MIPI0 path id1 control0
0x00000000  MIPI0 path id1 control1
0x00000000  MIPI0 path id2 control0
0x00000000  MIPI0 path id2 control1
0x00000000  MIPI0 path id3 control0
0x00000000  MIPI0 path id3 control1
0x00000000  MIPI0 path control

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID0 Y/RAW/RGB path
First address of odd frame for ID0
Y path
First address of even frame for
ID0 UV path
First address of odd frame for ID0
UV path

0x00000000  Virtual line width for ID0

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID1 Y/RAW/RGB path
First address of odd frame for ID1
Y/RAW/RGB path
First address of even frame for
ID1 UV path
First address of odd frame for ID1
UV path
Virtual line width of even frame
for ID1 path
First address of even frame for
ID2 Y/RAW/RGB path
First address of odd frame for ID2
Y/RAW/RGB path
First address of even frame for
ID2 UV path
First address of odd frame for ID2
UV path

Copyright 2022 © Rockchip Electronics Co., Ltd.

2067

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

VICAP_MIPI0_VLW_ID2

0x015C  W

VICAP_MIPI0_FRAME0_A
DDR_Y_ID3
VICAP_MIPI0_FRAME1_A
DDR_Y_ID3
VICAP_MIPI0_FRAME0_A
DDR_UV_ID3
VICAP_MIPI0_FRAME1_A
DDR_UV_ID3

0x0160  W

0x0164  W

0x0168  W

0x016C  W

VICAP_MIPI0_VLW_ID3

0x0170  W

0x0174  W
0x0178  W

0x00000000

0x00000000

0x00000000

0x00000000

Virtual line width of even frame
for ID2 path
First address of even frame for
ID3 Y/RAW/RGB path
First address of odd frame for ID3
Y/RAW/RGB path
First address of even frame for
ID3 UV path
First address of odd frame for ID3
UV path
Virtual line width of even frame
for ID3 path
0x00000000  MIPI0 path interrupt enable
0x00000000  MIPI0 path interrupt status

0x00000000

0x00000000

VICAP_MIPI0_INTEN
VICAP_MIPI0_INTSTAT
VICAP_MIPI0_LINE_INT_
NUM_ID0_1
VICAP_MIPI0_LINE_INT_
NUM_ID2_3
VICAP_MIPI0_LINE_CNT_
ID0_1
VICAP_MIPI0_LINE_CNT_
ID2_3
VICAP_MIPI0_ID0_CROP_
START
VICAP_MIPI0_ID1_CROP_
START
VICAP_MIPI0_ID2_CROP_
START
VICAP_MIPI0_ID3_CROP_
START
VICAP_MIPI0_FRAME_NU
M_VC0
VICAP_MIPI0_FRAME_NU
M_VC1
VICAP_MIPI0_FRAME_NU
M_VC2
VICAP_MIPI0_FRAME_NU
M_VC3
VICAP_MIPI0_ID0_EFFEC
T_CODE
VICAP_MIPI0_ID1_EFFEC
T_CODE

0x017C  W

0x00400040

0x0180  W

0x00400040

0x0184  W

0x00000000

0x0188  W

0x00000000

0x018C  W

0x00000000

0x0190  W

0x00000000

0x0194  W

0x00000000

0x0198  W

0x00000000

0x019C  W

0x00000000

0x01A0  W

0x00000000

0x01A4  W

0x00000000

0x01A8  W

0x00000000

Line number of the MIPI0 path
ID0/1 line interrupt
Line number of the MIPI0 path
ID2/3 line interrupt
Line count of the MIPI0 path
ID0/1
Line count of the MIPI0 path
ID2/3
The start point of MIPI0 ID0
cropping
The start point of MIPI0 ID1
cropping
The start point of MIPI0 ID2
cropping
The start point of MIPI0 ID3
cropping
The frame number of virtual
channel 0
The frame number of virtual
channel 1
The frame number of virtual
channel 2
The frame number of virtual
channel 3

0x01AC  W

0x00000000  The effect code of MIPI0 ID0

0x01B0  W

0x00000000  The effect code of MIPI0 ID1

Copyright 2022 © Rockchip Electronics Co., Ltd.

2068

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

0x01CC  W

0x01C8  W

0x01C0  W

0x01B4  W

0x01C4  W

0x01B8  W

0x01BC  W

VICAP_MIPI0_ID2_EFFEC
T_CODE
VICAP_MIPI0_ID3_EFFEC
T_CODE
VICAP_MIPI0_ON_PAD_V
ALUE
VICAP_MIPI0_SIZE_NUM_
ID0
VICAP_MIPI0_SIZE_NUM_
ID1
VICAP_MIPI0_SIZE_NUM_
ID2
VICAP_MIPI0_SIZE_NUM_
ID3
VICAP_MIPI1_ID0_CTRL0  0x0200  W
VICAP_MIPI1_ID0_CTRL1  0x0204  W
VICAP_MIPI1_ID1_CTRL0  0x0208  W
VICAP_MIPI1_ID1_CTRL1  0x020C  W
VICAP_MIPI1_ID2_CTRL0  0x0210  W
VICAP_MIPI1_ID2_CTRL1  0x0214  W
VICAP_MIPI1_ID3_CTRL0  0x0218  W
VICAP_MIPI1_ID3_CTRL1  0x021C  W
0x0220  W
VICAP_MIPI1_CTRL
VICAP_MIPI1_FRAME0_A
DDR_Y_ID0
VICAP_MIPI1_FRAME1_A
DDR_Y_ID0
VICAP_MIPI1_FRAME0_A
DDR_UV_ID0
VICAP_MIPI1_FRAME1_A
DDR_UV_ID0
VICAP_MIPI1_VLW_ID0
VICAP_MIPI1_FRAME0_A
DDR_Y_ID1
VICAP_MIPI1_FRAME1_A
DDR_Y_ID1
VICAP_MIPI1_FRAME0_A
DDR_UV_ID1
VICAP_MIPI1_FRAME1_A
DDR_UV_ID1

0x022C  W

0x0234  W

0x0224  W

0x0228  W

0x0238  W

0x0230  W

0x023C  W

0x0244  W

0x0240  W

0x00000000  The effect code of MIPI0 ID2

0x00000000  The effect code of MIPI0 ID3

0x00000000  The ON padding value of MIPI0

0x00000000  MIPI0 path ID0 SIZE NUMBER

0x00000000  MIPI0 path ID1 SIZE NUMBER

0x00000000  MIPI0 path ID2 SIZE NUMBER

0x00000000  MIPI0 path ID3 SIZE NUMBER

0x00000000  MIPI1 path id0 control0
0x00000000  MIPI1 path id0 control1
0x00000000  MIPI1 path id1 control0
0x00000000  MIPI1 path id1 control1
0x00000000  MIPI1 path id2 control0
0x00000000  MIPI1 path id2 control1
0x00000000  MIPI1 path id3 control0
0x00000000  MIPI1 path id3 control1
0x00000000  MIPI1 path control

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID0 Y/RAW/RGB path
First address of odd frame for ID0
Y path
First address of even frame for
ID0 UV path
First address of odd frame for ID0
UV path

0x00000000  Virtual line width for ID0

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID1 Y/RAW/RGB path
First address of odd frame for ID1
Y/RAW/RGB path
First address of even frame for
ID1 UV path
First address of odd frame for ID1
UV path
Virtual line width of even frame
for ID1 path

VICAP_MIPI1_VLW_ID1

0x0248  W

0x00000000

Copyright 2022 © Rockchip Electronics Co., Ltd.

2069

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

VICAP_MIPI1_FRAME0_A
DDR_Y_ID2
VICAP_MIPI1_FRAME1_A
DDR_Y_ID2
VICAP_MIPI1_FRAME0_A
DDR_UV_ID2
VICAP_MIPI1_FRAME1_A
DDR_UV_ID2

0x024C  W

0x0250  W

0x0254  W

0x0258  W

VICAP_MIPI1_VLW_ID2

0x025C  W

VICAP_MIPI1_FRAME0_A
DDR_Y_ID3
VICAP_MIPI1_FRAME1_A
DDR_Y_ID3
VICAP_MIPI1_FRAME0_A
DDR_UV_ID3
VICAP_MIPI1_FRAME1_A
DDR_UV_ID3

0x0260  W

0x0264  W

0x0268  W

0x026C  W

VICAP_MIPI1_VLW_ID3

0x0270  W

0x0274  W
0x0278  W

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID2 Y/RAW/RGB path
First address of odd frame for ID2
Y/RAW/RGB path
First address of even frame for
ID2 UV path
First address of odd frame for ID2
UV path
Virtual line width of even frame
for ID2 path
First address of even frame for
ID3 Y/RAW/RGB path
First address of odd frame for ID3
Y/RAW/RGB path
First address of even frame for
ID3 UV path
First address of odd frame for ID3
UV path
Virtual line width of even frame
for ID3 path
0x00000000  MIPI1 path interrupt enable
0x00000000  MIPI1 path interrupt status

0x00000000

0x00000000

0x00000000

0x00000000

VICAP_MIPI1_INTEN
VICAP_MIPI1_INTSTAT
VICAP_MIPI1_LINE_INT_
NUM_ID0_1
VICAP_MIPI1_LINE_INT_
NUM_ID2_3
VICAP_MIPI1_LINE_CNT_
ID0_1
VICAP_MIPI1_LINE_CNT_
ID2_3
VICAP_MIPI1_ID0_CROP_
START
VICAP_MIPI1_ID1_CROP_
START
VICAP_MIPI1_ID2_CROP_
START
VICAP_MIPI1_ID3_CROP_
START
VICAP_MIPI1_FRAME_NU
M_VC0
VICAP_MIPI1_FRAME_NU
M_VC1

0x027C  W

0x00400040

0x0280  W

0x00400040

0x0284  W

0x00000000

0x0288  W

0x00000000

0x028C  W

0x00000000

0x0290  W

0x00000000

0x0294  W

0x00000000

0x0298  W

0x00000000

0x029C  W

0x00000000

0x02A0  W

0x00000000

Line number of the MIPI1 path
ID0/1 line interrupt
Line number of the MIPI1 path
ID2/3 line interrupt
Line count of the MIPI1 path
ID0/1
Line count of the MIPI1 path
ID2/3
The start point of MIPI1 ID0
cropping
The start point of MIPI1 ID1
cropping
The start point of MIPI1 ID2
cropping
The start point of MIPI1 ID3
cropping
The frame number of virtual
channel 0
The frame number of virtual
channel 1

Copyright 2022 © Rockchip Electronics Co., Ltd.

2070

RKRK3588 TRM-Part1

Name

Offset  Size

0x02C0  W

0x02A8  W

0x02BC  W

0x02AC  W

0x02B0  W

0x02C8  W

0x02B8  W

0x02A4  W

0x02C4  W

0x02B4  W

VICAP_MIPI1_FRAME_NU
M_VC2
VICAP_MIPI1_FRAME_NU
M_VC3
VICAP_MIPI1_ID0_EFFEC
T_CODE
VICAP_MIPI1_ID1_EFFEC
T_CODE
VICAP_MIPI1_ID2_EFFEC
T_CODE
VICAP_MIPI1_ID3_EFFEC
T_CODE
VICAP_MIPI1_ON_PAD_V
ALUE
VICAP_MIPI1_SIZE_NUM_
ID0
VICAP_MIPI1_SIZE_NUM_
ID1
VICAP_MIPI1_SIZE_NUM_
ID2
VICAP_MIPI1_SIZE_NUM_
ID3
VICAP_MIPI2_ID0_CTRL0  0x0300  W
VICAP_MIPI2_ID0_CTRL1  0x0304  W
VICAP_MIPI2_ID1_CTRL0  0x0308  W
VICAP_MIPI2_ID1_CTRL1  0x030C  W
VICAP_MIPI2_ID2_CTRL0  0x0310  W
VICAP_MIPI2_ID2_CTRL1  0x0314  W
VICAP_MIPI2_ID3_CTRL0  0x0318  W
VICAP_MIPI2_ID3_CTRL1  0x031C  W
0x0320  W
VICAP_MIPI2_CTRL
VICAP_MIPI2_FRAME0_A
DDR_Y_ID0
VICAP_MIPI2_FRAME1_A
DDR_Y_ID0
VICAP_MIPI2_FRAME0_A
DDR_UV_ID0
VICAP_MIPI2_FRAME1_A
DDR_UV_ID0
VICAP_MIPI2_VLW_ID0
VICAP_MIPI2_FRAME0_A
DDR_Y_ID1

0x0334  W

0x0330  W

0x0328  W

0x0324  W

0x032C  W

0x02CC  W

0x0338  W

Reset
Value

Description

0x00000000

0x00000000

The frame number of virtual
channel 2
The frame number of virtual
channel 3

0x00000000  The effect code of MIPI1 ID0

0x00000000  The effect code of MIPI1 ID1

0x00000000  The effect code of MIPI2 ID2

0x00000000  The effect code of MIPI1 ID3

0x00000000  The ON padding value of MIPI1

0x00000000  MIPI1 path ID0 SIZE NUMBER

0x00000000  MIPI1 path ID1 SIZE NUMBER

0x00000000  MIPI1 path ID2 SIZE NUMBER

0x00000000  MIPI1 path ID3 SIZE NUMBER

0x00000000  MIPI2 path id0 control0
0x00000000  MIPI2 path id0 control1
0x00000000  MIPI2 path id1 control0
0x00000000  MIPI2 path id1 control1
0x00000000  MIPI2 path id2 control0
0x00000000  MIPI2 path id2 control1
0x00000000  MIPI2 path id3 control0
0x00000000  MIPI2 path id3 control1
0x00000000  MIPI2 path control

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID0 Y/RAW/RGB path
First address of odd frame for ID0
Y path
First address of even frame for
ID0 UV path
First address of odd frame for ID0
UV path

0x00000000  Virtual line width for ID0

0x00000000

First address of even frame for
ID1 Y/RAW/RGB path

Copyright 2022 © Rockchip Electronics Co., Ltd.

2071

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

VICAP_MIPI2_FRAME1_A
DDR_Y_ID1
VICAP_MIPI2_FRAME0_A
DDR_UV_ID1
VICAP_MIPI2_FRAME1_A
DDR_UV_ID1

0x033C  W

0x0340  W

0x0344  W

VICAP_MIPI2_VLW_ID1

0x0348  W

VICAP_MIPI2_FRAME0_A
DDR_Y_ID2
VICAP_MIPI2_FRAME1_A
DDR_Y_ID2
VICAP_MIPI2_FRAME0_A
DDR_UV_ID2
VICAP_MIPI2_FRAME1_A
DDR_UV_ID2

0x034C  W

0x0350  W

0x0354  W

0x0358  W

VICAP_MIPI2_VLW_ID2

0x035C  W

VICAP_MIPI2_FRAME0_A
DDR_Y_ID3
VICAP_MIPI2_FRAME1_A
DDR_Y_ID3
VICAP_MIPI2_FRAME0_A
DDR_UV_ID3
VICAP_MIPI2_FRAME1_A
DDR_UV_ID3

0x0360  W

0x0364  W

0x0368  W

0x036C  W

VICAP_MIPI2_VLW_ID3

0x0370  W

0x0374  W
0x0378  W

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

First address of odd frame for ID1
Y/RAW/RGB path
First address of even frame for
ID1 UV path
First address of odd frame for ID1
UV path
Virtual line width of even frame
for ID1 path
First address of even frame for
ID2 Y/RAW/RGB path
First address of odd frame for ID2
Y/RAW/RGB path
First address of even frame for
ID2 UV path
First address of odd frame for ID2
UV path
Virtual line width of even frame
for ID2 path
First address of even frame for
ID3 Y/RAW/RGB path
First address of odd frame for ID3
Y/RAW/RGB path
First address of even frame for
ID3 UV path
First address of odd frame for ID3
UV path
Virtual line width of even frame
for ID3 path
0x00000000  MIPI2 path interrupt enable
0x00000000  MIPI2 path interrupt status

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

VICAP_MIPI2_INTEN
VICAP_MIPI2_INTSTAT
VICAP_MIPI2_LINE_INT_
NUM_ID0_1
VICAP_MIPI2_LINE_INT_
NUM_ID2_3
VICAP_MIPI2_LINE_CNT_
ID0_1
VICAP_MIPI2_LINE_CNT_
ID2_3
VICAP_MIPI2_ID0_CROP_
START
VICAP_MIPI2_ID1_CROP_
START

0x037C  W

0x00400040

0x0380  W

0x00400040

0x0384  W

0x00000000

0x0388  W

0x00000000

0x038C  W

0x00000000

0x0390  W

0x00000000

Line number of the MIPI2 path
ID0/1 line interrupt
Line number of the MIPI2 path
ID2/3 line interrupt
Line count of the MIPI2 path
ID0/1
Line count of the MIPI2 path
ID2/3
The start point of MIPI2 ID0
cropping
The start point of MIPI2 ID1
cropping

Copyright 2022 © Rockchip Electronics Co., Ltd.

2072

RKRK3588 TRM-Part1

Name

Offset  Size

0x03B4  W

0x03AC  W

0x03B0  W

0x039C  W

0x03A0  W

0x03A8  W

0x03A4  W

0x03B8  W

0x0398  W

0x0394  W

VICAP_MIPI2_ID2_CROP_
START
VICAP_MIPI2_ID3_CROP_
START
VICAP_MIPI2_FRAME_NU
M_VC0
VICAP_MIPI2_FRAME_NU
M_VC1
VICAP_MIPI2_FRAME_NU
M_VC2
VICAP_MIPI2_FRAME_NU
M_VC3
VICAP_MIPI2_ID0_EFFEC
T_CODE
VICAP_MIPI2_ID1_EFFEC
T_CODE
VICAP_MIPI2_ID2_EFFEC
T_CODE
VICAP_MIPI2_ID3_EFFEC
T_CODE
VICAP_MIPI2_ON_PAD_V
ALUE
VICAP_MIPI2_SIZE_NUM_
ID0
VICAP_MIPI2_SIZE_NUM_
ID1
VICAP_MIPI2_SIZE_NUM_
ID2
VICAP_MIPI2_SIZE_NUM_
ID3
VICAP_MIPI3_ID0_CTRL0  0x0400  W
VICAP_MIPI3_ID0_CTRL1  0x0404  W
VICAP_MIPI3_ID1_CTRL0  0x0408  W
VICAP_MIPI3_ID1_CTRL1  0x040C  W
VICAP_MIPI3_ID2_CTRL0  0x0410  W
VICAP_MIPI3_ID2_CTRL1  0x0414  W
VICAP_MIPI3_ID3_CTRL0  0x0418  W
VICAP_MIPI3_ID3_CTRL1  0x041C  W
0x0420  W
VICAP_MIPI3_CTRL
VICAP_MIPI3_FRAME0_A
DDR_Y_ID0
VICAP_MIPI3_FRAME1_A
DDR_Y_ID0

0x03C4  W

0x03CC  W

0x03C0  W

0x03C8  W

0x03BC  W

0x0428  W

0x0424  W

Reset
Value

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

Description

The start point of MIPI2 ID2
cropping
The start point of MIPI2 ID3
cropping
The frame number of virtual
channel 0
The frame number of virtual
channel 1
The frame number of virtual
channel 2
The frame number of virtual
channel 3

0x00000000  The effect code of MIPI2 ID0

0x00000000  The effect code of MIPI2 ID1

0x00000000  The effect code of MIPI2 ID2

0x00000000  The effect code of MIPI2 ID3

0x00000000  The ON padding value of MIPI2

0x00000000  MIPI2 path ID0 SIZE NUMBER

0x00000000  MIPI2 path ID1 SIZE NUMBER

0x00000000  MIPI2 path ID2 SIZE NUMBER

0x00000000  MIPI2 path ID3 SIZE NUMBER

0x00000000  MIPI3 path id0 control0
0x00000000  MIPI3 path id0 control1
0x00000000  MIPI3 path id1 control0
0x00000000  MIPI3 path id1 control1
0x00000000  MIPI3 path id2 control0
0x00000000  MIPI3 path id2 control1
0x00000000  MIPI3 path id3 control0
0x00000000  MIPI3 path id3 control1
0x00000000  MIPI3 path control

0x00000000

0x00000000

First address of even frame for
ID0 Y/RAW/RGB path
First address of odd frame for ID0
Y path

Copyright 2022 © Rockchip Electronics Co., Ltd.

2073

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

0x042C  W

0x00000000

0x0430  W

0x00000000

First address of even frame for
ID0 UV path
First address of odd frame for ID0
UV path

0x0434  W

0x00000000  Virtual line width for ID0

VICAP_MIPI3_FRAME0_A
DDR_UV_ID0
VICAP_MIPI3_FRAME1_A
DDR_UV_ID0
VICAP_MIPI3_VLW_ID0
VICAP_MIPI3_FRAME0_A
DDR_Y_ID1
VICAP_MIPI3_FRAME1_A
DDR_Y_ID1
VICAP_MIPI3_FRAME0_A
DDR_UV_ID1
VICAP_MIPI3_FRAME1_A
DDR_UV_ID1

0x0438  W

0x043C  W

0x0440  W

0x0444  W

VICAP_MIPI3_VLW_ID1

0x0448  W

VICAP_MIPI3_FRAME0_A
DDR_Y_ID2
VICAP_MIPI3_FRAME1_A
DDR_Y_ID2
VICAP_MIPI3_FRAME0_A
DDR_UV_ID2
VICAP_MIPI3_FRAME1_A
DDR_UV_ID2

0x044C  W

0x0450  W

0x0454  W

0x0458  W

VICAP_MIPI3_VLW_ID2

0x045C  W

VICAP_MIPI3_FRAME0_A
DDR_Y_ID3
VICAP_MIPI3_FRAME1_A
DDR_Y_ID3
VICAP_MIPI3_FRAME0_A
DDR_UV_ID3
VICAP_MIPI3_FRAME1_A
DDR_UV_ID3

0x0460  W

0x0464  W

0x0468  W

0x046C  W

VICAP_MIPI3_VLW_ID3

0x0470  W

0x0474  W
0x0478  W

VICAP_MIPI3_INTEN
VICAP_MIPI3_INTSTAT
VICAP_MIPI3_LINE_INT_
NUM_ID0_1
VICAP_MIPI3_LINE_INT_
NUM_ID2_3
VICAP_MIPI3_LINE_CNT_
ID0_1

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID1 Y/RAW/RGB path
First address of odd frame for ID1
Y/RAW/RGB path
First address of even frame for
ID1 UV path
First address of odd frame for ID1
UV path
Virtual line width of even frame
for ID1 path
First address of even frame for
ID2 Y/RAW/RGB path
First address of odd frame for ID2
Y/RAW/RGB path
First address of even frame for
ID2 UV path
First address of odd frame for ID2
UV path
Virtual line width of even frame
for ID2 path
First address of even frame for
ID3 Y/RAW/RGB path
First address of odd frame for ID3
Y/RAW/RGB path
First address of even frame for
ID3 UV path
First address of odd frame for ID3
UV path
Virtual line width of even frame
for ID3 path
0x00000000  MIPI3 path interrupt enable
0x00000000  MIPI3 path interrupt status

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x047C  W

0x00400040

0x0480  W

0x00400040

0x0484  W

0x00000000

Line number of the MIPI3 path
ID0/1 line interrupt
Line number of the MIPI3 path
ID2/3 line interrupt
Line count of the MIPI3 path
ID0/1

Copyright 2022 © Rockchip Electronics Co., Ltd.

2074

RKRK3588 TRM-Part1

Name

Offset  Size

0x04A4  W

0x04AC  W

0x049C  W

0x0494  W

0x0490  W

0x04A8  W

0x04A0  W

0x048C  W

0x0498  W

0x0488  W

VICAP_MIPI3_LINE_CNT_
ID2_3
VICAP_MIPI3_ID0_CROP_
START
VICAP_MIPI3_ID1_CROP_
START
VICAP_MIPI3_ID2_CROP_
START
VICAP_MIPI3_ID3_CROP_
START
VICAP_MIPI3_FRAME_NU
M_VC0
VICAP_MIPI3_FRAME_NU
M_VC1
VICAP_MIPI3_FRAME_NU
M_VC2
VICAP_MIPI3_FRAME_NU
M_VC3
VICAP_MIPI3_ID0_EFFEC
T_CODE
VICAP_MIPI3_ID1_EFFEC
T_CODE
VICAP_MIPI3_ID2_EFFEC
T_CODE
VICAP_MIPI3_ID3_EFFEC
T_CODE
VICAP_MIPI3_ON_PAD_V
ALUE
VICAP_MIPI3_SIZE_NUM_
ID0
VICAP_MIPI3_SIZE_NUM_
ID1
VICAP_MIPI3_SIZE_NUM_
ID2
VICAP_MIPI3_SIZE_NUM_
ID3
VICAP_MIPI4_ID0_CTRL0  0x0500  W
VICAP_MIPI4_ID0_CTRL1  0x0504  W
VICAP_MIPI4_ID1_CTRL0  0x0508  W
VICAP_MIPI4_ID1_CTRL1  0x050C  W
VICAP_MIPI4_ID2_CTRL0  0x0510  W
VICAP_MIPI4_ID2_CTRL1  0x0514  W
VICAP_MIPI4_ID3_CTRL0  0x0518  W

0x04C4  W

0x04C0  W

0x04BC  W

0x04C8  W

0x04CC  W

0x04B0  W

0x04B8  W

0x04B4  W

Reset
Value

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

Description

Line count of the MIPI3 path
ID2/3
The start point of MIPI3 ID0
cropping
The start point of MIPI3 ID1
cropping
The start point of MIPI3 ID2
cropping
The start point of MIPI3 ID3
cropping
The frame number of virtual
channel 0
The frame number of virtual
channel 1
The frame number of virtual
channel 2
The frame number of virtual
channel 3

0x00000000  The effect code of MIPI3 ID0

0x00000000  The effect code of MIPI3 ID1

0x00000000  The effect code of MIPI3 ID2

0x00000000  The effect code of MIPI3 ID3

0x00000000  The ON padding value of MIPI3

0x00000000  MIPI3 path ID0 SIZE NUMBER

0x00000000  MIPI3 path ID1 SIZE NUMBER

0x00000000  MIPI3 path ID2 SIZE NUMBER

0x00000000  MIPI3 path ID3 SIZE NUMBER

0x00000000  MIPI4 path id0 control0
0x00000000  MIPI4 path id0 control1
0x00000000  MIPI4 path id1 control0
0x00000000  MIPI4 path id1 control1
0x00000000  MIPI4 path id2 control0
0x00000000  MIPI4 path id2 control1
0x00000000  MIPI4 path id3 control0

Copyright 2022 © Rockchip Electronics Co., Ltd.

2075

RKRK3588 TRM-Part1

Name

Offset  Size

0x0524  W

0x052C  W

0x0528  W

VICAP_MIPI4_ID3_CTRL1  0x051C  W
0x0520  W
VICAP_MIPI4_CTRL
VICAP_MIPI4_FRAME0_A
DDR_Y_ID0
VICAP_MIPI4_FRAME1_A
DDR_Y_ID0
VICAP_MIPI4_FRAME0_A
DDR_UV_ID0
VICAP_MIPI4_FRAME1_A
DDR_UV_ID0
VICAP_MIPI4_VLW_ID0
VICAP_MIPI4_FRAME0_A
DDR_Y_ID1
VICAP_MIPI4_FRAME1_A
DDR_Y_ID1
VICAP_MIPI4_FRAME0_A
DDR_UV_ID1
VICAP_MIPI4_FRAME1_A
DDR_UV_ID1

0x0544  W

0x0538  W

0x0540  W

0x053C  W

0x0534  W

0x0530  W

VICAP_MIPI4_VLW_ID1

0x0548  W

VICAP_MIPI4_FRAME0_A
DDR_Y_ID2
VICAP_MIPI4_FRAME1_A
DDR_Y_ID2
VICAP_MIPI4_FRAME0_A
DDR_UV_ID2
VICAP_MIPI4_FRAME1_A
DDR_UV_ID2

0x054C  W

0x0550  W

0x0554  W

0x0558  W

VICAP_MIPI4_VLW_ID2

0x055C  W

VICAP_MIPI4_FRAME0_A
DDR_Y_ID3
VICAP_MIPI4_FRAME1_A
DDR_Y_ID3
VICAP_MIPI4_FRAME0_A
DDR_UV_ID3
VICAP_MIPI4_FRAME1_A
DDR_UV_ID3

0x0560  W

0x0564  W

0x0568  W

0x056C  W

VICAP_MIPI4_VLW_ID3

0x0570  W

VICAP_MIPI4_INTEN
VICAP_MIPI4_INTSTAT

0x0574  W
0x0578  W

Reset
Value

Description

0x00000000  MIPI4 path id3 control1
0x00000000  MIPI4 path control

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID0 Y/RAW/RGB path
First address of odd frame for ID0
Y path
First address of even frame for
ID0 UV path
First address of odd frame for ID0
UV path

0x00000000  Virtual line width for ID0

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID1 Y/RAW/RGB path
First address of odd frame for ID1
Y/RAW/RGB path
First address of even frame for
ID1 UV path
First address of odd frame for ID1
UV path
Virtual line width of even frame
for ID1 path
First address of even frame for
ID2 Y/RAW/RGB path
First address of odd frame for ID2
Y/RAW/RGB path
First address of even frame for
ID2 UV path
First address of odd frame for ID2
UV path
Virtual line width of even frame
for ID2 path
First address of even frame for
ID3 Y/RAW/RGB path
First address of odd frame for ID3
Y/RAW/RGB path
First address of even frame for
ID3 UV path
First address of odd frame for ID3
UV path
Virtual line width of even frame
for ID3 path
0x00000000  MIPI4 path interrupt enable
0x00000000  MIPI4 path interrupt status

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

Copyright 2022 © Rockchip Electronics Co., Ltd.

2076

RKRK3588 TRM-Part1

Name

Offset  Size

0x0588  W

0x0594  W

0x0590  W

0x058C  W

0x0584  W

0x057C  W

0x0580  W

0x05A0  W

0x0598  W

0x059C  W

VICAP_MIPI4_LINE_INT_
NUM_ID0_1
VICAP_MIPI4_LINE_INT_
NUM_ID2_3
VICAP_MIPI4_LINE_CNT_
ID0_1
VICAP_MIPI4_LINE_CNT_
ID2_3
VICAP_MIPI4_ID0_CROP_
START
VICAP_MIPI4_ID1_CROP_
START
VICAP_MIPI4_ID2_CROP_
START
VICAP_MIPI4_ID3_CROP_
START
VICAP_MIPI4_FRAME_NU
M_VC0
VICAP_MIPI4_FRAME_NU
M_VC1
VICAP_MIPI4_FRAME_NU
M_VC2
VICAP_MIPI4_FRAME_NU
M_VC3
VICAP_MIPI4_ID0_EFFEC
T_CODE
VICAP_MIPI4_ID1_EFFEC
T_CODE
VICAP_MIPI4_ID2_EFFEC
T_CODE
VICAP_MIPI4_ID3_EFFEC
T_CODE
VICAP_MIPI4_ON_PAD_V
ALUE
VICAP_MIPI4_SIZE_NUM_
ID0
VICAP_MIPI4_SIZE_NUM_
ID1
VICAP_MIPI4_SIZE_NUM_
ID2
VICAP_MIPI4_SIZE_NUM_
ID3
VICAP_MIPI5_ID0_CTRL0  0x0600  W

0x05B8  W

0x05C4  W

0x05A4  W

0x05CC  W

0x05B4  W

0x05BC  W

0x05C0  W

0x05B0  W

0x05A8  W

0x05AC  W

0x05C8  W

Reset
Value

0x00400040

0x00400040

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

0x00000000

Description

Line number of the MIPI4 path
ID0/1 line interrupt
Line number of the MIPI4 path
ID2/3 line interrupt
Line count of the MIPI4 path
ID0/1
Line count of the MIPI4 path
ID2/3
The start point of MIPI4 ID0
cropping
The start point of MIPI4 ID1
cropping
The start point of MIPI4 ID2
cropping
The start point of MIPI4 ID3
cropping
The frame number of virtual
channel 0
The frame number of virtual
channel 1
The frame number of virtual
channel 2
The frame number of virtual
channel 3

0x00000000  The effect code of MIPI4 ID0

0x00000000  The effect code of MIPI4 ID1

0x00000000  The effect code of MIPI4 ID2

0x00000000  The effect code of MIPI4 ID3

0x00000000  The ON padding value of MIPI4

0x00000000  MIPI4 path ID0 SIZE NUMBER

0x00000000  MIPI4 path ID1 SIZE NUMBER

0x00000000  MIPI4 path ID2 SIZE NUMBER

0x00000000  MIPI4 path ID3 SIZE NUMBER

0x00000000  MIPI5 path id0 control0

Copyright 2022 © Rockchip Electronics Co., Ltd.

2077

RKRK3588 TRM-Part1

Name

Offset  Size

0x0624  W

VICAP_MIPI5_ID0_CTRL1  0x0604  W
VICAP_MIPI5_ID1_CTRL0  0x0608  W
VICAP_MIPI5_ID1_CTRL1  0x060C  W
VICAP_MIPI5_ID2_CTRL0  0x0610  W
VICAP_MIPI5_ID2_CTRL1  0x0614  W
VICAP_MIPI5_ID3_CTRL0  0x0618  W
VICAP_MIPI5_ID3_CTRL1  0x061C  W
0x0620  W
VICAP_MIPI5_CTRL
VICAP_MIPI5_FRAME0_A
DDR_Y_ID0
VICAP_MIPI5_FRAME1_A
DDR_Y_ID0
VICAP_MIPI5_FRAME0_A
DDR_UV_ID0
VICAP_MIPI5_FRAME1_A
DDR_UV_ID0
VICAP_MIPI5_VLW_ID0
VICAP_MIPI5_FRAME0_A
DDR_Y_ID1
VICAP_MIPI5_FRAME1_A
DDR_Y_ID1
VICAP_MIPI5_FRAME0_A
DDR_UV_ID1
VICAP_MIPI5_FRAME1_A
DDR_UV_ID1

0x063C  W

0x0628  W

0x0630  W

0x0634  W

0x062C  W

0x0640  W

0x0638  W

0x0644  W

VICAP_MIPI5_VLW_ID1

0x0648  W

0x00000000

VICAP_MIPI5_FRAME0_A
DDR_Y_ID2
VICAP_MIPI5_FRAME1_A
DDR_Y_ID2
VICAP_MIPI5_FRAME0_A
DDR_UV_ID2
VICAP_MIPI5_FRAME1_A
DDR_UV_ID2

0x064C  W

0x00000000

0x0650  W

0x00000000

0x0654  W

0x00000000

0x0658  W

0x00000000

VICAP_MIPI5_VLW_ID2

0x065C  W

0x00000000

VICAP_MIPI5_FRAME0_A
DDR_Y_ID3
VICAP_MIPI5_FRAME1_A
DDR_Y_ID3
VICAP_MIPI5_FRAME0_A
DDR_UV_ID3

0x0660  W

0x00000000

0x0664  W

0x00000000

0x0668  W

0x00000000

Reset
Value

Description

0x00000000  MIPI5 path id0 control1
0x00000000  MIPI5 path id1 control0
0x00000000  MIPI5 path id1 control1
0x00000000  MIPI5 path id2 control0
0x00000000  MIPI5 path id2 control1
0x00000000  MIPI5 path id3 control0
0x00000000  MIPI5 path id3 control1
0x00000000  MIPI5 path control

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID0 Y/RAW/RGB path
First address of odd frame for ID0
Y path
First address of even frame for
ID0 UV path
First address of odd frame for ID0
UV path

0x00000000  Virtual line width for ID0

0x00000000

0x00000000

0x00000000

0x00000000

First address of even frame for
ID1 Y/RAW/RGB path
First address of odd frame for ID1
Y/RAW/RGB path
First address of even frame for
ID1 UV path
First address of odd frame for ID1
UV path
Virtual line width of even frame
for ID1 path
First address of even frame for
ID2 Y/RAW/RGB path
First address of odd frame for ID2
Y/RAW/RGB path
First address of even frame for
ID2 UV path
First address of odd frame for ID2
UV path
Virtual line width of even frame
for ID2 path
First address of even frame for
ID3 Y/RAW/RGB path
First address of odd frame for ID3
Y/RAW/RGB path
First address of even frame for
ID3 UV path

Copyright 2022 © Rockchip Electronics Co., Ltd.

2078

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

VICAP_MIPI5_FRAME1_A
DDR_UV_ID3

0x066C  W

VICAP_MIPI5_VLW_ID3

0x0670  W

0x00000000

0x00000000

First address of odd frame for ID3
UV path
Virtual line width of even frame
for ID3 path
0x00000000  MIPI5 path interrupt enable
0x00000000  MIPI5 path interrupt status

0x0674  W
0x0678  W

VICAP_MIPI5_INTEN
VICAP_MIPI5_INTSTAT
VICAP_MIPI5_LINE_INT_
NUM_ID0_1
VICAP_MIPI5_LINE_INT_
NUM_ID2_3
VICAP_MIPI5_LINE_CNT_
ID0_1
VICAP_MIPI5_LINE_CNT_
ID2_3
VICAP_MIPI5_ID0_CROP_
START
VICAP_MIPI5_ID1_CROP_
START
VICAP_MIPI5_ID2_CROP_
START
VICAP_MIPI5_ID3_CROP_
START
VICAP_MIPI5_FRAME_NU
M_VC0
VICAP_MIPI5_FRAME_NU
M_VC1
VICAP_MIPI5_FRAME_NU
M_VC2
VICAP_MIPI5_FRAME_NU
M_VC3
VICAP_MIPI5_ID0_EFFEC
T_CODE
VICAP_MIPI5_ID1_EFFEC
T_CODE
VICAP_MIPI5_ID2_EFFEC
T_CODE
VICAP_MIPI5_ID3_EFFEC
T_CODE
VICAP_MIPI5_ON_PAD_V
ALUE
VICAP_MIPI5_SIZE_NUM_
ID0

0x067C  W

0x00400040

0x0680  W

0x00400040

0x0684  W

0x00000000

0x0688  W

0x00000000

0x068C  W

0x00000000

0x0690  W

0x00000000

0x0694  W

0x00000000

0x0698  W

0x00000000

0x069C  W

0x00000000

0x06A0  W

0x00000000

0x06A4  W

0x00000000

0x06A8  W

0x00000000

Line number of the MIPI5 path
ID0/1 line interrupt
Line number of the MIPI5 path
ID2/3 line interrupt
Line count of the MIPI5 path
ID0/1
Line count of the MIPI5 path
ID2/3
The start point of MIPI5 ID0
cropping
The start point of MIPI5 ID1
cropping
The start point of MIPI5 ID2
cropping
The start point of MIPI5 ID3
cropping
The frame number of virtual
channel 0
The frame number of virtual
channel 1
The frame number of virtual
channel 2
The frame number of virtual
channel 3

0x06AC  W

0x00000000  The effect code of MIPI5 ID0

0x06B0  W

0x00000000  The effect code of MIPI5 ID1

0x06B4  W

0x00000000  The effect code of MIPI5 ID2

0x06B8  W

0x00000000  The effect code of MIPI5 ID3

0x06BC  W

0x00000000  The ON padding value of MIPI5

0x06C0  W

0x00000000  MIPI5 path ID0 SIZE NUMBER

Copyright 2022 © Rockchip Electronics Co., Ltd.

2079

RKRK3588 TRM-Part1

Name

Offset  Size

Reset
Value

Description

0x06C8  W

0x06CC  W

0x0714  W

0x06C4  W

0x071C  W

0x070C  W

0x0720  W

0x0708  W

0x0718  W

0x0710  W

0x0700  W
0x0704  W

VICAP_MIPI5_SIZE_NUM_
ID1
VICAP_MIPI5_SIZE_NUM_
ID2
VICAP_MIPI5_SIZE_NUM_
ID3
VICAP_SCL_CH_CTRL
VICAP_SCL_CTRL
VICAP_SCL_FRAME0_ADD
R_CH0
VICAP_SCL_FRAME1_ADD
R_CH0
VICAP_SCL_VLW_CH0
VICAP_SCL_FRAME0_ADD
R_CH1
VICAP_SCL_FRAME1_ADD
R_CH1
VICAP_SCL_VLW_CH1
VICAP_SCL_FRAME0_ADD
R_CH2
VICAP_SCL_FRAME1_ADD
R_CH2
VICAP_SCL_VLW_CH2
VICAP_SCL_FRAME0_ADD
R_CH3
VICAP_SCL_FRAME1_ADD
R_CH3
VICAP_SCL_VLW_CH3
VICAP_SCL_CH0_BLACK_
LEVEL
VICAP_SCL_CH1_BLACK_
LEVEL
VICAP_SCL_CH2_BLACK_
LEVEL
VICAP_SCL_CH3_BLACK_
LEVEL
VICAP_TOISP0_CH_CTRL  0x0780  W
VICAP_TOISP0_CROP_SIZ
E
VICAP_TOISP0_CROP_ST
ART
VICAP_TOISP1_CH_CTRL  0x078C  W

0x0730  W

0x0738  W

0x0734  W

0x0784  W

0x072C  W

0x0724  W

0x073C  W

0x0744  W

0x0788  W

0x0740  W

0x0728  W

0x00000000  MIPI5 path ID1 SIZE NUMBER

0x00000000  MIPI5 path ID2 SIZE NUMBER

0x00000000  MIPI5 path ID3 SIZE NUMBER

0x00000000  Scale channel0 path control
0x00000000  MIPI3 path control

0x00000000

First address of even frame for
CH0 path
First address of odd frame for
CH0 path
0x00000000  Virtual line width for CH0

0x00000000

0x00000000

First address of even frame for
CH1 path
First address of odd frame for
CH1 path
0x00000000  Virtual line width for CH0

0x00000000

0x00000000

First address of even frame for
CH2 path
First address of odd frame for
CH2 path
0x00000000  Virtual line width for CH2

0x00000000

0x00000000

First address of even frame for
CH3 path
First address of odd frame for
CH1 path
0x00000000  Virtual line width for CH3

0x00000000

0x00000000  Scale channel0 black level

0x00000000  Scale channel1 black level

0x00000000  Scale channel2 black level

0x00000000  Scale channel3 black level

0x00000000  TOISP0 path control

0x01E002D0

0x00000000

The expected width and height of
received image
The start point of toisp0 path
cropping

0x00000000  TOISP1 path control

Copyright 2022 © Rockchip Electronics Co., Ltd.

2080


