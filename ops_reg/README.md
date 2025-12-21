Rounddown/LUT notes (RK3588 ops_reg)

Round off (nearest integer, ties-to-even)
- Where: `npu/include/rknnops.h`, `current_alu_algorithm == 23`.
- LUT: alternating entries `0x0000, 0x0020`, with `LUT_INFO index_select=5`.
- EW: `EW_LUT_BYPASS=0`, `EW_OP_BYPASS=1`, `EW_OP_CVT_BYPASS=1`.
- OUT_CVT: disabled.
- Result: LUT interpolation yields integer steps (1/32 granularity) and the output is rounded to the nearest integer in FP16.

Fraction extraction (in progress)
- Goal: `frac(x)` for positive FP16 values below 1024.
- Approach: scale input by 1024, use low 10 bits as LUT index, and map index -> Q0.15 fraction.
- Registers (algo 21):
  - `REG_DPU_EW_CFG`: enable EW data mode + op convert, keep LUT enabled.
  - `REG_DPU_EW_CVT_SCALE_VALUE`: scale = 1024, shift = 0 (scale input before LUT).
  - `REG_DPU_LUT_CFG`: `LUT_EXPAND_EN=1`.
  - `REG_DPU_LUT_INFO`: `LUT_*_INDEX_SELECT=0` (use low 10 bits as index).
  - `REG_DPU_LUT_LE/LO_*`: range set to 0..1024 (positive only).
  - `REG_DPU_OUT_CVT_*`: convert Q0.15 -> FP16 with `minus_exp=15`.
- LUT contents:
  - LE table: `entry[i] = i << 5` for i=0..511.
- LO table: `entry[i] = (i + 512) << 5` for i=0..511.
- This maps index 0..1023 to values 0..32736 (Q0.15).
- Expected: output ~= (x * 1024 mod 1024) / 1024 in FP16.

Test suite
- Build + run all ops_reg tests: `./run_tests.sh`
- Run all tests in one process: `./main all`
- Run a single test: `./main <test>`
