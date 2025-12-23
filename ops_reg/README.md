Here’s the reasoning for the 0/32 LUT pattern and the updated Python model.

Why 0/32 + LE/LO split + INDEX_SELECT(5) yields round‑to‑nearest‑even

- In the SDP LUT index path (sdp_y_idx.cpp), linear FP16 mode computes index = floor((x - start) / 2^INDEX_SELECT) and a 35‑bit fraction = remainder / 2^INDEX_SELECT. With INDEX_SELECT=5, the fraction is quantized in 1/32 steps of the pre‑LUT scaled input.
- The RK3588 roundoff flow scales input so that each 1.0 in the original value maps to a 32‑step LUT fraction. That makes the LUT fraction line up with the original fractional part at 1/32 resolution.
- Alternating LUT entries 0 and 32 produces a sawtooth after interpolation (sdp_y_inp.cpp): even→odd entries ramp 0→32, odd→even ramp 32→0. The midpoint (value 16) is exactly the 0.5 boundary; the alternating direction gives the even/odd tie behavior.
- Splitting the range with LE_START=0, LE_END=512, LO_START=512, LO_END=1024 and HYBRID_PRIORITY=LO selects the correct table for 0–512 vs 512–1024, avoiding oflow/uflow while keeping the same ramp behavior across the whole domain.
- Note: in stock NVDLA FP16 mode the LUT values are interpreted as FP16 bits. If your modified RTL treats the LUT as fixed‑point or applies a post‑scale, then “32” is literal; otherwise 0x0020 is a subnormal FP16. Your modified implementation likely includes that fixed‑point interpretation or scaling.
Python model aligned to hw/ SDP LUT
I rewrote roundoff.py to follow the HLS model:

- LE/LO tables + priority selection (sdp_y_idx.cpp)
- 35‑bit fraction interpolation path (sdp_y_inp.cpp)
- uflow/oflow extrapolation path (slope + offset) in the same way as the wrapper
File updated: roundoff.py

If your modified RTL uses 1025 entries, set LE_TBL_ENTRY/LO_TBL_ENTRY in roundoff.py to match your actual LE/LO sizes (defaults are the hw/ values 65/257).

Next steps (optional)

Update LE_TBL_ENTRY/LO_TBL_ENTRY and the LUT programming values to your exact 1025‑entry RTL.
Run roundoff.py to compare the LUT output with the reference rounding.

===

Here’s how the LUT works in the NVDLA hardware under hw/, and how that compares to your Python model.

LUT in hw (NVDLA SDP path)

- Index generation happens in NV_NVDLA_SDP_CORE_Y_idx, which computes LE/LO indices, fractions, and uflow/oflow based on *_START, *_END, *_INDEX_SELECT, and *_INDEX_OFFSET (linear vs exponent) in sdp_y_idx.cpp.
- There are two tables (LE and LO) with fixed depths (LE=65, LO=257) defined in sdp.h and mirrored in sdp_hls_wrapper.h.
- The wrapper reads y0/y1 from the selected table and either interpolates (y0*(1-f)+y1*f) or extrapolates using slope/shift/offset when uflow/oflow is set; see sdp_hls_wrapper.cpp and sdp_y_inp.cpp.
- For FP16, indices come from FpFloatToIntFrac (shift by index_select), and the fraction is a 35‑bit value, not a coarse 1/32 step.


======

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
