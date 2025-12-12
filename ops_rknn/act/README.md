RK’s activation LUTs in ops_rknn/act mirror the NVDLA 2‑table layout. The captures in ops_rknn/act/sigmoid.py (1026 entries) and ops_rknn/act/silu.py (1024 entries) show how the two 513‑entry halves are packed and indexed.

# Layout & indexing (sigmoid)

Files: ops_rknn/act/sigmoid.py, dump ops_rknn/act/sigmoid2.
Table length 1026 ⇒ 513 usable offsets on each side of zero; center index 512 is sigmoid(0)=0.5.
Input quantization: idx = ((round(|x| * 2596) + 16) >> 5), clamped to 512. One bin covers ~0.0062 around zero; subsequent bins are ~0.01233 apart. The last usable bin maps to |x| ≈ 6.305.
LUT values are Q0.15 unsigned: lut[idx]/32768. Example: index 0 ≈0.0018, center=16384 →0.5, near the end ≈32708 →0.999.

# Layout & indexing (SiLU)

File: ops_rknn/act/silu.py, dump ops_rknn/act/silu2.
Length 1024 (two 512‑entry halves, zero crossing between 511/512).
Input is rounded to FP16 first, then offset_f = (|x|_fp16 * 2824) / 32; hardware linearly interpolates between neighboring entries. |x| up to ~5.79 hits the last bin.
LUT values are signed int16 already scaled by ~5692 (see OUTPUT_SCALE), so the hardware output is lut_interp / 5692.1.

# How the tables are generated (current dumps)

Choose a symmetric coverage so that offset 512 lands where the function is already saturated (≈±6.3 for sigmoid, ≈±5.8 for SiLU).
Compute the grid implied by the index formula above (include the +16 bias for sigmoid, FP16 rounding for SiLU).
Evaluate the float function at each grid point (center at x=0), quantize to the target Q format (Q0.15 for sigmoid; fitted int16 scale for SiLU), and store.
Out‑of‑range inputs reuse the end entries (or hardware can apply linear overflow slopes; see ops_rknn/act/README.md from the NVDLA doc).

# Quantizing for highest precision

Match the hardware grid exactly: generate samples at (offset<<5 − 16)/2596 for sigmoid and (offset<<5)/2824 (with FP16 rounding) for SiLU.
Use double/float64 when computing the reference values, then round to the hardware’s integer format (nearest, with saturation to the int16/uint16 range).
Keep the “dense” portion of the function inside the covered range (e.g., put the knees of sigmoid/tanh inside ±3; put SiLU’s curvature inside ±4). If you can reprogram the two 513‑entry halves, dedicate one half as a dense table over the steep region and the other as a coarse table over the tails.

# Feasibility and expected error with a 513‑step per‑side LUT (nearest/linear)

sigmoid: already measured in the script with a 5e‑3 tolerance; simulated max abs err ≈3e‑3 over [-6.3, 6.3] with nearest lookup.
tanh: works well with the same grid; expect ≈1e‑2 abs error (Q0.15 + grid spacing).
GELU (tanh approximation form): smooth and saturating; expect ≈1.3e‑2 abs error with nearest lookup; interpolation or a denser center table can push below 8e‑3.
CELU/SELU: similar curvature to ELU; expect ≈1.2e‑2 abs error. Use the dense table on the negative branch where curvature is strongest.
logarithm: domain is (0, ∞) with very high curvature near 0. A two‑table scheme can work if one table is dedicated to a dense region near 0 (e.g., [1e‑4, 1]) and the other covers the tail; still expect noticeable relative error near 0 unless you use exponential indexing. Practical target: <1e‑2 relative for x≥1e‑3.
log_softmax: not a single‑input nonlinearity; LUT can only approximate sub‑parts (exp/log) but the normalization term depends on the whole vector. Expecting LUT‑only high precision is unrealistic; you’d need full softmax accumulation plus an exp/log LUT.
Overall rule of thumb with this grid/Q format: smooth, saturating 1‑D activations (sigmoid, tanh, GELU, SiLU) can stay within 3e‑3–1.3e‑2 abs error; sharper or unbounded functions (log, log_softmax) need special handling or will exceed that unless you sacrifice range for density.