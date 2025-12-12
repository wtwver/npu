"""
SELU LUT generator/approximation for RK3588-style NVDLA tables.

- Table: 1026 entries (513 per side), center index 512 at x=0.
- Indexing: idx = ((round(|x| * 2596) + 16) >> 5), clamped to 512.
- Coverage: roughly |x| <= 6.305 with ~0.01233 step away from zero.
- Output format: signed int16 scaled by OUTPUT_SCALE (computed from range).
"""

from __future__ import annotations

import numpy as np

LUT_SIZE = 1026
INDEX_SCALE = 2596.0
INDEX_SHIFT = 5
ROUND_BIAS = 1 << (INDEX_SHIFT - 1)

CENTER_INDEX = (LUT_SIZE - 1) // 2
MAX_OFFSET = CENTER_INDEX

SELU_ALPHA = 1.6732632423543772
SELU_LAMBDA = 1.0507009873554805


def selu_fn(x: float) -> float:
    if x > 0:
        return SELU_LAMBDA * x
    return SELU_LAMBDA * SELU_ALPHA * (np.exp(x / SELU_ALPHA) - 1.0)


def _generate_lut() -> tuple[np.ndarray, float]:
    ys = []
    for i in range(LUT_SIZE):
        offset = i - CENTER_INDEX
        x_abs = ((abs(offset) << INDEX_SHIFT) - ROUND_BIAS) / INDEX_SCALE
        x = x_abs if offset >= 0 else -x_abs
        ys.append(selu_fn(x))

    ys_arr = np.array(ys, dtype=np.float64)
    max_abs = float(np.max(np.abs(ys_arr)))
    scale = ((1 << 15) - 1) / max_abs if max_abs > 0 else 1.0
    lut = np.clip(np.round(ys_arr * scale), -32768, 32767).astype(np.int16)
    return lut, scale


selu_lut, OUTPUT_SCALE = _generate_lut()


def _lut_offset(x_abs: float) -> int:
    scaled = int(np.round(x_abs * INDEX_SCALE))
    idx = (scaled + ROUND_BIAS) >> INDEX_SHIFT
    if idx > MAX_OFFSET:
        idx = MAX_OFFSET
    return idx


def hardware_selu(x_float: float) -> float:
    x_abs = abs(float(x_float))
    offset = _lut_offset(x_abs)
    lut_index = CENTER_INDEX + offset if x_float >= 0 else CENTER_INDEX - offset
    lut_raw = int(selu_lut[lut_index])
    return lut_raw / OUTPUT_SCALE


if __name__ == "__main__":
    sample_inputs = np.linspace(-6.0, 6.0, num=16, dtype=np.float32)
    expected = np.array([selu_fn(float(x)) for x in sample_inputs], dtype=np.float64)

    print(f"Generated SELU LUT with {len(selu_lut)} entries (OUTPUT_SCALE={OUTPUT_SCALE:.4f})")
    print("Testing hardware SELU approximation (NVDLA-style indexing):")

    tol = 1.3e-2
    hw_vals = []
    for x, exp_val in zip(sample_inputs, expected):
        offset = _lut_offset(abs(float(x)))
        lut_idx = CENTER_INDEX + offset if x >= 0 else CENTER_INDEX - offset
        hw = hardware_selu(float(x))
        hw_vals.append(hw)
        print(
            f"x={x: .5f} idx={offset:3d} lut[{lut_idx}]=0x{int(selu_lut[lut_idx]) & 0xFFFF:04x} "
            f"hw={hw:.6f} ref={exp_val:.6f}"
        )

    hw_vals = np.array(hw_vals, dtype=np.float64)
    diffs = np.abs(hw_vals - expected)
    print(f"Max abs diff: {diffs.max():.6f} (tol {tol})")
