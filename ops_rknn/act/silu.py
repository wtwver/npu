#############

# 0.195254 * 2824 (0x6984 fp16) = 0x4409e400 fp32 = 551.5625 
# LUT[551.5625/32] = LUT[17.2] = 598 to 636 = (551.5625/32-17) * (636-598) + 598 = 607
# 607 * 0.01329 * 0.01329 = 0.1072108287
# silu(0.195254) = 0.1072108287

# 0.860757 * 2824 = 0x4517f030 = 2431.0117
# LUT[2431.0117/32] = LUT[75.969115625] = 3372 to 3429  = ( 2431.0117/32 -75 ) * (3429-3372) + 3372 = 3427
###

import re
import numpy as np
from pathlib import Path

# RK3588 SiLU LUT dump (REG_DPU_LUT_ACCESS_DATA) is in ./silu2.
# - 1024 signed int16 entries with the zero crossing between indices 511/512.
# - Indexing uses |x_fp16| * 2824 (float32), then divides by 32 to obtain a
#   fractional LUT position; hardware linearly interpolates adjacent entries.
# - LUT values already encode silu(x) scaled by ~5692 (re-fit over [-4, 4]
#   to minimize max error), so no additional 2**-15 shift.
SCRIPT_DIR = Path(__file__).resolve().parent
SILU_LUT_PATH = SCRIPT_DIR / "silu2"
SILU_LUT_SIZE = 1024
INDEX_SCALE = 2824.0
INDEX_SHIFT = 5
OUTPUT_SCALE = 5692.1


def _parse_silu_lut(path: Path) -> np.ndarray:
    text = path.read_text().splitlines()
    vals = []
    for line in text:
        m = re.search(r"LUT_ACCESS_DATA[^,]*\((\d+)\)", line)
        if m:
            vals.append(int(m.group(1)))
    if len(vals) != SILU_LUT_SIZE:
        raise ValueError(f"Expected {SILU_LUT_SIZE} LUT entries, found {len(vals)} in {path}")
    # Interpret as signed int16 without overflow warnings.
    return np.array(vals, dtype=np.uint16).view(np.int16)


def _load_silu_lut() -> np.ndarray:
    if not SILU_LUT_PATH.exists():
        raise FileNotFoundError(f"LUT dump not found: {SILU_LUT_PATH}")
    return _parse_silu_lut(SILU_LUT_PATH)


silu_lut = _load_silu_lut()
CENTER_INDEX = (len(silu_lut) - 1) // 2  # 511 for 1024-entry table (zero between 511/512)
MAX_OFFSET = min(CENTER_INDEX, len(silu_lut) - 1 - CENTER_INDEX)


def _lut_offset(x_abs: float) -> tuple[int, float]:
    """
    Quantize |x| into a LUT offset and fractional stride using the hardware scheme.

    Args:
        x_abs: Absolute input value (float).

    Returns:
        base (int): Integer LUT offset (>=0).
        frac (float): Fractional part within [0, 1) used for interpolation.
    """
    x_f16 = np.float16(x_abs)
    scaled = float(np.float32(x_f16) * np.float32(INDEX_SCALE))
    offset_f = scaled / float(1 << INDEX_SHIFT)
    base = int(np.floor(offset_f))
    frac = float(offset_f - base)
    if base >= MAX_OFFSET:
        base = MAX_OFFSET
        frac = 0.0
    return base, frac


def hardware_silu(x_float: float):
    """
    Approximate SiLU directly from the hardware LUT.

    Returns:
        silu_hw (float): SiLU approximation.
        lut_index (int): Base index into the LUT.
        lut_raw (int): Interpolated LUT value (rounded to int16 domain).
    """
    x_abs = abs(float(x_float))
    offset_base, offset_frac = _lut_offset(x_abs)
    if x_float >= 0:
        lut_index = CENTER_INDEX + offset_base
        lut_next = min(lut_index + 1, len(silu_lut) - 1)
    else:
        lut_index = CENTER_INDEX - offset_base
        lut_next = max(lut_index - 1, 0)

    lut_lo = float(silu_lut[lut_index])
    lut_hi = float(silu_lut[lut_next])
    lut_interp = lut_lo + (lut_hi - lut_lo) * offset_frac
    lut_raw = int(np.round(lut_interp))
    silu_hw = lut_raw / OUTPUT_SCALE
    return silu_hw, lut_index, lut_raw


if __name__ == "__main__":
    sample_inputs = np.array([
        0.195254, 0.860757, 0.411054, 0.179533,
        -0.305381, 0.583576, -0.249651, 1.56709,
        1.85465, -0.466234, 1.1669, 0.11558,
        0.272178, 1.70239, -1.71586, -1.65148,
    ], dtype=np.float32)
    expected = sample_inputs / (1.0 + np.exp(-sample_inputs))

    print(f"Loaded SiLU LUT from {SILU_LUT_PATH} with {len(silu_lut)} entries")
    print("Testing hardware SiLU approximation (direct LUT read):")

    tol = 1.2e-2
    hw_vals = []
    for x, exp_val in zip(sample_inputs, expected):
        silu_hw, lut_idx, lut_raw = hardware_silu(x)
        hw_vals.append(silu_hw)
        offset_base, offset_frac = _lut_offset(abs(float(x)))
        offset = offset_base + offset_frac
        offset_signed = offset if x >= 0 else -offset
        neighbor_idx = lut_idx + (1 if x >= 0 else -1)
        lut_raw_u = lut_raw & 0xFFFF
        print(
            f"x={x: .6f} (fp16 0x{np.uint16(np.float16(x).view(np.uint16)).item():04x}) "
            f"idx={offset_signed:+6.2f} lut[{lut_idx}->{neighbor_idx}]=0x{lut_raw_u:04x} "
            f"raw={lut_raw:6d} silu_hw={silu_hw:.6f} ref={exp_val:.6f}"
        )

    hw_vals = np.array(hw_vals, dtype=np.float32)
    diffs = np.abs(hw_vals - expected)
    matches = np.all(diffs <= tol)
    print(f"Match expected within {tol}: {'YES' if matches else 'NO'} (max diff {diffs.max():.6f})")
