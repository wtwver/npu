import re
import numpy as np
from pathlib import Path

# NVDLA-style sigmoid LUT as captured on RK3588:
# - Dumped via REG_DPU_LUT_ACCESS_DATA into /tmp/sigmoid2
# - 1026 entries, centered at index 512 ~= sigmoid(0) = 0.5 (Q0.15)
# - Positive side covers roughly 0..+2 using a rounded divide-by-32 of a 2596x scale.
LUT_DUMP_PATH = Path("./sigmoid2")
LUT_SIZE = 1026
INDEX_SCALE = 2596.0          # multiplier applied to |x|
INDEX_SHIFT = 5               # divide by 32
ROUND_BIAS = 1 << (INDEX_SHIFT - 1)


def _parse_lut_from_dump(path: Path) -> np.ndarray:
    text = path.read_text().splitlines()
    vals = []
    for line in text:
        m = re.search(r"LUT_ACCESS_DATA[^,]*\((\d+)\)", line)
        if m:
            vals.append(int(m.group(1)))
    if len(vals) != LUT_SIZE:
        raise ValueError(f"Expected {LUT_SIZE} LUT entries, found {len(vals)} in {path}")
    return np.array(vals, dtype=np.uint16)


def _load_lut() -> np.ndarray:
    if not LUT_DUMP_PATH.exists():
        raise FileNotFoundError(f"LUT dump not found: {LUT_DUMP_PATH}")
    return _parse_lut_from_dump(LUT_DUMP_PATH)


sigmoid_lut = _load_lut()
CENTER_INDEX = (len(sigmoid_lut) - 1) // 2  # 512 for 1026-entry table
MAX_OFFSET = min(CENTER_INDEX, len(sigmoid_lut) - 1 - CENTER_INDEX)


def _lut_offset(x_abs: float) -> int:
    """Quantize |x| into a LUT offset using NVDLA-style rounding."""
    scaled = int(np.round(x_abs * INDEX_SCALE))
    idx = (scaled + ROUND_BIAS) >> INDEX_SHIFT
    if idx > MAX_OFFSET:
        idx = MAX_OFFSET
    return idx


def hardware_sigmoid(x_float: float) -> float:
    """Approximate sigmoid using the hardware LUT."""
    x_abs = abs(float(x_float))
    offset = _lut_offset(x_abs)
    lut_index = CENTER_INDEX + offset if x_float >= 0 else CENTER_INDEX - offset
    lut_val = int(sigmoid_lut[lut_index])
    return lut_val / float(1 << 15)


if __name__ == "__main__":
    sample_inputs = np.array([
        0.195254, 0.860757, 0.411054, 0.179533,
        -0.305381, 0.583576, -0.249651, 1.56709,
        1.85465, -0.466234, 1.1669, 0.11558,
        0.272178, 1.70239, -1.71586, -1.65148,
    ], dtype=np.float32)
    expected = np.array([
        0.548674, 0.702836, 0.601359, 0.544771,
        0.424233, 0.641871, 0.437914, 0.82741,
        0.864654, 0.38549, 0.762601, 0.528868,
        0.567637, 0.845815, 0.15241, 0.160924,
    ], dtype=np.float32)

    print(f"Loaded LUT from {LUT_DUMP_PATH} with {len(sigmoid_lut)} entries")
    print("Testing hardware sigmoid approximation (NVDLA-style indexing):")

    tol = 5e-3
    hw_vals = []
    for x in sample_inputs:
        x_q = int(np.round(abs(x) * INDEX_SCALE))
        lut_offset = (x_q + ROUND_BIAS) >> INDEX_SHIFT
        lut_index = CENTER_INDEX + lut_offset if x >= 0 else CENTER_INDEX - lut_offset
        lut_val = int(sigmoid_lut[lut_index])
        hw = lut_val / float(1 << 15)
        hw_vals.append(hw)
        ref = 1.0 / (1.0 + np.exp(-x))
        print(
            f"x={x: .6f} idx={lut_offset:3d} lut[{lut_index}]=0x{lut_val:04x} "
            f"hw={hw:.6f} ref={ref:.6f}"
        )

    hw_vals = np.array(hw_vals, dtype=np.float32)
    diffs = np.abs(hw_vals - expected)
    matches = np.all(diffs <= tol)
    print(f"Match expected within {tol}: {'YES' if matches else 'NO'} (max diff {diffs.max():.6f})")
