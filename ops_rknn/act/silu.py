import re
import numpy as np
from pathlib import Path

# Use the same sigmoid LUT interpretation as sigmoid.py and apply x * sigmoid(x).
SIGMOID_LUT_PATH = Path("./sigmoid2")
SIGMOID_LUT_SIZE = 1026
INDEX_SCALE = 2596.0
INDEX_SHIFT = 5
ROUND_BIAS = 1 << (INDEX_SHIFT - 1)


def _parse_sigmoid_lut(path: Path) -> np.ndarray:
    text = path.read_text().splitlines()
    vals = []
    for line in text:
        m = re.search(r"LUT_ACCESS_DATA[^,]*\((\d+)\)", line)
        if m:
            vals.append(int(m.group(1)))
    if len(vals) != SIGMOID_LUT_SIZE:
        raise ValueError(f"Expected {SIGMOID_LUT_SIZE} LUT entries, found {len(vals)} in {path}")
    return np.array(vals, dtype=np.uint16)


def _load_sigmoid_lut() -> np.ndarray:
    if not SIGMOID_LUT_PATH.exists():
        raise FileNotFoundError(f"LUT dump not found: {SIGMOID_LUT_PATH}")
    return _parse_sigmoid_lut(SIGMOID_LUT_PATH)


sigmoid_lut = _load_sigmoid_lut()
CENTER_INDEX = (len(sigmoid_lut) - 1) // 2
MAX_OFFSET = min(CENTER_INDEX, len(sigmoid_lut) - 1 - CENTER_INDEX)


def _lut_offset(x_abs: float) -> int:
    scaled = int(np.round(x_abs * INDEX_SCALE))
    idx = (scaled + ROUND_BIAS) >> INDEX_SHIFT
    if idx > MAX_OFFSET:
        idx = MAX_OFFSET
    return idx


def hardware_sigmoid(x_float: float):
    x_abs = abs(float(x_float))
    offset = _lut_offset(x_abs)
    lut_index = CENTER_INDEX + offset if x_float >= 0 else CENTER_INDEX - offset
    lut_val = int(sigmoid_lut[lut_index])
    hw = lut_val / float(1 << 15)
    return hw, lut_index, lut_val


if __name__ == "__main__":
    # Reuse the same inputs as sigmoid.py for consistency.
    sample_inputs = np.array([
        0.195254, 0.860757, 0.411054, 0.179533,
        -0.305381, 0.583576, -0.249651, 1.56709,
        1.85465, -0.466234, 1.1669, 0.11558,
        0.272178, 1.70239, -1.71586, -1.65148,
    ], dtype=np.float32)
    expected = sample_inputs / (1.0 + np.exp(-sample_inputs))

    print(f"Loaded sigmoid LUT from {SIGMOID_LUT_PATH} with {len(sigmoid_lut)} entries")
    print("Testing hardware SiLU approximation (x * sigmoid_lut):")

    tol = 5e-3
    hw_vals = []
    for x, exp_val in zip(sample_inputs, expected):
        sig_hw, lut_idx, lut_val = hardware_sigmoid(x)
        hw = float(x) * sig_hw
        hw_vals.append(hw)
        offset = lut_idx - CENTER_INDEX
        print(
            f"x={x: .6f} (fp16 0x{np.uint16(np.float16(x).view(np.uint16)).item():04x}) "
            f"idx={offset:+4d} lut[{lut_idx}]=0x{lut_val:04x} "
            f"sig_hw={sig_hw:.6f} silu_hw={hw:.6f} ref={exp_val:.6f}"
        )

    hw_vals = np.array(hw_vals, dtype=np.float32)
    diffs = np.abs(hw_vals - expected)
    matches = np.all(diffs <= tol)
    print(f"Match expected within {tol}: {'YES' if matches else 'NO'} (max diff {diffs.max():.6f})")
