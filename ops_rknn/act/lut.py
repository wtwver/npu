import numpy as np

LE_TBL_ENTRY = 512
INDEX_SELECT = 5
LUT_INPUT_SCALE = 1 << INDEX_SELECT

x = 4.231
print(f"original input x {x}\n")

x_fp16 = np.float16(x)
print(f"x in fp16 {x_fp16:.8f}   HEX {hex(x_fp16.view(np.uint16))}\n")

# set up LUT with alternating 0, 32
LUT = []
for _ in range(LE_TBL_ENTRY):
    LUT.append(0)
    LUT.append(32)
print(f"LUT[4]={LUT[4]}, LUT[5]={LUT[5]}")

# linear LUT index/fraction (roundoff uses input scale that cancels index_select)
x_scaled = float(x_fp16) * LUT_INPUT_SCALE
scaled = x_scaled / float(2 ** INDEX_SELECT)
index = int(np.floor(scaled))
frac = float(scaled - index)
print(f"scaled={scaled:.8f} index={index} frac={frac:.8f}\n")

y0 = float(LUT[index])
y1 = float(LUT[index + 1])
lut_interp = y0 * (1.0 - frac) + y1 * frac
print(f"LUT[{index}] * (1.0 - {frac}) + LUT[{index+1}] * {frac}={lut_interp:.6f}\n")

# implicit FP16 rounding as a carry: flip the fraction when LUT slope is negative
base_int = int(np.floor(float(x_fp16)))
direction = np.sign(y1 - y0)  # +1 for rising LUT, -1 for falling LUT
frac_dir = 0.5 * (1.0 + direction) * frac + 0.5 * (1.0 - direction) * (1.0 - frac)
round_step = int(np.rint(frac_dir))
out = float(base_int + round_step)
print(f"frac_dir={frac_dir:.8f} round_step={round_step} out={out:.6f}")
