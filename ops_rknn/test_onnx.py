#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = 'conv2d_simple.onnx'
session = ort.InferenceSession(model_path)

# Create input data in NCHW format [1, 3, 5, 7]
input_data = np.array([
    # Channel 0
    1.,  2.,  3.,  4.,  5.,  6.,  7.,
    8.,  9., 10., 11., 12., 13., 14.,
    15., 16., 17., 18., 19., 20., 21.,
    22., 23., 24., 25., 26., 27., 28.,
    29., 30., 31., 32., 33., 34., 35.,
    # Channel 1
    36., 37., 38., 39., 40., 41., 42.,
    43., 44., 45., 46., 47., 48., 49.,
    50., 51., 52., 53., 54., 55., 56.,
    57., 58., 59., 60., 61., 62., 63.,
    64., 65., 66., 67., 68., 69., 70.,
    # Channel 2
    71., 72., 73., 74., 75., 76., 77.,
    78., 79., 80., 81., 82., 83., 84.,
    85., 86., 87., 88., 89., 90., 91.,
    92., 93., 94., 95., 96., 97., 98.,
    99.,100.,101.,102.,103.,104.,105.,
], dtype=np.float32).reshape(1, 3, 5, 7)

print(f"Input shape: {input_data.shape}")

# Run inference
outputs = session.run(None, {'input': input_data})
output_data = outputs[0]

print(f"Output shape: {output_data.shape}")
print(f"\nOutput (1x6x4x7):")
for oc in range(6):
    print(f"  Output Channel {oc}:")
    for h in range(4):
        print(f"    ", end='')
        for w in range(7):
            print(f"{output_data[0, oc, h, w]:8.1f} ", end='')
        print()
