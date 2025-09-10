from rknnlite.api import RKNNLite
import numpy as np
import sys

rknn_lite = RKNNLite()
ret = rknn_lite.load_rknn(sys.argv[1])
if ret != 0:
    print(f"Failed to load RKNN model: {ret}")
    exit(1)

# Use correct input shape (1,1) as shown in ONNX model info
# The model expects TWO inputs for addition operation with 2D shape
input_data_1 = np.random.rand(1, 1).astype(np.float16)
input_data_2 = np.random.rand(1, 1).astype(np.float16)
print(f"Input 1 shape: {input_data_1.shape}, dtype: {input_data_1.dtype}")
print(f"Input 2 shape: {input_data_2.shape}, dtype: {input_data_2.dtype}")

ret = rknn_lite.init_runtime()
if ret != 0:
    print(f"Failed to init runtime: {ret}")
    exit(1)

try:
    # Provide both inputs as required by the addition model
    outputs = rknn_lite.inference(inputs=[input_data_1, input_data_2])
    if outputs is not None and len(outputs) > 0:
        print(f"Inference completed successfully!")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output type: {type(outputs[0])}")
        print(f"Output dtype: {outputs[0].dtype}")
        print(f"Output sample values: {outputs[0].flatten()[:5]}")  # Show first 5 values
        
        # Verify the addition operation
        expected_output = input_data_1 + input_data_2
        print(f"Expected output shape: {expected_output.shape}")
        print(f"Expected output sample values: {expected_output.flatten()[:5]}")
        
        # Check if results match (within tolerance for float16)
        if np.allclose(outputs[0], expected_output, rtol=1e-3, atol=1e-3):
            print("✅ Results match expected addition!")
        else:
            print("❌ Results don't match expected addition")
            
    else:
        print("Inference failed: outputs is None or empty")
except Exception as e:
    print(f"Inference error: {e}")

rknn_lite.release()