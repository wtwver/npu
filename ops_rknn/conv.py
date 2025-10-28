import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("conv1d_simple.onnx")

x = np.array([[[1., 2., 3., 4., 5.]]], dtype=np.float32)  # [1,1,5]
w = np.array([[[1., 0., -1.]]], dtype=np.float32)         # [1,1,3]

y, = sess.run(None, {'input': x, 'weight': w})
print(y)  # [[[ -2., -2., -2. ]]]
