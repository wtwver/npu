# ONNX Model Testing Results

**Total Models:** 79
**Passed:** 39
**Failed:** 40
**Success Rate:** 49.4%

## Failed Models
### arange_0.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/arange_0.onnx failed:This is an invalid model. In Node, ("", Arange, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Arange with domain_version of 10

### arange_1.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/arange_1.onnx failed:This is an invalid model. In Node, ("", Arange, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Arange with domain_version of 10

### arange_2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/arange_2.onnx failed:This is an invalid model. In Node, ("", Arange, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Arange with domain_version of 10

### celu.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/celu.onnx failed:This is an invalid model. In Node, ("", CELU, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for CELU with domain_version of 10

### dot.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/dot.onnx failed:This is an invalid model. In Node, ("", Dot, "", -1) : ("input1": tensor(float),"input2": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Dot with domain_version of 10

### empty_0.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/empty_0.onnx failed:This is an invalid model. In Node, ("", Empty, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Empty with domain_version of 10

### empty_1.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/empty_1.onnx failed:This is an invalid model. In Node, ("", Empty, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Empty with domain_version of 10

### empty_2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/empty_2.onnx failed:This is an invalid model. In Node, ("", Empty, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Empty with domain_version of 10

### exp2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/exp2.onnx failed:This is an invalid model. In Node, ("", Exp2, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Exp2 with domain_version of 10

### eye_0.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/eye_0.onnx failed:This is an invalid model. In Node, ("", Eye, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Eye with domain_version of 10

### eye_1.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/eye_1.onnx failed:This is an invalid model. In Node, ("", Eye, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Eye with domain_version of 10

### eye_2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/eye_2.onnx failed:This is an invalid model. In Node, ("", Eye, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Eye with domain_version of 10

### full_0.onnx
**Error:** [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Load model from models/full_0.onnx failed:Invalid model. Node input 'shape' is not a graph input, initializer, or output of a previous node.

### full_1.onnx
**Error:** [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Load model from models/full_1.onnx failed:Invalid model. Node input 'shape' is not a graph input, initializer, or output of a previous node.

### full_2.onnx
**Error:** [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Load model from models/full_2.onnx failed:Invalid model. Node input 'shape' is not a graph input, initializer, or output of a previous node.

### gelu.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/gelu.onnx failed:This is an invalid model. In Node, ("", Gelu, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Gelu with domain_version of 10

### gemm.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/gemm.onnx failed:This is an invalid model. In Node, ("", Gemm, "", -1) : ("input1": tensor(float),"input2": tensor(float),) -> ("output": tensor(float),) , Error Node with schema(::Gemm:9) has input size 2 not in range [min=3, max=3].

### hardsigmoid.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/hardsigmoid.onnx failed:This is an invalid model. In Node, ("", Hardsigmoid, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Hardsigmoid with domain_version of 10

### hardswish.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/hardswish.onnx failed:This is an invalid model. In Node, ("", Hardswish, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Hardswish with domain_version of 10

### linspace_0.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/linspace_0.onnx failed:This is an invalid model. In Node, ("", Linspace, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Linspace with domain_version of 10

### linspace_1.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/linspace_1.onnx failed:This is an invalid model. In Node, ("", Linspace, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Linspace with domain_version of 10

### linspace_2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/linspace_2.onnx failed:This is an invalid model. In Node, ("", Linspace, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Linspace with domain_version of 10

### log2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/log2.onnx failed:This is an invalid model. In Node, ("", Log2, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Log2 with domain_version of 10

### logsigmoid.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/logsigmoid.onnx failed:This is an invalid model. In Node, ("", Logsigmoid, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Logsigmoid with domain_version of 10

### matmul.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/matmul.onnx failed:This is an invalid model. In Node, ("", Matmul, "", -1) : ("input1": tensor(float),"input2": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Matmul with domain_version of 10

### mod.onnx
**Error:** [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Mod node. Name:'' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/math/element_wise_ops.cc:2111 void onnxruntime::mod_internal::CallModImpl<T, typename std::enable_if<std::is_floating_point<_Tp>::value, void>::type>::operator()(bool, onnxruntime::OpKernelContext*) const [with T = float; typename std::enable_if<std::is_floating_point<_Tp>::value, void>::type = void] fmod was false. fmod attribute must be true for floating point types


### ones_0.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/ones_0.onnx failed:This is an invalid model. In Node, ("", Ones, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Ones with domain_version of 10

### ones_1.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/ones_1.onnx failed:This is an invalid model. In Node, ("", Ones, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Ones with domain_version of 10

### ones_2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/ones_2.onnx failed:This is an invalid model. In Node, ("", Ones, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Ones with domain_version of 10

### quick_gelu.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/quick_gelu.onnx failed:This is an invalid model. In Node, ("", Quick_gelu, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Quick_gelu with domain_version of 10

### relu6.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/relu6.onnx failed:This is an invalid model. In Node, ("", Relu6, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Relu6 with domain_version of 10

### reshape.onnx
**Error:** [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(int32)) , expected: (tensor(int64))

### round.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/round.onnx failed:This is an invalid model. In Node, ("", Round, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Round with domain_version of 10

### rsqrt.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/rsqrt.onnx failed:This is an invalid model. In Node, ("", Rsqrt, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Rsqrt with domain_version of 10

### selu.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/selu.onnx failed:This is an invalid model. In Node, ("", SELU, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for SELU with domain_version of 10

### silu.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/silu.onnx failed:This is an invalid model. In Node, ("", Silu, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Silu with domain_version of 10

### trunc.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/trunc.onnx failed:This is an invalid model. In Node, ("", Trunc, "", -1) : ("input": tensor(float),) -> ("output": tensor(float),) , Error No Op registered for Trunc with domain_version of 10

### zeros_0.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/zeros_0.onnx failed:This is an invalid model. In Node, ("", Zeros, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Zeros with domain_version of 10

### zeros_1.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/zeros_1.onnx failed:This is an invalid model. In Node, ("", Zeros, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Zeros with domain_version of 10

### zeros_2.onnx
**Error:** [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from models/zeros_2.onnx failed:This is an invalid model. In Node, ("", Zeros, "", -1) : () -> ("output": tensor(float),) , Error No Op registered for Zeros with domain_version of 10

## Passed Models
### abs.onnx
**Inference Time:** 0.59ms
**Output Summary:**
- output: shape (1, 32, 32), range [0.002, 1.994]

### acos.onnx
**Inference Time:** 0.19ms
**Output Summary:**
- output: shape (1, 32, 32), range [nan, nan]

### adaptive_avg_pool2d.onnx
**Inference Time:** 0.22ms
**Output Summary:**
- output: shape (1, 3, 16, 16), range [-1.748, 1.631]

### adaptive_max_pool2d.onnx
**Inference Time:** 0.16ms
**Output Summary:**
- output: shape (1, 3, 16, 16), range [-1.239, 1.995]

### add.onnx
**Inference Time:** 0.15ms
**Output Summary:**
- output: shape (1, 32, 32), range [-3.885, 3.796]

### add_1.onnx
**Inference Time:** 0.13ms
**Output Summary:**
- output: shape (1, 1, 1), range [1.285, 1.285]

### asin.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 32, 32), range [nan, nan]

### atan.onnx
**Inference Time:** 0.15ms
**Output Summary:**
- output: shape (1, 32, 32), range [-1.105, 1.107]

### avg_pool2d.onnx
**Inference Time:** 0.17ms
**Output Summary:**
- output: shape (1, 3, 16, 16), range [-1.213, 1.172]

### cat.onnx
**Inference Time:** 0.18ms
**Output Summary:**
- output: shape (2, 32, 32), range [-2.000, 1.993]

### ceil.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 32, 32), range [-1.000, 2.000]

### conv1d.onnx
**Inference Time:** 0.17ms
**Output Summary:**
- output: shape (1, 16, 30), range [-10.293, 11.622]

### conv2d.onnx
**Inference Time:** 0.27ms
**Output Summary:**
- output: shape (1, 16, 30, 30), range [-29.663, 33.531]

### cos.onnx
**Inference Time:** 0.14ms
**Output Summary:**
- output: shape (1, 32, 32), range [-0.413, 1.000]

### div.onnx
**Inference Time:** 0.13ms
**Output Summary:**
- output: shape (1, 32, 32), range [-785.807, 160.404]

### elu.onnx
**Inference Time:** 0.11ms
**Output Summary:**
- output: shape (1, 32, 32), range [-0.864, 1.996]

### erf.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 32, 32), range [-0.995, 0.995]

### exp.onnx
**Inference Time:** 0.13ms
**Output Summary:**
- output: shape (1, 32, 32), range [0.135, 7.349]

### flatten.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 1024), range [-1.992, 1.995]

### floor.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 32, 32), range [-2.000, 1.000]

### leaky_relu.onnx
**Inference Time:** 0.13ms
**Output Summary:**
- output: shape (1, 32, 32), range [-0.200, 1.998]

### log.onnx
**Inference Time:** 0.14ms
**Output Summary:**
- output: shape (1, 32, 32), range [nan, nan]

### max_pool2d.onnx
**Inference Time:** 0.28ms
**Output Summary:**
- output: shape (1, 3, 16, 16), range [-0.246, 1.998]

### mul.onnx
**Inference Time:** 0.18ms
**Output Summary:**
- output: shape (1, 32, 32), range [-3.881, 3.735]

### neg.onnx
**Inference Time:** 0.14ms
**Output Summary:**
- output: shape (1, 32, 32), range [-2.000, 1.998]

### pow.onnx
**Inference Time:** 0.17ms
**Output Summary:**
- output: shape (1, 32, 32), range [nan, nan]

### reduce_max.onnx
**Inference Time:** 0.15ms
**Output Summary:**
- output: shape (1, 1, 1), range [1.999, 1.999]

### reduce_mean.onnx
**Inference Time:** 0.16ms
**Output Summary:**
- output: shape (1, 1, 1), range [0.013, 0.013]

### reduce_min.onnx
**Inference Time:** 0.14ms
**Output Summary:**
- output: shape (1, 1, 1), range [-1.997, -1.997]

### reduce_sum.onnx
**Inference Time:** 0.17ms
**Output Summary:**
- output: shape (1, 1, 1), range [-37.476, -37.476]

### relu.onnx
**Inference Time:** 0.24ms
**Output Summary:**
- output: shape (1, 32, 32), range [0.000, 1.999]

### sigmoid.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 32, 32), range [0.119, 0.880]

### sign.onnx
**Inference Time:** 0.12ms
**Output Summary:**
- output: shape (1, 32, 32), range [-1.000, 1.000]

### sin.onnx
**Inference Time:** 0.24ms
**Output Summary:**
- output: shape (1, 32, 32), range [-1.000, 1.000]

### softplus.onnx
**Inference Time:** 0.3ms
**Output Summary:**
- output: shape (1, 32, 32), range [0.127, 2.116]

### sqrt.onnx
**Inference Time:** 0.15ms
**Output Summary:**
- output: shape (1, 32, 32), range [nan, nan]

### sub.onnx
**Inference Time:** 0.17ms
**Output Summary:**
- output: shape (1, 32, 32), range [-3.811, 3.788]

### tan.onnx
**Inference Time:** 0.19ms
**Output Summary:**
- output: shape (1, 32, 32), range [-7748.337, 356.410]

### transpose.onnx
**Inference Time:** 0.17ms
**Output Summary:**
- output: shape (1, 32, 32), range [-1.994, 1.997]

