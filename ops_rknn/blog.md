Home
Blog/Gemlog
Public Services/Demo
whoami
Martin's blog/website thingy
Old blogGithubGemini
Benchmarking RK3588 NPU matrix multiplication performance
My goal with my RK3588 dev board is to eventually run a large language model on it. For now, it seems the RWKV model is the easiest to get running. At least I don't need to somehow get MultiHeadAttention to work on the NPU. However, during experimenting with rwkv.cpp, which uses pure CPU for inference on the RK3588, I noticed that quantization makes a huge difference in inference speed. So much so that INT4 about 4x faster then FP16. Not a good sign. It indicates to me that memory bandwidth is the bottleneck. If so, the NPU may not make much of a difference.

rwkv.cpp - INT4/INT5/INT8 and FP16 inference on CPU for RWKV language model
So, benchmarking time! First question I need answer is how fast the NPU can do matrix multiplication. According to the spec, the NPU on RK3588 can do 0.5 TFLOPS at FP16 under matrix multiplication. (Marketing materials says 6TOPS, but that only applies to INT4 and is doing convolution). Let's compare that against some baseline numbers like NumPy and PyTorch.

First, as of RKNN2 v1.5.0. The only way to get the NPU to do stuff is via ONNX. The low-level matrix multiplication API is completely broken. So I wrote a quck script that generates ONNX models that multiplies an input with a random matrix. Then the model exported to RKNN.

# part of genmatmul.py
def make_matmul(M, N, K):
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['C'],
    )
    b_mat = np.random.randn(N, K).astype(np.float32)
    b_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['B'],
        value=helper.make_tensor(
            name='B',
            data_type=onnx.TensorProto.FLOAT,
            dims=b_mat.shape,
            vals=b_mat.flatten(),
        ),
    )
    graph = helper.make_graph(
        nodes=[b_node, matmul_node],
        name='matmul',
        inputs=[
            helper.make_tensor_value_info(
                'A',
                onnx.TensorProto.FLOAT,
                [M, N],
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                'C',
                onnx.TensorProto.FLOAT,
                [M, K],
            ),
        ],
    )

    model = helper.make_model(graph, producer_name='onnx-matmul')
    onnx.checker.check_model(model)
    model = version_converter.convert_version(model, 14)
    return model
With the files generated, I can now load the model into RKNN and run it. According to np.show_config(), my installation uses LAPACK for BLAS, which is quite slow. The CPU on the RK3588 contains 2 Cortex-A73 and 6 Cortex-A53. I aslo tried running with TinyGrad, when has OpenCL support. However, it seems TinyGrad is not optimized enough to beat PyTorch (running on the CPU!!).

Bar graph comparing RKNN and NumPy matrix multiplication performance
Image: Bar graph comparing RKNN and NumPy matrix multiplication performance
According to the result. RKNN is indeed the fastest in the bunch. However, there's a big caveat, RKNN internally runs at FP16 even though the original ONNX graph is FP32. Yet non of the other libraries does matrix multiplication at FP16. Thus this result is more like comparing oranges to tangerines. They are similar, but not quite the same. Also, according to the graph, the NPU only starts winning PyTorch and TinyGrad somewhere between N = 2048 and 4096. I have to assume this is some overhead in RKNN. Which I really hope they can resolve by fixing the low-level API.

Another question I'm asking myself. How well does the NPU scale with matrix size? To answer this, I generated a bunch of ONNX models with different matrix sizes. Starting from 128 and increment by 128 until 4096. The blue line is the measured time it takes to run matrix multiplication on the NPU. And the red line is simply the blue line fitted with a exponential curve (since matrix multiplication is ~O(n^3)).

Line graph showing the time it takes to run matrix multiplication on the NPU
Image: Line graph showing the time it takes to run matrix multiplication on the NPU
And the fit log from ROOT:

****************************************
Minimizer is Minuit / Migrad
Chi2                      =    0.0840229
NDf                       =           29
Edm                       =  5.32403e-09
NCalls                    =           68
Constant                  =     -2.98728   +/-   0.0811982   
Slope                     =  0.000918918   +/-   2.29814e-05 
For me, this graph shows 2 properties that I'm very intrigued with.

There's no sudden jump in time. The NPU doesn't seem to cache anything. Likely it does chunked fetches from memory. And a local buffer holds it as long as needed.
However, there is a few steps in the curve. Maybe they are using different strategies for different matrix sizes.
For piratical sizes, the NPU takes really little time. Good news for me trying to run RWKV.
Matrix multiplication doesn't scale well. We will run into issues if we try to widen language models.
That's everything I care enough to test. See ya.

Author's profile. Made my my friend.
Martin Chang
Systems software, HPC, GPGPU and AI. I mostly write stupid C++ code. Sometimes does AI research. Chronic VRChat addict
I run TLGS, a major search engine on Gemini. Used by Buran by default.


 martin \at clehaxze.tw
 Matrix: @clehaxze:matrix.clehaxze.tw
 Jami: a72b62ac04a958ca57739247aa1ed4fe0d11d2df
© Copyright Martin Chang 2020-2024 In case of a security issue. Please refer to the security.txt file for contact and encryption information.
The content on this site is all CC-BY-SA (or MIT/GPLv3+ dual license for code)


Home
Blog/Gemlog
Public Services/Demo
whoami
Martin's blog/website thingy
Old blogGithubGemini
Benchmarking RK3588 NPU matrix multiplication performance EP2
Not long after my last benchmarking attempt. Rockchip releases a SDK update that fixes the crashing matrix multiplication API. Now I'm no longer restricted to using ONNX. Now I can directly do matrix multiplication from C! And now I can do an apple to apple comparison with OpenBLAS. That's benchmarking. Actually, I knew the SDK update days before writing this post. But I held on because I'm working on something more exiting - porting Large Language Models to run on RK3588 NPU. The result is.. well, you'll see. I also got an oppertunity to speak at Skymizer's interal tech forum because of my work. I'll share the side deck after I gave the talk.

Article of my last benchmarking attempt
RKNPU MatMul API
You can fine the MatMul API in the following link. The example proided is intimidating.. but it's actually not that hard to use once you get the hang of it. The API is very similar to older API designs like OpenGL.

rockchip-linux/rknpu2 Matrix multiplication API demo
First, you need to setup a descriptor of the kind of multiplication you want to do. This contains the shape of the input and output matrices. And the data type. Futhermore, it also contains the "foramt" of the input and output matrices. Currently only float16 and int8 are supported.

rknn_matmul_info info;
info.M             = M;
info.K             = K;
info.N             = N;
info.type          = RKNN_TENSOR_FLOAT16;
info.native_layout = 0;
info.perf_layout   = 0;
Shape of the matrix is pretty self explanatory if you ever used BLAS before. If not, consider the following Python code. 2 matrices are created. One is 64x128, the other is 128x256. According to the rule of matrix multiplication, the result should have a shape of 64x256. In code, this is represented by the M, K, N variables. Where M=64, K=128, N=256 in this case. Or have a look at the diagram below.

a = np.random.rand(64, 128)
b = np.random.rand(128, 256)
c = np.matmul(a, b)
The dimension of matrix multiplication
Image: The dimension of matrix multiplication
native_layout and perf_layout are the "format" of the input and output matrices. When set to 0, the matrix multiplication API works like you expected. The input and output matrices are stored in row-major order. However, the NPU does not perform well when multiplying matrices in row-major order. The runtime will reorder the matrices to an internal format sutible for the NPU before sending it for processing. This can be slow if you plan of multiplying multiple different matrices. Setting them to 1 will tell the runtime that you have already reordered the matrices and it can skip the reordering step.

We created the descriptor. Now we need to create the actual matrix multiplication object. This is done by calling rknn_matmul_create. The function takes the descriptor and gives you back a handle to the matrix multiplication object. And an IO descriptor. With these, we can call rknn_create_mem to allocate memory for the input and output matrices. Then rknn_matmul_set_io_mem to tell the matrix multiplication object where the input and output matrices are located.

rknn_matmul_io_attr io_attr;
rknn_matmul_ctx ctx;
rknn_matmul_create(&ctx, &info, &io_attr);

rknn_tensor_mem* A = rknn_create_mem(ctx, io_attr.A.size);
rknn_tensor_mem* B = rknn_create_mem(ctx, io_attr.B.size);
rknn_tensor_mem* C = rknn_create_mem(ctx, io_attr.C.size);

// For demo, we initialize the input matrices with 0
memcpy(A->virt_addr, a, io_attr.A.size);
memcpy(B->virt_addr, b, io_attr.B.size);

rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
Finally, call rknn_matmul_run to perform the matrix multiplication. After the call, the result will be stored in the memory pointed by C->virt_addr.

rknn_matmul_run(ctx);

float* result = (float*)C->virt_addr;
// Do something with the result
Updating values for the input matrices
One thing to note is that the driver seems to create a copy of the input matrices. So if you want to update the values of the input matrices, you need to call rknn_matmul_set_io_mem again. Otherwise, the values will not be updated.

// Update the values of the input matrices
typedef __fp16 float16; // GCC 16-bit float extension
for (int i = 0; i < io_attr.A.size / sizeof(float); i++) {
    ((float16*)A->virt_addr)[i] = i;
}

// Remember to call rknn_matmul_set_io_mem again. Otherwise, the values will not be updated
rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
Benchmarking
That's the gist of the API. Now let's benchmark it. The benchmarking code is pretty much the same as the last time. Except it's in C now and I have finer control over how it gets executed.

Matrix multiplication speed comparison between RKNN and OpenBLAS
Image: Matrix multiplication speed comparison between RKNN and OpenBLAS
The source code can be found in the following link.

Benchmarking code for RKNN matrix multiplication
Author's profile. Made my my friend.
Martin Chang
Systems software, HPC, GPGPU and AI. I mostly write stupid C++ code. Sometimes does AI research. Chronic VRChat addict
I run TLGS, a major search engine on Gemini. Used by Buran by default.


 martin \at clehaxze.tw
 Matrix: @clehaxze:matrix.clehaxze.tw
 Jami: a72b62ac04a958ca57739247aa1ed4fe0d11d2df
© Copyright Martin Chang 2020-2024 In case of a security issue. Please refer to the security.txt file for contact and encryption information.
The content on this site is all CC-BY-SA (or MIT/GPLv3+ dual license for code)

Home
Blog/Gemlog
Public Services/Demo
whoami
Martin's blog/website thingy
Old blogGithubGemini
Benchmarking RK3588 NPU matrix multiplication performance EP3
Today is the last day of CNY and being honest, I have nothing to do. Out of nowhere, I decided to look deeper into RK3588's NPU performance characteristics. To figure out what it actually needs to be performant. Like how batch size and native/normal layout affects the performance. I haven't done any of these before, as llama.cpp only supports one configuration, the normal layout with batch size of 1. But knowing more could unlock more potential right?

You can find the previous posts on the topic here:

Benchmarking RK3588 NPU matrix multiplication performance EP2
Benchmarking RK3588 NPU matrix multiplication performance
The source code for the benchmarking tool is available here:

My RK3588 NPU matrix multiplication benchmarking tool
Some concepts for the Rockchip matrix multiplication API
The matrix multiplication API specifies quite many things. The list includes:

The shape of the two input matrices (M, N, K)
The type of the input matrices (float16, int8, int4)
The layout of the input matrix A and output matrix C (normal/native)
The layout of the input matrix B (normal/native)
Matrix shapes and types are quite self-explanatory. Matrix layout needs some explanation. The normal layout is the row-major layout that we are all used to. The native layout is specific to the NPU and and basically interleaves the rows of the matrix. The header file not the documents specifies how it impacts the performance. Which I find out is it has major impact.

Results
The benchmarking tool generates a 11MB CSV that I ran some analysis on. The results are quite informative. The analysis script is written in C++ and the ROOT data analysis framework. In case you are wondering.

Best case performance
My first question coming into mind is how fast exactly is the NPU. I got bit by the NPU's performance, as of now I can't get llama.cpp to run meaningfully faster on the NPU vs the CPU. And I've seen reports saying the NPU tops at 10GFLOPS FP16 in practice. How fast is it really? Turns out the NPU can get to 900 GFLOPS FP16 doing an 128x1024x8192 matrix multiplication. That's close to the theoretical peak of 1 TFLOPS. But the performance quickly drops with K getting larger and large.

I suspect the NPU can only hold so many data in its SRAM. And the performance drop off is due spilling to the DDR.

The NPU reaching 900 GFLOPS FP16
Image: The NPU reaching 900 GFLOPS FP16
INT8 performance is as expected, 2x of FP16. And the performance drop off along the K axis is not as severe. This time peaking at ~1.8 TOPS around 128x1024x8192. INT4 is another 2x of INT8, peaking at almost 4 TOPS.

The NPU reaching 1.8 TOPS INT8
Image: The NPU reaching 1.8 TOPS INT8
The NPU reaching almost 4 TOPS INT4
Image: The NPU reaching almost 4 TOPS INT4
Suffering from low M and memory bound
In the majority of the cases, llama.cpp will be doing GEMV instead of GEMM. Where the M dimension is 1. I suspect the NPU is going to suffer from this. And the results are as expected. The performance drops off significantly when M is low. The M==1 data point isn't visible on all 3 lines.

Performance graph with M=1 but K == N == 4096.
Image: Performance graph with M=1 but K == N == 4096.
Interestingly, the performance graph looks exactly like the compute bound vs memory bound graph. The performance first increases linearly, then flattens out. However the ratios between the lines doesn't match what memory bounded would look like. I suspect there are inefficiencies in the NPU's memory access pattern. Or compute units are not fully utilized.

The roofline model (Wikipedia, licensed under CC SA 4.0)
Image: The roofline model (Wikipedia, licensed under CC SA 4.0)
However, the prformance does not drop any bit if matrix A is native or not when M == 1. I suspect this is a special case in their SDK since for this specific case, the actual data remains the same no matter the layout.

Normal layout with M=1 topping at 11 GFLOPS
Image: Normal layout with M=1 topping at 11 GFLOPS
Either layout with M=1 reaching the same performance (not the best plot ever, but I'm too lazy to fix it)
Image: Either layout with M=1 reaching the same performance (not the best plot ever, but I'm too lazy to fix it)
Doing further digging. I find that the SDK reorders matrix A during rknn_matmul_set_io_mem if the layout is normal. But it does spend time converting matrix C from native to normal layout. Applications will want to change matrix A often, so the performance impact will become even more significant.

Native matrix A, C is critical to performance
The NPU's performance is significantly impacted by the layout of the input matrix A (and by extension, the output matrix C). As seen above, the NPU can reach 900 GFLOPS FP16 with native layout. But the peak performance drops to ~750 GFLOPS with normal layout. And the peak shifted.

Lower performance with normal layout
Image: Lower performance with normal layout
Matrix B being normal/native does not affect matrix multiplication, but impacts initialization time
Finally, the layout of matrix B does not impact the performance of the matrix multiplication. But it does impact the initialization time. Same reason as using the normal layout for matrix A. The SDK reorders matrix B during rknn_set_io_mem. However, most applications will happily use the same matrix B for many matrix multiplications. So only the initialization time is impacted.

Matrix B layout does not impact performance
Image: Matrix B layout does not impact performance
However is impacts initialization time
Image: However is impacts initialization time
Conclusion
I don't know what would be helpful advice based on this data. Hardware vendors should optimize GEMV I guess. Pushing GEMV to also run at 900 GFLOPS would be a huge win. But there's nothing much I can do with current hardware. There's generally a few things I can do. But non of them apply to LLaMA or GGML without major effort.

Develop network architectures that uses 1024x8192 matrix multiplications for every batch.
Optimize and work around the input reordering. This is done by useful-transformers now. But it's a major pain to do.
I'll see what I come up with. But I don't like the picture I'm seeing. Hopefully Rockchip can fix this with some driver magic.

Author's profile. Made my my friend.
Martin Chang
Systems software, HPC, GPGPU and AI. I mostly write stupid C++ code. Sometimes does AI research. Chronic VRChat addict
I run TLGS, a major search engine on Gemini. Used by Buran by default.


 martin \at clehaxze.tw
 Matrix: @clehaxze:matrix.clehaxze.tw
 Jami: a72b62ac04a958ca57739247aa1ed4fe0d11d2df
© Copyright Martin Chang 2020-2024 In case of a security issue. Please refer to the security.txt file for contact and encryption information.
The content on this site is all CC-BY-SA (or MIT/GPLv3+ dual license for code)