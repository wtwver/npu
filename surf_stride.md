# nvdla sw
explain how compiler calculate surf_stride

The compiler calculates surf_stride (surface stride) through a multi-layered approach involving tensor surface descriptors and engine-specific suggestions.

Core Calculation Method
The main entry point is TensorSurfaceDesc::surfaceStride() in Surface.cpp Surface.cpp:119-142 , which dispatches to specific surface type handlers based on the surface category:

Feature Data: Calculated by FeatureDataDesc::surfaceStride() Surface.cpp:341-344
Image Data: Returns 0 for image surfaces Surface.cpp:213-216
Bias/BatchNorm/Scale Data: Each has their own calculation methods
Feature Data Surface Stride Calculation
For feature data (the most common case), the calculation is straightforward:

return lineStride(tsd) * tsd->dimensions().h;
Where lineStride is calculated as Surface.cpp:336-339 :

return tsd->dimensions().w * channelsPerGroup(tsd) * tsd->surfaceFormat().bytesPerElement();
Engine Node Suggestions
Each engine node type provides suggestSurfaceStride() methods that:

Create a probe tensor surface descriptor with suggested dimensions
Reset the surface stride to force recalculation
Call the surface descriptor's surfaceStride() method
For example, in RubikNode.cpp RubikNode.cpp:139-164 :

surface::TensorSurfaceDesc probeTSD = *tsd;  
Dims4 surfDims = suggestSurfaceDims(tsd);  
probeTSD.setDimensions(surfDims);  
probeTSD.resetSurfaceStride();  
surfaceStride = probeTSD.surfaceStride();
Special Cases
Split Operations: Surface stride is propagated from upstream nodes to maintain consistency SplitOp.cpp:274-276
Image Surfaces: Always return 0 as surface stride Surface.cpp:213-216
Weight Data: Returns 0 for weight surfaces Surface.cpp:135
Runtime Representation
The calculated surface stride is stored in the runtime tensor descriptor as part of an 8-element stride array IRuntime.h:128 , where surface stride typically occupies index 2 for NCHW format.

Notes
The surface stride calculation is cached in each tensor surface descriptor (m_surface_stride) to avoid recalculation Surface.cpp:122-126 . The actual value depends on the surface format, dimensions, and bytes per element of the tensor data type.

Wiki pages you might want to explore:

NVDLA Software Overview (nvdla/sw)


# ONNC
explain how compiler calculate surf_stride
Fast
View as codemap
The compiler calculates surf_stride (surface stride) differently based on the data cube type in the NVIDIA DLA backend. The primary calculation happens in the NvDlaCubeInfo constructor.

Surface Stride Calculation by Cube Type
Feature Data Cubes (NVDLA_CUBE_FEATURE)
For feature data, the surface stride is calculated as:

stride_line    = UNIT_ALIGNMENT(dim_w * FEATURE_ATOM_CUBE_SIZE, 32);  
stride_surface = stride_line * dim_h;
NvDlaMeta.cpp:406-408

Weight Data Cubes (NVDLA_CUBE_WEIGHT)
For weight data, surface stride is set to 0:

stride_surface = 0;
NvDlaMeta.cpp:441-442

SDP Data Cubes
For Single Data Processor (SDP) operations, surface stride varies by data type:

One-byte ALU/MUL operations:

stride_surface = stride_line * dim_h;
NvDlaMeta.cpp:452-453

Both one-byte operations:

stride_surface = stride_line * dim_h;
NvDlaMeta.cpp:462-463

Two-byte ALU/MUL operations:

stride_surface = stride_line * dim_h;
NvDlaMeta.cpp:472-473

Both two-byte operations:

stride_surface = stride_line * dim_h;
NvDlaMeta.cpp:482-483

Alternative Calculation in genmem
The genmem module provides an alternative surface stride calculation for compact layouts:

int Cube::get_compact_surface_stride(TYPE type, int W, int H, int C)  
{  
  if (type == TYPE::PLANAR)  
    cout << "[Warning] Type::Planar don't have surface stride\n";  
  else  
    return get_compact_line_stride(type, W, H, C) * H;  
}
Cube.cpp:67-74

Usage in Operations
The calculated surface stride is used in various DLA operations like pooling, where it's assigned to the operation's surface descriptor:

avgpool_surf->src_data.surf_stride = X_cube.stride_surface;  
avgpool_surf->dst_data.surf_stride = Y_cube.stride_surface;
AveragePool.inc:78-79

Notes
The surface stride calculation is hardware-specific to NVIDIA DLA and depends on:

Data cube type (feature, weight, or SDP operands)
Tensor dimensions (width, height, channels)
Hardware constants like FEATURE_ATOM_CUBE_SIZE and alignment requirements
Memory layout considerations (compact vs standard layouts)
The calculation ensures proper memory addressing for the DLA hardware's memory access patterns.

Wiki pages you might want to explore:

NVIDIA DLA Backend (ONNC/onnc)