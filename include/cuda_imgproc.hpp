#pragma once
// cuda_imgproc — umbrella header.
//
// Host-only translation units (compiled with g++) get the Image container,
// PNG I/O and the CPU reference implementations. Translation units compiled
// by nvcc (__CUDACC__ defined) additionally get the CUDA core and the GPU
// kernels + host wrappers. One include works everywhere:
//
//     #include "cuda_imgproc.hpp"

#include "cuda_imgproc/core/image.hpp"

#ifdef __CUDACC__
#include "cuda_imgproc/core/cuda_check.cuh"
#include "cuda_imgproc/core/gpu_buffer.cuh"
#endif
