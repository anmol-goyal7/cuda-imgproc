#pragma once
// cuda_imgproc — umbrella header.
//
// Host-only translation units (compiled with g++) get the Image container,
// PNG I/O and the CPU reference implementations. Translation units compiled
// by nvcc (__CUDACC__ defined) additionally get the CUDA core and the GPU
// kernels + host wrappers. One include works everywhere:
//
//     #include "cuda_imgproc.hpp"

#include "cuda_imgproc/core/filters.hpp"
#include "cuda_imgproc/core/image.hpp"
#include "cuda_imgproc/cpu/ops.hpp"
#include "cuda_imgproc/cpu/ops_omp.hpp"

#ifdef __CUDACC__
#include "cuda_imgproc/core/cuda_check.cuh"
#include "cuda_imgproc/core/gpu_buffer.cuh"
#include "cuda_imgproc/kernels/augment.cuh"
#include "cuda_imgproc/kernels/color.cuh"
#include "cuda_imgproc/kernels/convolution.cuh"
#include "cuda_imgproc/kernels/histogram.cuh"
#endif
