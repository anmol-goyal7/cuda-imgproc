#pragma once
// Stage 1 — color conversion. The "hello world" of CUDA image processing.
//
// Concepts introduced here:
//   * the 1D grid-of-blocks thread hierarchy and the global thread index
//     i = blockIdx.x * blockDim.x + threadIdx.x
//   * boundary checks (the grid is rounded up to whole blocks, so trailing
//     threads must do nothing)
//   * memory coalescing: why "thread i touches pixel i" is the fastest
//     possible access pattern
//   * warp divergence (rgb_to_hsv's data-dependent branches)
//
// Both kernels are *memory-bound*: a handful of multiplies per pixel is far
// cheaper than moving the pixel across the memory bus, so their speed is set
// by DRAM bandwidth, not FLOPs. That makes access pattern everything.

#include <cuda_runtime.h>

#include <cstdint>

#include "../core/cuda_check.cuh"
#include "../core/gpu_buffer.cuh"
#include "../core/image.hpp"

namespace cig {

// rgb_to_gray_kernel — BT.601 luma:  Y = 0.299 R + 0.587 G + 0.114 B.
//
// Geometry:  1D grid, blockDim.x = 256; thread i handles pixel i.
// Memory:    global only.
// Traffic:   3 bytes read + 1 byte written per pixel, i.e. 4N bytes total —
//            the streaming minimum for this op.
//
// Coalescing: consecutive threads read consecutive addresses. The 32 threads
// of a warp read bytes [96t, 96t+95] of the RGB buffer — three full 32-byte
// sectors with every byte consumed — and write bytes [32t, 32t+31] of the
// gray buffer, exactly one sector. The hardware merges each warp's accesses
// into these few wide transactions; scattered accesses would instead cost one
// transaction per thread and throughput would collapse by an order of
// magnitude.
//
// The luma is evaluated as fmaf(0.299,R, fmaf(0.587,G, 0.114*B)). fma is
// exactly specified by IEEE 754 (one rounding, not two), so the CPU reference
// computes bit-identical results with std::fmaf — which is what lets the test
// suite require *exact* GPU-vs-CPU agreement. Plain `a*b + c` would leave
// each compiler free to fuse (or not) and the two platforms could disagree
// by one bit, i.e. one gray level at the rounding boundary.
//
// Kernels are `static` because this is a header-only library: each
// translation unit that includes this header gets its own private kernel
// instance, so two .cu files in one program never collide at link time.
static __global__ void rgb_to_gray_kernel(const std::uint8_t* __restrict__ rgb,
                                          std::uint8_t* __restrict__ gray, int n_pixels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;  // grid is rounded up to whole blocks

    const float r = rgb[3 * i + 0];
    const float g = rgb[3 * i + 1];
    const float b = rgb[3 * i + 2];
    const float y = fmaf(0.299f, r, fmaf(0.587f, g, 0.114f * b));
    gray[i] = static_cast<std::uint8_t>(y + 0.5f);  // round to nearest
}

// rgb_to_hsv_kernel — RGB -> HSV, OpenCV's uint8 encoding: H = degrees/2 in
// [0,180), S and V scaled to [0,255].
//
// Geometry:  1D grid, blockDim.x = 256; thread i handles pixel i.
// Memory:    global only. 3 bytes in, 3 bytes out per pixel.
//
// Warp divergence: which branch computes the hue depends on which channel is
// the max — a data-dependent condition. When threads of one warp disagree,
// the hardware runs each taken branch serially with the other threads masked
// off (worst case here: all three branches, ~3x the branch cost). On natural
// images neighboring pixels usually share a dominant channel, so most warps
// are uniform and the average penalty is small; on random noise it is
// maximal. Either way the branches are a few FLOPs inside a memory-bound
// kernel, so the divergence rarely shows up in wall time — worth knowing,
// not worth contorting the code to avoid.
static __global__ void rgb_to_hsv_kernel(const std::uint8_t* __restrict__ rgb,
                                         std::uint8_t* __restrict__ hsv, int n_pixels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;

    const int r = rgb[3 * i + 0];
    const int g = rgb[3 * i + 1];
    const int b = rgb[3 * i + 2];
    const int v = r > g ? (r > b ? r : b) : (g > b ? g : b);
    const int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
    const int diff = v - mn;

    std::uint8_t s8 = 0;
    if (v != 0) {
        s8 = static_cast<std::uint8_t>(255.0f * static_cast<float>(diff) /
                                           static_cast<float>(v) +
                                       0.5f);
    }

    std::uint8_t h8 = 0;
    if (diff != 0) {
        float hue;
        if (v == r) {
            hue = 60.0f * static_cast<float>(g - b) / static_cast<float>(diff);
        } else if (v == g) {
            hue = 120.0f + 60.0f * static_cast<float>(b - r) / static_cast<float>(diff);
        } else {
            hue = 240.0f + 60.0f * static_cast<float>(r - g) / static_cast<float>(diff);
        }
        if (hue < 0.0f) hue += 360.0f;
        int hh = static_cast<int>(hue * 0.5f + 0.5f);  // degrees/2 -> [0,180)
        if (hh >= 180) hh -= 180;                      // hue just below 360° rounds to 180: wrap
        h8 = static_cast<std::uint8_t>(hh);
    }

    hsv[3 * i + 0] = h8;
    hsv[3 * i + 1] = s8;
    hsv[3 * i + 2] = static_cast<std::uint8_t>(v);
}

// ---------------------------------------------------------- host wrappers
//
// Device-level wrappers: caller owns the buffers, nothing is transferred.
// The launch is asynchronous; CUDA_CHECK(cudaGetLastError()) catches launch
// *configuration* errors immediately (execution errors surface at the next
// synchronizing call).

inline void rgb_to_gray(const GpuBuffer<std::uint8_t>& d_rgb, GpuBuffer<std::uint8_t>& d_gray,
                        int width, int height, cudaStream_t stream = 0) {
    const int n = width * height;
    const int block = 256;
    const int grid = (n + block - 1) / block;  // ceil-div: cover the tail
    rgb_to_gray_kernel<<<grid, block, 0, stream>>>(d_rgb.get(), d_gray.get(), n);
    CUDA_CHECK(cudaGetLastError());
}

inline void rgb_to_hsv(const GpuBuffer<std::uint8_t>& d_rgb, GpuBuffer<std::uint8_t>& d_hsv,
                       int width, int height, cudaStream_t stream = 0) {
    const int n = width * height;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    rgb_to_hsv_kernel<<<grid, block, 0, stream>>>(d_rgb.get(), d_hsv.get(), n);
    CUDA_CHECK(cudaGetLastError());
}

// Image-level convenience: allocate, upload, launch, download. The final
// copy_to_host needs no explicit synchronize: on the default stream a
// blocking memcpy is ordered after the kernel and returns only when the data
// is on the host.

inline Image rgb_to_gray(const Image& img) {
    GpuBuffer<std::uint8_t> d_in(img.size());
    GpuBuffer<std::uint8_t> d_out(img.n_pixels());
    d_in.copy_from_host(img.data.data(), img.size());
    rgb_to_gray(d_in, d_out, img.width, img.height);
    Image out(img.width, img.height, 1);
    d_out.copy_to_host(out.data.data(), out.size());
    return out;
}

inline Image rgb_to_hsv(const Image& img) {
    GpuBuffer<std::uint8_t> d_in(img.size());
    GpuBuffer<std::uint8_t> d_out(img.size());
    d_in.copy_from_host(img.data.data(), img.size());
    rgb_to_hsv(d_in, d_out, img.width, img.height);
    Image out(img.width, img.height, 3);
    d_out.copy_to_host(out.data.data(), out.size());
    return out;
}

}  // namespace cig
