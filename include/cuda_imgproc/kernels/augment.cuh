#pragma once
// Stage 2 — geometric augmentation: flips, rotation, resize.
//
// Concepts introduced here:
//   * 2D grids and 2D blocks: images are 2D, so let the thread hierarchy be
//     2D too — thread (tx,ty) of block (bx,by) owns output pixel
//     (bx*16+tx, by*16+ty). 16x16 = 256 threads per block, same occupancy
//     math as stage 1's 1D blocks, but indices now match image geometry.
//   * inverse mapping: geometric transforms iterate over *output* pixels and
//     compute where each came from. Forward mapping (scatter) would leave
//     holes and race on writes; inverse mapping (gather) writes each output
//     exactly once — and coalesced.
//   * bilinear interpolation as a shared __device__ helper.
//   * the gather trade-off: output writes stay perfectly coalesced while
//     source reads may land anywhere. Reads go through the cache hierarchy
//     and neighboring threads still hit neighboring source pixels, so
//     locality (not perfect coalescing) saves the day.
//
// All ops here support 1- and 3-channel interleaved uint8 images.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "../core/cuda_check.cuh"
#include "../core/gpu_buffer.cuh"
#include "../core/image.hpp"

namespace cig {

namespace device_detail {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Bilinear interpolation over the 4 integer pixels around (xs, ys), channel c
// of an interleaved image; out-of-range coordinates clamp to the edge. Used
// by both rotate and resize. Mirrors cpu::detail::bilinear_sample term for
// term (the CPU reference must interpolate identically for the ±1 tolerance
// in the tests to be meaningful).
__device__ __forceinline__ std::uint8_t bilinear_sample(const std::uint8_t* __restrict__ src,
                                                        int w, int h, int ch, int c, float xs,
                                                        float ys) {
    xs = clampf(xs, 0.0f, static_cast<float>(w - 1));
    ys = clampf(ys, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(xs);  // truncation == floor: xs >= 0 after clamp
    const int y0 = static_cast<int>(ys);
    const int x1 = clampi(x0 + 1, 0, w - 1);
    const int y1 = clampi(y0 + 1, 0, h - 1);
    const float fx = xs - static_cast<float>(x0);
    const float fy = ys - static_cast<float>(y0);

    const float v00 = src[(static_cast<std::size_t>(y0) * w + x0) * ch + c];
    const float v10 = src[(static_cast<std::size_t>(y0) * w + x1) * ch + c];
    const float v01 = src[(static_cast<std::size_t>(y1) * w + x0) * ch + c];
    const float v11 = src[(static_cast<std::size_t>(y1) * w + x1) * ch + c];

    const float v = v00 * (1.0f - fx) * (1.0f - fy) + v10 * fx * (1.0f - fy) +
                    v01 * (1.0f - fx) * fy + v11 * fx * fy;
    return static_cast<std::uint8_t>(v + 0.5f);
}

}  // namespace device_detail

// flip_horizontal_kernel — output (x,y) gathers source (W-1-x, y).
//
// Geometry:  2D grid of 16x16 blocks, one thread per output pixel.
// Memory:    global only; N bytes in + N bytes out per channel.
//
// Writes are coalesced as always with gather. Reads walk each row backwards:
// a warp still touches one contiguous span of bytes per row (just reversed),
// so the same cache sectors are fetched — reversal costs nothing extra.
static __global__ void flip_horizontal_kernel(const std::uint8_t* __restrict__ src,
                                              std::uint8_t* __restrict__ dst, int w, int h,
                                              int ch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const std::size_t d = (static_cast<std::size_t>(y) * w + x) * ch;
    const std::size_t s = (static_cast<std::size_t>(y) * w + (w - 1 - x)) * ch;
    for (int c = 0; c < ch; ++c) dst[d + c] = src[s + c];
}

// flip_vertical_kernel — output (x,y) gathers source (x, H-1-y). Rows move
// wholesale, so reads and writes are both perfectly coalesced.
static __global__ void flip_vertical_kernel(const std::uint8_t* __restrict__ src,
                                            std::uint8_t* __restrict__ dst, int w, int h,
                                            int ch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const std::size_t d = (static_cast<std::size_t>(y) * w + x) * ch;
    const std::size_t s = (static_cast<std::size_t>(h - 1 - y) * w + x) * ch;
    for (int c = 0; c < ch; ++c) dst[d + c] = src[s + c];
}

// rotate_kernel — rotation by theta about the image center via inverse
// mapping: output pixel (x,y) samples source coordinates
//
//   xs =  cos(t)*(x-cx) + sin(t)*(y-cy) + cx
//   ys = -sin(t)*(x-cx) + cos(t)*(y-cy) + cy,   (cx,cy) = ((W-1)/2, (H-1)/2)
//
// (the *inverse* rotation matrix applied to output coordinates — rotating the
// sampling grid backwards by theta rotates the image forwards by theta).
// Non-integer sources are bilinearly interpolated; sources outside the image
// clamp to the edge. cos/sin are computed once on the host and passed in:
// every thread would otherwise burn two transcendentals on identical inputs.
//
// Geometry:  2D grid of 16x16 blocks, one thread per output pixel.
// Memory:    global only; 4 reads + 1 write per output byte.
static __global__ void rotate_kernel(const std::uint8_t* __restrict__ src,
                                     std::uint8_t* __restrict__ dst, int w, int h, int ch,
                                     float cos_t, float sin_t) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;
    const float xs = cos_t * dx + sin_t * dy + cx;
    const float ys = -sin_t * dx + cos_t * dy + cy;

    for (int c = 0; c < ch; ++c) {
        dst[(static_cast<std::size_t>(y) * w + x) * ch + c] =
            device_detail::bilinear_sample(src, w, h, ch, c, xs, ys);
    }
}

// resize_kernel — bilinear resize: output (x,y) samples source (x*sx, y*sy)
// with sx = srcW/dstW, sy = srcH/dstH (computed once on the host).
//
// Geometry:  2D grid of 16x16 blocks over the *destination* image.
// Memory:    global only.
//
// Output writes are coalesced (gather again); source reads are generally not:
// for a 4x downscale, adjacent threads read pixels 4 apart, so a warp's reads
// spread over 4x more sectors than a unit-stride walk would need. That is the
// accepted trade-off of inverse mapping — the scattered side sits behind the
// read cache, where oversized fetches may still be reused by the next row of
// threads, while the write side (which has no such safety net) stays perfect.
static __global__ void resize_kernel(const std::uint8_t* __restrict__ src, int sw, int sh,
                                     std::uint8_t* __restrict__ dst, int dw, int dh, int ch,
                                     float sx, float sy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    const float xs = static_cast<float>(x) * sx;
    const float ys = static_cast<float>(y) * sy;
    for (int c = 0; c < ch; ++c) {
        dst[(static_cast<std::size_t>(y) * dw + x) * ch + c] =
            device_detail::bilinear_sample(src, sw, sh, ch, c, xs, ys);
    }
}

// ---------------------------------------------------------- host wrappers

namespace detail {
// ceil-div grid for a 16x16 block over a w x h output.
inline dim3 grid2d(int w, int h) {
    return dim3((static_cast<unsigned>(w) + 15u) / 16u, (static_cast<unsigned>(h) + 15u) / 16u);
}
}  // namespace detail

inline void flip_horizontal(const GpuBuffer<std::uint8_t>& d_src, GpuBuffer<std::uint8_t>& d_dst,
                            int width, int height, int channels, cudaStream_t stream = 0) {
    const dim3 block(16, 16);
    flip_horizontal_kernel<<<detail::grid2d(width, height), block, 0, stream>>>(
        d_src.get(), d_dst.get(), width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}

inline void flip_vertical(const GpuBuffer<std::uint8_t>& d_src, GpuBuffer<std::uint8_t>& d_dst,
                          int width, int height, int channels, cudaStream_t stream = 0) {
    const dim3 block(16, 16);
    flip_vertical_kernel<<<detail::grid2d(width, height), block, 0, stream>>>(
        d_src.get(), d_dst.get(), width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}

inline void rotate(const GpuBuffer<std::uint8_t>& d_src, GpuBuffer<std::uint8_t>& d_dst,
                   int width, int height, int channels, float theta, cudaStream_t stream = 0) {
    // Same std::cos/std::sin float calls as the CPU reference, so both paths
    // hand bit-identical rotation coefficients to their inner loops.
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    const dim3 block(16, 16);
    rotate_kernel<<<detail::grid2d(width, height), block, 0, stream>>>(
        d_src.get(), d_dst.get(), width, height, channels, ct, st);
    CUDA_CHECK(cudaGetLastError());
}

inline void resize(const GpuBuffer<std::uint8_t>& d_src, int src_w, int src_h,
                   GpuBuffer<std::uint8_t>& d_dst, int dst_w, int dst_h, int channels,
                   cudaStream_t stream = 0) {
    const float sx = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float sy = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const dim3 block(16, 16);
    resize_kernel<<<detail::grid2d(dst_w, dst_h), block, 0, stream>>>(
        d_src.get(), src_w, src_h, d_dst.get(), dst_w, dst_h, channels, sx, sy);
    CUDA_CHECK(cudaGetLastError());
}

// Image-level convenience: allocate + upload + launch + download.

namespace detail {
template <typename LaunchFn>
inline Image run_same_size_op(const Image& img, LaunchFn launch) {
    GpuBuffer<std::uint8_t> d_in(img.size());
    GpuBuffer<std::uint8_t> d_out(img.size());
    d_in.copy_from_host(img.data.data(), img.size());
    launch(d_in, d_out);
    Image out(img.width, img.height, img.channels);
    d_out.copy_to_host(out.data.data(), out.size());
    return out;
}
}  // namespace detail

inline Image flip_horizontal(const Image& img) {
    return detail::run_same_size_op(img, [&](const GpuBuffer<std::uint8_t>& in,
                                             GpuBuffer<std::uint8_t>& out) {
        flip_horizontal(in, out, img.width, img.height, img.channels);
    });
}

inline Image flip_vertical(const Image& img) {
    return detail::run_same_size_op(img, [&](const GpuBuffer<std::uint8_t>& in,
                                             GpuBuffer<std::uint8_t>& out) {
        flip_vertical(in, out, img.width, img.height, img.channels);
    });
}

inline Image rotate(const Image& img, float theta) {
    return detail::run_same_size_op(img, [&](const GpuBuffer<std::uint8_t>& in,
                                             GpuBuffer<std::uint8_t>& out) {
        rotate(in, out, img.width, img.height, img.channels, theta);
    });
}

inline Image resize(const Image& img, int dst_w, int dst_h) {
    GpuBuffer<std::uint8_t> d_in(img.size());
    GpuBuffer<std::uint8_t> d_out(static_cast<std::size_t>(dst_w) * dst_h * img.channels);
    d_in.copy_from_host(img.data.data(), img.size());
    resize(d_in, img.width, img.height, d_out, dst_w, dst_h, img.channels);
    Image out(dst_w, dst_h, img.channels);
    d_out.copy_to_host(out.data.data(), out.size());
    return out;
}

}  // namespace cig
