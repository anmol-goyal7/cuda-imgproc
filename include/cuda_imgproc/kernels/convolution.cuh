#pragma once
// Stage 3 — 2D convolution: shared-memory tiling + a constant-memory filter
// bank, with a deliberately naive twin for comparison.
//
// Concepts introduced here:
//   * __shared__ memory as a block-local, program-managed cache
//   * cooperative loading with a halo, and the __syncthreads() barrier
//   * __constant__ memory and its broadcast path
//   * data reuse arithmetic: counting global-memory transactions saved
//
// Why tiling? Convolution has heavy *overlap*: neighboring output pixels read
// nearly the same k x k input window, so each input byte is needed by up to
// k^2 different threads. Reading it k^2 times from DRAM is waste; instead a
// block stages the input region it collectively needs into shared memory
// (on-chip, ~100x lower latency, per-block visibility) and every thread reads
// its window from there.
//
// The arithmetic for TILE = 16, k = 5 (radius r = 2):
//   * one block computes a 16x16 output tile = 256 threads;
//   * it needs the (16+2r)x(16+2r) = 20x20 = 400-byte input tile (the 16x16
//     core plus an r-wide "halo" ring — the borders of the window of the
//     tile-edge pixels);
//   * cooperative load: 400 global reads per block, coalesced row by row;
//   * naive version: 256 threads x 25 reads = 6400 global reads per block.
//   400 vs 6400 -> a 16x reduction in global-memory traffic. The naive twin
//   kernel exists so the benchmark can measure what that reduction is worth
//   on real hardware (L1/L2 caches absorb some of the redundancy, so the
//   measured gap is smaller than 16x — that, too, is the lesson).
//
// Why __constant__ memory for the weights? Every thread of every warp reads
// c_filter[i*k+j] at the same (i,j) at the same time — a warp-uniform
// address. Constant memory is backed by a small broadcast cache built for
// exactly that: one fetch serves all 32 lanes in a cycle. The weights would
// pollute shared memory (they never change per-block) and waste registers if
// passed by value; __constant__ is their natural tier.

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "../core/cuda_check.cuh"
#include "../core/filters.hpp"
#include "../core/gpu_buffer.cuh"
#include "../core/image.hpp"

namespace cig {

// One filter slot, sized for the largest supported kernel (7x7). 196 bytes of
// the 64 KB constant budget. `static` = one private copy per translation
// unit; the upload/launch helpers below are `static inline` for the same
// reason, so a TU's wrappers always talk to that TU's own filter slot.
static __constant__ float c_filter[kMaxFilterSize * kMaxFilterSize];

// convolve2d_tiled_kernel<TILE> — shared-memory tiled convolution,
// single-channel uint8, clamp-to-edge borders, weights in c_filter.
//
// Geometry:  2D grid of TILE x TILE blocks; thread (tx,ty) of block (bx,by)
//            computes output pixel (bx*TILE+tx, by*TILE+ty).
// Memory:    __shared__ input tile (TILE+2r)^2 bytes, __constant__ weights,
//            coalesced global writes.
// Traffic:   (TILE+2r)^2 global reads per block vs TILE^2 * k^2 naive —
//            ~1.56 vs 25 reads per output pixel at TILE=16, k=5.
//
// The shared array is statically sized for the largest supported radius; the
// live portion, (TILE+2r)^2 for the runtime k, is indexed with runtime
// tile_w. 484 bytes of shared memory per block is nowhere near the ~48-100 KB
// per-SM limit, so occupancy is unaffected.
template <int TILE = 16>
__global__ void convolve2d_tiled_kernel(const std::uint8_t* __restrict__ src,
                                        std::uint8_t* __restrict__ dst, int w, int h, int k) {
    constexpr int kMaxRadius = kMaxFilterSize / 2;
    __shared__ std::uint8_t tile[(TILE + 2 * kMaxRadius) * (TILE + 2 * kMaxRadius)];

    const int r = k / 2;
    const int tile_w = TILE + 2 * r;

    // Top-left of the shared tile in image coordinates. blockIdx is unsigned:
    // cast before subtracting r or the arithmetic wraps at block (0,0).
    const int origin_x = static_cast<int>(blockIdx.x) * TILE - r;
    const int origin_y = static_cast<int>(blockIdx.y) * TILE - r;

    // Cooperative load. The tile has more elements than the block has
    // threads ((TILE+2r)^2 vs TILE^2), so each thread loads up to two: a
    // flat strided loop is simpler and no slower than a "core + edges"
    // special case. Out-of-image halo cells clamp to the nearest edge pixel.
    //
    // Every thread participates — including threads whose *output* pixel
    // falls outside the image (blocks straddling the right/bottom edge).
    // Returning early before a __syncthreads() some warps still expect to
    // reach would deadlock the block; the output bounds check must come
    // *after* the barrier.
    const int tid = static_cast<int>(threadIdx.y) * TILE + static_cast<int>(threadIdx.x);
    for (int idx = tid; idx < tile_w * tile_w; idx += TILE * TILE) {
        const int lx = idx % tile_w;
        const int ly = idx / tile_w;
        int gx = origin_x + lx;
        int gy = origin_y + ly;
        gx = gx < 0 ? 0 : (gx >= w ? w - 1 : gx);  // clamp to edge
        gy = gy < 0 ? 0 : (gy >= h ? h - 1 : gy);
        tile[idx] = src[static_cast<std::size_t>(gy) * w + gx];
    }

    // Barrier: no thread may read the tile until every thread finished
    // writing it. Shared memory has no implicit ordering between threads —
    // this is the synchronization that makes the cooperative cache coherent.
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x) * TILE + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * TILE + static_cast<int>(threadIdx.y);
    if (x >= w || y >= h) return;

    // Output pixel (x,y)'s window starts at image coords (x-r, y-r), which is
    // tile coords (tx, ty) exactly — the halo offset cancels the -r.
    // Accumulate in float with explicit fma, rows outer / columns inner: the
    // identical sequence of IEEE-defined operations as the CPU reference and
    // the naive kernel, hence bit-identical results.
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            acc = fmaf(c_filter[i * k + j],
                       static_cast<float>(
                           tile[(static_cast<int>(threadIdx.y) + i) * tile_w +
                                (static_cast<int>(threadIdx.x) + j)]),
                       acc);
        }
    }
    acc = acc < 0.0f ? 0.0f : (acc > 255.0f ? 255.0f : acc);  // clamp: uint8 range
    dst[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(acc + 0.5f);
}

// convolve2d_naive_kernel — identical arithmetic, zero staging: all k^2 reads
// per thread go straight to global memory (through L1/L2). Kept, not as a
// strawman, but as the measurable baseline that shows what shared-memory
// tiling actually buys — run `bench_gpu` and compare.
static __global__ void convolve2d_naive_kernel(const std::uint8_t* __restrict__ src,
                                               std::uint8_t* __restrict__ dst, int w, int h,
                                               int k) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int r = k / 2;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        int sy = y + i - r;
        sy = sy < 0 ? 0 : (sy >= h ? h - 1 : sy);  // clamp to edge
        for (int j = 0; j < k; ++j) {
            int sx = x + j - r;
            sx = sx < 0 ? 0 : (sx >= w ? w - 1 : sx);
            acc = fmaf(c_filter[i * k + j],
                       static_cast<float>(src[static_cast<std::size_t>(sy) * w + sx]), acc);
        }
    }
    acc = acc < 0.0f ? 0.0f : (acc > 255.0f ? 255.0f : acc);
    dst[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(acc + 0.5f);
}

// ---------------------------------------------------------- host wrappers

// Copies a filter's weights into this TU's constant-memory slot and returns
// the kernel size. Synchronous, ~100 bytes — negligible next to any launch,
// but split out from the launch wrappers so benchmarks can upload once and
// time kernels alone.
static inline int upload_filter(Filter f) {
    const int k = filter_size(f);
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, filter_weights(f),
                                  sizeof(float) * static_cast<std::size_t>(k) * k));
    return k;
}

// Raw launchers: assume the filter is already resident in c_filter.
static inline void convolve2d_tiled(const GpuBuffer<std::uint8_t>& d_src,
                                    GpuBuffer<std::uint8_t>& d_dst, int width, int height,
                                    int k, cudaStream_t stream = 0) {
    constexpr int TILE = 16;
    const dim3 block(TILE, TILE);
    const dim3 grid((static_cast<unsigned>(width) + TILE - 1) / TILE,
                    (static_cast<unsigned>(height) + TILE - 1) / TILE);
    convolve2d_tiled_kernel<TILE><<<grid, block, 0, stream>>>(d_src.get(), d_dst.get(), width,
                                                              height, k);
    CUDA_CHECK(cudaGetLastError());
}

static inline void convolve2d_naive(const GpuBuffer<std::uint8_t>& d_src,
                                    GpuBuffer<std::uint8_t>& d_dst, int width, int height,
                                    int k, cudaStream_t stream = 0) {
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(width) + 15u) / 16u,
                    (static_cast<unsigned>(height) + 15u) / 16u);
    convolve2d_naive_kernel<<<grid, block, 0, stream>>>(d_src.get(), d_dst.get(), width,
                                                        height, k);
    CUDA_CHECK(cudaGetLastError());
}

// upload + launch in one call; single-channel buffers.
static inline void convolve2d(const GpuBuffer<std::uint8_t>& d_src,
                              GpuBuffer<std::uint8_t>& d_dst, int width, int height, Filter f,
                              bool tiled = true, cudaStream_t stream = 0) {
    const int k = upload_filter(f);
    if (tiled) {
        convolve2d_tiled(d_src, d_dst, width, height, k, stream);
    } else {
        convolve2d_naive(d_src, d_dst, width, height, k, stream);
    }
}

// Image-level convenience. The kernels are single-channel; multi-channel
// images are split into planes on the host, convolved plane by plane, and
// re-interleaved (matches cpu::convolve2d(Image, Filter)).
static inline Image convolve2d(const Image& img, Filter f, bool tiled = true) {
    const std::size_t n = img.n_pixels();
    Image out(img.width, img.height, img.channels);
    upload_filter(f);
    const int k = filter_size(f);

    GpuBuffer<std::uint8_t> d_in(n);
    GpuBuffer<std::uint8_t> d_out(n);

    if (img.channels == 1) {
        d_in.copy_from_host(img.data.data(), n);
        if (tiled) {
            convolve2d_tiled(d_in, d_out, img.width, img.height, k);
        } else {
            convolve2d_naive(d_in, d_out, img.width, img.height, k);
        }
        d_out.copy_to_host(out.data.data(), n);
        return out;
    }

    std::vector<std::uint8_t> plane(n), filtered(n);
    for (int c = 0; c < img.channels; ++c) {
        for (std::size_t i = 0; i < n; ++i) plane[i] = img.data[i * img.channels + c];
        d_in.copy_from_host(plane.data(), n);
        if (tiled) {
            convolve2d_tiled(d_in, d_out, img.width, img.height, k);
        } else {
            convolve2d_naive(d_in, d_out, img.width, img.height, k);
        }
        d_out.copy_to_host(filtered.data(), n);
        for (std::size_t i = 0; i < n; ++i) out.data[i * img.channels + c] = filtered[i];
    }
    return out;
}

}  // namespace cig
