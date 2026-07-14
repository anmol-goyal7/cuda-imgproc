#pragma once
// Stage 4 — histogram equalization, in three kernel launches:
//
//   1. histogram_kernel  — 256-bin histogram (shared-memory atomics,
//                          grid-stride loop, per-block merge)
//   2. scan_kernel       — Blelloch prefix scan over the histogram -> CDF,
//                          then the equalization LUT
//   3. remap_kernel      — per-pixel LUT lookup
//
// Concepts introduced here:
//   * atomics, and the cost hierarchy: shared-memory atomics are roughly an
//     order of magnitude (~10-20x) cheaper than global-memory atomics, which
//     serialize colliding updates through L2
//   * grid-stride loops: decoupling grid size from data size
//   * work-efficient parallel scan (up-sweep / down-sweep)
//   * multi-kernel algorithms: a kernel launch is the only global barrier —
//     the histogram must be *complete* before the scan may start, and the
//     LUT before the remap, so the phases are separate launches on one
//     stream rather than one kernel with an (impossible) device-wide sync.
//
// Equalization formula (the standard cdf_min formulation):
//   lut[v] = round(255 * (cdf[v] - cdf_min) / (N - cdf_min))
// with cdf the inclusive histogram CDF and cdf_min its first nonzero entry.
// The LUT arithmetic runs in double on both CPU and GPU: double division is
// exempt from --use_fast_math, so both sides round identically and GPU output
// matches the CPU reference bit for bit.

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>

#include "../core/cuda_check.cuh"
#include "../core/gpu_buffer.cuh"
#include "../core/image.hpp"

namespace cig {

// histogram_kernel — pass 1: count pixel values into g_hist[256].
//
// Geometry:  1D grid, blockDim.x = 256 (required: thread t owns bin t during
//            zeroing and merging), gridDim capped at 4096.
// Memory:    __shared__ 256-bin block-local histogram; global atomics only
//            for the final merge.
//
// The naive approach — every pixel does atomicAdd on the global histogram —
// funnels millions of updates onto 256 memory locations; colliding atomics
// serialize, so the whole GPU queues behind L2. Instead each block counts
// into its own shared-memory histogram (atomics there are resolved in the
// on-chip SRAM, ~10-20x cheaper, and only contend within one block), then
// merges: 256 global atomicAdds per block instead of one per pixel — for a
// full block's worth of pixels, a >250x cut in global atomic traffic.
//
// The grid-stride loop lets a fixed-size grid walk any image: thread i
// handles pixels i, i+stride, i+2*stride, ... where stride = total threads.
// Consecutive threads still touch consecutive addresses in every pass, so
// reads stay coalesced. Capping the grid at 4096 blocks bounds the merge
// traffic (4096*256 global atomics max) while still oversubscribing every SM.
static __global__ void histogram_kernel(const std::uint8_t* __restrict__ data, std::size_t n,
                                        unsigned int* __restrict__ g_hist) {
    __shared__ unsigned int s_hist[256];
    s_hist[threadIdx.x] = 0;  // cooperative zeroing: thread t owns bin t
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&s_hist[data[i]], 1u);
    }
    __syncthreads();  // all counting done before anyone merges

    atomicAdd(&g_hist[threadIdx.x], s_hist[threadIdx.x]);
}

// scan_kernel — pass 2: exclusive Blelloch scan of the 256-bin histogram in
// shared memory, then the equalization LUT.
//
// Geometry:  a single block of 256 threads (one per bin). 256 values are far
//            too small to justify a multi-block scan with inter-block
//            carries; one block means __syncthreads() is a full barrier for
//            the whole problem.
// Memory:    __shared__ 256-entry workspace, 3 global reads + 1 write per
//            thread. This kernel is negligible next to passes 1 and 3; it
//            exists to keep the CDF computation on-device (a round trip to
//            the host would cost more than the whole algorithm).
//
// Blelloch's scan is *work-efficient*: an up-sweep builds a binary reduction
// tree in place (O(n) adds), a down-sweep converts it into an exclusive
// prefix sum (O(n) more), versus O(n log n) for the naive doubling scan.
// At n = 256 either would do; work-efficiency is the concept being taught.
// This implementation computes the exclusive scan, then each thread adds its
// own bin count to get the inclusive CDF the formula needs.
static __global__ void scan_kernel(const unsigned int* __restrict__ g_hist,
                                   std::uint8_t* __restrict__ g_lut, unsigned int n_pixels) {
    __shared__ unsigned int scan[256];
    __shared__ unsigned int cdf_min;

    const int tid = static_cast<int>(threadIdx.x);
    const unsigned int count = g_hist[tid];
    scan[tid] = count;
    if (tid == 0) cdf_min = 0;
    __syncthreads();

    // Up-sweep: after the pass with stride s, scan[i] for i = 2s-1, 4s-1, ...
    // holds the sum of the 2s elements ending at i. Only threads whose target
    // index is in range do work, but the barrier is hit by all 256 threads —
    // __syncthreads() inside divergent control flow would deadlock.
    for (int stride = 1; stride < 256; stride <<= 1) {
        const int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < 256) scan[idx] += scan[idx - stride];
        __syncthreads();
    }

    // Down-sweep: zero the root, then walk the tree back down swapping and
    // adding, which redistributes the partial sums into an exclusive scan.
    if (tid == 0) scan[255] = 0;
    __syncthreads();
    for (int stride = 128; stride >= 1; stride >>= 1) {
        const int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < 256) {
            const unsigned int left = scan[idx - stride];
            scan[idx - stride] = scan[idx];
            scan[idx] += left;
        }
        __syncthreads();
    }

    // scan[v] is now the exclusive CDF; inclusive is exclusive + own count.
    const unsigned int inclusive = scan[tid] + count;

    // cdf_min = inclusive CDF of the first occupied bin. That bin is the
    // unique one with a nonzero count and an exclusive CDF of zero, so
    // exactly one thread writes — no atomics needed.
    if (count != 0 && scan[tid] == 0) cdf_min = inclusive;
    __syncthreads();

    const unsigned int denom = n_pixels - cdf_min;
    std::uint8_t out;
    if (denom == 0) {
        // Degenerate image (every pixel identical): 0/0 in the formula; map
        // identically. Matches cpu::equalize_hist.
        out = static_cast<std::uint8_t>(tid);
    } else {
        double num = static_cast<double>(inclusive) - static_cast<double>(cdf_min);
        if (num < 0.0) num = 0.0;  // bins below the first occupied one (unused)
        out = static_cast<std::uint8_t>(255.0 * num / static_cast<double>(denom) + 0.5);
    }
    g_lut[tid] = out;
}

// remap_kernel — pass 3: dst[i] = lut[src[i]].
//
// Geometry:  1D grid, blockDim.x = 256, one thread per pixel.
// Memory:    global; the 256-byte LUT is read through L1/L2 and hits cache
//            for every access after the first few.
//
// Embarrassingly parallel and bandwidth-bound exactly like stage 1's
// grayscale kernel: 1 byte in, 1 byte out, one dependent load between them.
// Reads and writes are perfectly coalesced; the LUT gather is random-access
// but into 256 bytes of hot cache, effectively free.
static __global__ void remap_kernel(const std::uint8_t* __restrict__ src,
                                    std::uint8_t* __restrict__ dst,
                                    const std::uint8_t* __restrict__ lut, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = lut[src[i]];
}

// ---------------------------------------------------------- host wrappers

// Full equalization on device buffers. d_hist (256 uints) and d_lut (256
// bytes) are caller-provided workspace so benchmark loops don't allocate
// per iteration.
inline void equalize_hist(const GpuBuffer<std::uint8_t>& d_src, GpuBuffer<std::uint8_t>& d_dst,
                          std::size_t n_pixels, GpuBuffer<unsigned int>& d_hist,
                          GpuBuffer<std::uint8_t>& d_lut, cudaStream_t stream = 0) {
    // The CDF/LUT math (and scan_kernel's parameter) is 32-bit; beyond
    // 2^32 - 1 pixels the counts would silently wrap.
    detail::require(n_pixels > 0 && n_pixels <= UINT_MAX,
                    "equalize_hist: pixel count must fit in 32 bits");
    detail::require(d_src.size() >= n_pixels && d_dst.size() >= n_pixels,
                    "equalize_hist: device buffer smaller than the image");
    detail::require(d_hist.size() >= 256 && d_lut.size() >= 256,
                    "equalize_hist: workspace needs 256 histogram bins and 256 LUT entries");
    CUDA_CHECK(cudaMemsetAsync(d_hist.get(), 0, d_hist.bytes(), stream));

    const int block = 256;  // histogram_kernel/scan_kernel contract: 256 threads
    const unsigned int grid_hist =
        static_cast<unsigned int>(std::min<std::size_t>((n_pixels + block - 1) / block, 4096));
    histogram_kernel<<<grid_hist, block, 0, stream>>>(d_src.get(), n_pixels, d_hist.get());
    CUDA_CHECK(cudaGetLastError());

    scan_kernel<<<1, block, 0, stream>>>(d_hist.get(), d_lut.get(),
                                         static_cast<unsigned int>(n_pixels));
    CUDA_CHECK(cudaGetLastError());

    const unsigned int grid_remap = static_cast<unsigned int>((n_pixels + block - 1) / block);
    remap_kernel<<<grid_remap, block, 0, stream>>>(d_src.get(), d_dst.get(), d_lut.get(), n_pixels);
    CUDA_CHECK(cudaGetLastError());
}

inline void equalize_hist(const GpuBuffer<std::uint8_t>& d_src, GpuBuffer<std::uint8_t>& d_dst,
                          std::size_t n_pixels, cudaStream_t stream = 0) {
    GpuBuffer<unsigned int> d_hist(256);
    GpuBuffer<std::uint8_t> d_lut(256);
    equalize_hist(d_src, d_dst, n_pixels, d_hist, d_lut, stream);
    // The launches above are asynchronous but the workspace dies at scope
    // exit — synchronize so the device is finished with d_hist/d_lut before
    // their destructors free them.
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Image-level convenience. Single-channel only, enforced: one global
// histogram over interleaved RGB would push all three channels through the
// same LUT (shifting hues), and equalizing channels independently is rarely
// what anyone wants either — convert to gray, or equalize HSV's V plane.
inline Image equalize_hist(const Image& img) {
    detail::require(img.channels == 1, "equalize_hist: image must be single-channel");
    const std::size_t n = img.n_pixels();
    GpuBuffer<std::uint8_t> d_in(n), d_out(n);
    d_in.copy_from_host(img.data.data(), n);
    equalize_hist(d_in, d_out, n);
    Image out(img.width, img.height, 1);
    d_out.copy_to_host(out.data.data(), n);
    return out;
}

}  // namespace cig
