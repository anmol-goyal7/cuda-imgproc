# Benchmarking Methodology

This document describes the benchmarking strategy for `cuda-imgproc` — what we measure, how we measure it, and what results we expect.

---

## Goal

Quantify the speedup of GPU-accelerated image processing over CPU implementations, across different operations, image resolutions, and batch sizes. The benchmarks serve two purposes:

1. **Demonstrate** that GPU parallelism provides significant real-world speedup for image processing
2. **Understand** where the speedup comes from — compute-bound vs memory-bound, kernel overhead vs data transfer cost

---

## Implementations Compared

### CPU Single-Threaded

Plain C++ with nested `for` loops. No SIMD intrinsics, no multithreading, no compiler-specific optimizations beyond `-O2`. This is the baseline — the simplest correct implementation.

```cpp
// Example: grayscale, CPU single-threaded
for (int i = 0; i < num_pixels; i++) {
    out[i] = 0.299f * in[3*i] + 0.587f * in[3*i+1] + 0.114f * in[3*i+2];
}
```

### CPU Multi-Threaded (OpenMP)

Same algorithm as the single-threaded version, but with OpenMP parallelization across available CPU cores. Minimal code change — just a `#pragma omp parallel for` directive.

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < num_pixels; i++) {
    out[i] = 0.299f * in[3*i] + 0.587f * in[3*i+1] + 0.114f * in[3*i+2];
}
```

This shows what you get from "easy" CPU parallelism — typically 4–16x speedup depending on core count.

### CUDA GPU

Full GPU implementation with appropriate optimizations for each operation:
- Global memory access with coalescing for simple operations
- Shared memory tiling for convolution
- Atomic operations and parallel scan for histogram equalization

---

## What We Measure

### Single-Image Latency

Time to process one image at a given resolution. Measured across:

**Resolutions:** 256x256, 512x512, 1024x1024, 2048x2048, 4096x4096

**Operations:** RGB→Grayscale, RGB→HSV, Horizontal Flip, Rotation (45°), Conv2D 3x3, Conv2D 5x5, Gaussian Blur 5x5, Sobel Edge Detection, Histogram Equalization

### Batch Throughput

Images processed per second when processing a batch. This amortizes fixed costs (kernel launch overhead, memory allocation) and reveals sustained throughput.

**Batch sizes:** 100, 1000, 5000, 10000

**Resolution:** 512x512 (fixed, to isolate batch scaling behavior)

### GPU-Specific Metrics

- **Kernel-only time:** Time spent executing the GPU kernel, excluding memory transfers
- **End-to-end time:** Total time including host→device and device→host `cudaMemcpy`
- **Transfer overhead:** `(end_to_end - kernel_only) / end_to_end` — shows how much time is "wasted" on data movement

---

## Methodology

### Warm-Up

Before timed runs, execute 5 warm-up iterations. This ensures:
- GPU clocks are boosted to their maximum frequency (NVIDIA GPUs clock down when idle)
- CUDA context is fully initialized (first CUDA call has overhead)
- CPU caches are warm

### Timing

**CPU:** `std::chrono::high_resolution_clock`

```cpp
auto start = std::chrono::high_resolution_clock::now();
// ... operation ...
auto end = std::chrono::high_resolution_clock::now();
double ms = std::chrono::duration<double, std::milli>(end - start).count();
```

**GPU:** CUDA events (preferred over wall-clock timing because they measure GPU time directly, without CPU synchronization jitter)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### Statistical Reporting

For each configuration, run 100 timed iterations and report:
- **Median** (primary metric — robust to outliers)
- **Mean** and **standard deviation** (for context)
- **Min / Max** (to identify variance)

### Speedup Calculation

```
Speedup = CPU_single_thread_median / CUDA_median
```

A speedup of 50x means the GPU version is 50 times faster than the single-threaded CPU version.

---

## Expected Results

Based on theoretical analysis of arithmetic intensity and memory bandwidth:

### Color Conversion (Memory-Bound)

- Very low arithmetic intensity (3 multiplies + 2 adds per pixel)
- Performance limited by memory bandwidth, not compute
- GPU advantage: ~200-400 GB/s memory bandwidth vs ~40-50 GB/s on CPU
- **Expected speedup: 5–15x** (limited by transfer overhead for small images, bandwidth-bound for large)

### 2D Convolution (Compute-Bound for Large Kernels)

- Arithmetic intensity scales with kernel size: K² multiply-adds per output pixel
- 3x3: 9 ops/pixel — still somewhat memory-bound
- 5x5: 25 ops/pixel — transitioning to compute-bound
- 7x7: 49 ops/pixel — firmly compute-bound
- **Expected speedup: 20–100x** depending on kernel size and shared memory utilization

### Histogram Equalization (Mixed)

- Histogram pass: limited by atomic contention
- Prefix scan: limited by parallelism for small arrays (256 bins)
- Remap pass: memory-bound (simple lookup table)
- **Expected speedup: 10–30x** — atomics add overhead, but the reduction + remap passes parallelize well

### Batch Processing

- Amortizes kernel launch overhead and memory allocation
- Can overlap transfers with computation using streams
- **Expected speedup: 50–200x** at large batch sizes with streams

---

## Reproducing the Benchmarks

```bash
# Build the benchmark suite
make benchmark

# Run all benchmarks (outputs CSV to results/)
./build/benchmarks/run_all

# Run a specific benchmark
./build/benchmarks/bench_conv2d --resolution 1024 --kernel-size 5 --runs 100

# Generate plots from CSV results
python3 scripts/plot_benchmarks.py results/
```

Output plots will be saved to `results/plots/` as PNG files.
