# Technical Architecture

This document describes the design decisions, memory management strategy, and internal organization of `cuda-imgproc`.

---

## Design Philosophy: Header-Only Library

`cuda-imgproc` is designed as a **header-only** library. Every kernel, utility function, and data structure lives in `.cuh` header files. To use any operation, you include a single header — no separate `.cu` compilation units, no static or shared libraries to link against.

**Why header-only?**

- **Zero build friction.** Users include a header and compile with `nvcc`. No CMake `find_package`, no `-lcuda_imgproc`, no version mismatches.
- **Inlining opportunity.** Since kernel code is visible at the compilation unit level, `nvcc` can inline device functions and optimize across call boundaries.
- **Transparency.** Every line of kernel code is visible in the header — nothing is hidden behind a compiled binary.

**Trade-off:** Compilation time increases because the full kernel source is parsed for every translation unit that includes it. For a project of this scale (a handful of kernels), this is negligible.

---

## Kernel Organization

Each image processing operation lives in its own header file under `include/cuda_imgproc/`:

```
include/cuda_imgproc/
├── common.cuh        Shared utilities, macros, types
├── color.cuh         Color space conversion kernels
├── augment.cuh       Geometric augmentation kernels
├── conv2d.cuh        2D convolution kernels
└── histogram.cuh     Histogram computation and equalization
```

### `common.cuh` — Shared Utilities

Contains:
- `CUDA_CHECK()` macro for error checking every CUDA API call
- `DeviceBuffer<T>` RAII wrapper for GPU memory
- `Image` struct representing an image (width, height, channels, pixel data)
- `load_image()` / `save_image()` wrappers around stb_image
- Helper functions for computing grid/block dimensions

### Per-Operation Headers

Each operation header follows the same structure:

1. **Kernel function(s)** — `__global__` device code
2. **Host wrapper function** — Allocates device memory, copies data to GPU, launches kernel, copies result back, frees device memory
3. **Convenience overloads** — Variants for different input types or parameter sets

Example pattern:

```cpp
// Device kernel (internal)
__global__ void rgb_to_gray_kernel(const uint8_t* in, uint8_t* out, int N) {
    // ... kernel code ...
}

// Host-side API (public)
Image to_grayscale(const Image& input) {
    DeviceBuffer<uint8_t> d_in(input.size());
    DeviceBuffer<uint8_t> d_out(input.width * input.height);

    CUDA_CHECK(cudaMemcpy(d_in.data, input.data, input.size(), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (input.width * input.height + threads - 1) / threads;
    rgb_to_gray_kernel<<<blocks, threads>>>(d_in.data, d_out.data, input.width * input.height);
    CUDA_CHECK(cudaGetLastError());

    Image output(input.width, input.height, 1);
    CUDA_CHECK(cudaMemcpy(output.data, d_out.data, output.size(), cudaMemcpyDeviceToHost));

    return output;
}
```

---

## Memory Management Strategy

### Host ↔ Device Transfer Pattern

Every operation follows the same data flow:

```
                    ┌──────────────────────────────────┐
                    │         HOST (CPU + RAM)          │
                    │                                   │
                    │   ┌─────────┐     ┌─────────┐    │
                    │   │  Input  │     │ Output  │    │
                    │   │  Image  │     │ Image   │    │
                    │   └────┬────┘     └────▲────┘    │
                    │        │               │         │
                    └────────┼───────────────┼─────────┘
                             │               │
                    cudaMemcpy H2D    cudaMemcpy D2H
                             │               │
                    ┌────────▼───────────────┼─────────┐
                    │       DEVICE (GPU + VRAM)         │
                    │                                   │
                    │   ┌─────────┐     ┌─────────┐    │
                    │   │ d_input │────►│d_output │    │
                    │   │ buffer  │     │ buffer  │    │
                    │   └─────────┘     └─────────┘    │
                    │        Kernel execution           │
                    └──────────────────────────────────┘
```

### Steps for every operation:

1. **Allocate** device buffers for input and output (`cudaMalloc` via `DeviceBuffer`)
2. **Copy** input data from host to device (`cudaMemcpy` with `cudaMemcpyHostToDevice`)
3. **Launch** the kernel with computed grid/block dimensions
4. **Synchronize** and check for errors (`cudaGetLastError`)
5. **Copy** output data from device to host (`cudaMemcpy` with `cudaMemcpyDeviceToHost`)
6. **Free** device buffers (automatic via `DeviceBuffer` destructor)

### RAII for GPU Memory

The `DeviceBuffer<T>` class ensures GPU memory is always freed, even when exceptions are thrown or early returns occur. It follows C++ RAII principles:

- Constructor calls `cudaMalloc`
- Destructor calls `cudaFree`
- Move-only (no copying — GPU pointers should not be aliased)

### Future: Pinned Memory & Streams

For batch processing, the plan is to use:
- **Pinned (page-locked) host memory** via `cudaMallocHost()` for faster transfers
- **CUDA streams** for overlapping computation and data transfer — while one image is being processed, the next is being uploaded, and the previous is being downloaded

```
Stream 1:  [Upload img1] [Kernel img1] [Download img1]
Stream 2:            [Upload img2] [Kernel img2] [Download img2]
Stream 3:                      [Upload img3] [Kernel img3] [Download img3]
           ──────────────────────────────────────────────────► time
```

---

## Error Handling

### Philosophy

Fail early, fail loud. Every CUDA API call is checked. If something goes wrong — out of memory, invalid launch configuration, kernel crash — the program prints a clear error message with the exact file and line, then exits.

### The `CUDA_CHECK` Macro

```cpp
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
```

Used as:

```cpp
CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());  // Check for launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Check for execution errors
```

---

## Benchmark Harness

### Design

The benchmark suite measures CPU single-thread, CPU OpenMP, and CUDA implementations of every operation. It is designed to produce reproducible, statistically meaningful results.

### Execution Flow

```
For each (operation, resolution, implementation):
    1. Load or generate test image
    2. Run 5 warm-up iterations (discarded)
    3. Run 100 timed iterations
    4. Record each iteration's time
    5. Compute: median, mean, std dev, min, max
    6. Write results to CSV
```

### Timing

- **CPU timing:** `std::chrono::high_resolution_clock` around the operation
- **GPU timing:** `cudaEventRecord` before and after the kernel, `cudaEventElapsedTime` for the delta — this avoids CPU-side timer overhead and synchronization artifacts
- **Two GPU measurements:**
  - **Kernel only:** Time between kernel launch and completion (excludes memory transfers)
  - **End-to-end:** Time including `cudaMemcpy` in both directions (real-world cost)

### Output

Results are written to CSV files that can be loaded into matplotlib for visualization:

```
operation,resolution,implementation,median_ms,mean_ms,std_ms,min_ms,max_ms
grayscale,1024x1024,cpu,12.34,12.56,0.82,11.02,15.23
grayscale,1024x1024,openmp,3.12,3.24,0.31,2.88,4.15
grayscale,1024x1024,cuda_kernel,0.08,0.09,0.01,0.07,0.12
grayscale,1024x1024,cuda_e2e,0.45,0.48,0.05,0.41,0.62
```
