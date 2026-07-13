# cuda-imgproc

A header-only C++17/CUDA image-processing library built from first principles: four stages of classic operations, each implemented as readable, heavily commented kernels alongside bit-matching CPU references and an honest benchmark harness.

## Why this exists

Production GPU libraries are built to be *used*, not *read*. Call `cv::cuda::cvtColor` and a result comes back — but the kernel, the memory traffic it generates, and the reasons it is shaped the way it is are buried under dispatch layers, template machinery, and a decade of micro-optimizations. That is the right trade-off for production and the wrong one for learning: the concepts that actually determine GPU performance (coalescing, shared-memory tiling, atomics contention, warp divergence) are invisible exactly where they matter.

This library inverts the trade-off. Every kernel is small enough to read in one sitting, and its header comment states the grid geometry, the memory tiers it touches, and the global-traffic arithmetic that justifies its design. Each operation also ships a single-thread CPU reference implementing *identical* arithmetic — so correctness is checked by exact comparison, not by eyeball — plus an OpenMP variant, so the benchmark answers the fair question ("GPU vs all the CPU cores you already have"), not the flattering one.

## Quickstart

Requirements: CUDA Toolkit (nvcc, tested on 12.x, sm_75+), g++ with C++17 and OpenMP, GNU Make. No other dependencies — stb_image is vendored in `third_party/`.

```cpp
// demo.cu
#include "cuda_imgproc.hpp"

int main() {
    cig::Image img  = cig::load_png("input.png", 3);              // RGB, uint8
    cig::Image gray = cig::rgb_to_gray(img);                      // GPU: upload → kernel → download
    cig::Image soft = cig::convolve2d(gray, cig::Filter::gaussian5x5);
    cig::save_png("blurred.png", soft);
}
```

```sh
nvcc -O3 -std=c++17 -Iinclude -Ithird_party demo.cu -o demo && ./demo
```

The `Image`-level calls above are the convenience API (each one round-trips over PCIe). The real API operates on `cig::GpuBuffer<uint8_t>` device buffers, so a pipeline uploads once, runs any number of kernels, and downloads once — `examples/pipeline_demo.cu`, `tests/test_gpu.cu` and `bench/bench_gpu.cu` show both styles.

## The four-stage concept ladder

Each stage exists to teach the next CUDA concept; difficulty is cumulative.

| stage | ops | CUDA concepts | difficulty |
|---|---|---|---|
| 1 — color conversion | `rgb_to_gray`, `rgb_to_hsv` | thread hierarchy, global indexing, boundary checks, memory coalescing, warp divergence | ★ |
| 2 — geometric augmentation | `flip_horizontal`, `flip_vertical`, `rotate`, `resize` | 2D grids/blocks, inverse mapping (gather vs scatter), bilinear interpolation device functions | ★★ |
| 3 — 2D convolution | `convolve2d` (tiled + naive), 5-filter bank | shared-memory tiling, halo loads, `__syncthreads()`, `__constant__` memory broadcast, data-reuse arithmetic | ★★★ |
| 4 — histogram equalization | `equalize_hist` (3 kernels) | shared vs global atomics, grid-stride loops, Blelloch work-efficient scan, multi-kernel algorithms as global barriers | ★★★★ |

## Memory-hierarchy map

Which memory tier each stage leans on, and why:

| tier | used by | why |
|---|---|---|
| global (DRAM) | all stages | the images live here; stages 1–2 are pure streaming, so their entire design goal is coalesced access |
| shared (per-SM SRAM) | stage 3 (input tile + halo), stage 4 (per-block histogram, scan workspace) | data touched many times by one block: k² reuse per pixel in convolution; millions of atomic increments folded into 256 on-chip counters in the histogram |
| constant (broadcast cache) | stage 3 (filter weights) | every thread reads the same weight at the same moment — a warp-uniform address served to all 32 lanes in one fetch |
| registers | everywhere | accumulators and interpolation temporaries; spilling is avoided by keeping kernels small |

## Correctness testing

Two suites, one contract: **the GPU must reproduce the CPU reference, not resemble it.**

* `make test-cpu` (any machine) — handcrafted fixtures whose expected outputs are computed by hand in the comments (BT.601 luma values, a sharpen convolution done on paper, an equalization LUT derived bin by bin), plus structural properties (flips are involutions, rotate(π) equals flipping both axes, equalization flattens the CDF) and OpenMP-vs-single-thread byte equality.
* `make test-gpu ARCH=sm_75` (GPU machine) — for every op, a seeded `std::mt19937` image (bit-identical on all platforms) goes through the CPU reference and the GPU kernel; outputs are compared with per-op tolerances at 256×256 and 1023×1021 (deliberately not multiples of the block size, so every boundary check earns its keep):

| tolerance | ops | reason |
|---|---|---|
| exact (0) | `rgb_to_gray`, flips, `equalize_hist`, tiled-vs-naive convolution | integer moves; fma chains and double-precision LUT math that IEEE 754 pins down on both compilers |
| 0 expected, ±1 tolerated | convolution vs CPU | identical explicit-fma accumulation order should be bit-exact even under `--use_fast_math`; a ±1 gray level is tolerated and the achieved max-diff is printed, so the run documents which held |
| ±1 | `rotate`, `resize`, `rgb_to_hsv` | float interpolation/reciprocal paths where fast-math may round the last bit differently; hue compared on its mod-180 ring |

Where bit-exactness is claimed it is *engineered*, not hoped for: both sides evaluate luma and convolution sums via explicit `fmaf` chains in the same order (IEEE 754 defines fused multiply-add exactly), and the equalization LUT divides in `double`, which `--use_fast_math` does not touch.

## Benchmark methodology

* Synthetic seeded pseudo-random images at 256², 512², 1024², 2048², 4096²; generated before timing — no disk I/O anywhere near a timed region.
* Every measurement: **10 warm-up iterations, then 100 timed runs; the CSV records the median, stdout adds the standard deviation.** (Exception, documented in the harness: the CPU sides of the 100-image batch rows use 5 timed runs, each already spanning 100 images.)
* **GPU kernel-only** time via `cudaEvent` pairs around the launch — device-clock timestamps, excluding transfers and launch latency. **GPU wall** time via `std::chrono` around upload + kernel + download, recorded separately, so PCIe cost is visible instead of hidden.
* CPU single-thread and OpenMP (all logical cores) via `std::chrono`.
* Ops: `rgb_to_gray`, `gaussian5x5` (tiled *and* naive), bilinear resize 4096²→1024², `equalize_hist`, plus batch mode — 100 images at 1024² — reported as per-image time and images/s for `rgb_to_gray` and `gaussian5x5`.
* The harness prints the device name, SM count and memory clock, and stamps them into the CSV header.

Reproduce: `colab/README.md` (one pasted cell on a free Colab T4), then `make readme-results` locally. `make bench-cpu` runs the CPU side alone on any machine.

## Results

<!-- BENCH:BEGIN -->
<!-- generated by scripts/render_results.py from results/results_Tesla_T4.csv — do not edit by hand -->

#### Tesla T4

*sms: 40; mem_clock_mhz: 5001; omp_threads: 2*

| op | resolution | cpu 1-thread (ms) | openmp (ms) | gpu kernel (ms) | gpu wall (ms) | speedup vs 1t | speedup vs omp |
|---|---|---|---|---|---|---|---|
| `rgb_to_gray` | 256x256 | 0.393 | 0.270 | 0.0083 | 0.0996 | 47.4× | 32.6× |
| `rgb_to_gray` | 512x512 | 1.57 | 3.75 | 0.0123 | 0.322 | 128.0× | 305.3× |
| `rgb_to_gray` | 1024x1024 | 6.57 | 6.32 | 0.0328 | 1.10 | 200.5× | 192.9× |
| `rgb_to_gray` | 2048x2048 | 16.6 | 17.7 | 0.143 | 3.66 | 115.9× | 123.7× |
| `rgb_to_gray` | 4096x4096 | 67.2 | 70.6 | 0.551 | 14.3 | 122.1× | 128.3× |
| `gaussian5x5_tiled` | 256x256 | 3.85 | 5.79 | 0.0143 | 0.0683 | 268.7× | 404.1× |
| `gaussian5x5_naive` | 256x256 | 3.85 | 5.79 | 0.0164 | 0.0677 | 235.1× | 353.6× |
| `gaussian5x5_tiled` | 512x512 | 22.8 | 15.7 | 0.0520 | 0.209 | 439.3× | 301.4× |
| `gaussian5x5_naive` | 512x512 | 22.8 | 15.7 | 0.0656 | 0.225 | 348.2× | 238.9× |
| `gaussian5x5_tiled` | 1024x1024 | 62.7 | 70.9 | 0.184 | 0.786 | 341.9× | 386.1× |
| `gaussian5x5_naive` | 1024x1024 | 62.7 | 70.9 | 0.238 | 0.838 | 264.1× | 298.3× |
| `gaussian5x5_tiled` | 2048x2048 | 255 | 264 | 0.721 | 2.35 | 353.8× | 366.1× |
| `gaussian5x5_naive` | 2048x2048 | 255 | 264 | 0.559 | 2.27 | 456.2× | 472.1× |
| `gaussian5x5_tiled` | 4096x4096 | 1020 | 1057 | 2.86 | 8.04 | 356.9× | 370.0× |
| `gaussian5x5_naive` | 4096x4096 | 1020 | 1057 | 2.08 | 8.37 | 489.6× | 507.7× |
| `equalize_hist` | 256x256 | 0.0777 | 0.0807 | 0.0164 | 0.0646 | 4.7× | 4.9× |
| `equalize_hist` | 512x512 | 0.318 | 0.312 | 0.0284 | 0.192 | 11.2× | 11.0× |
| `equalize_hist` | 1024x1024 | 1.28 | 1.21 | 0.0762 | 0.668 | 16.8× | 15.9× |
| `equalize_hist` | 2048x2048 | 5.06 | 4.88 | 0.129 | 2.07 | 39.3× | 37.8× |
| `equalize_hist` | 4096x4096 | 20.4 | 19.8 | 0.835 | 7.74 | 24.4× | 23.7× |
| `resize_bilinear` | 4096x4096->1024x1024 | 24.8 | 22.8 | 0.172 | 11.2 | 144.0× | 132.5× |
| `rgb_to_gray_batch100` (931 img/s) | 1024x1024 | 6.02 | 4.49 | 0.0300 | 1.07 | 200.8× | 149.6× |
| `gaussian5x5_batch100` (1441 img/s) | 1024x1024 | 65.6 | 73.4 | 0.0840 | 0.694 | 780.3× | 873.9× |

*Medians of 100 timed runs after 10 warm-ups. `gpu kernel` = cudaEvent time around the kernel(s) only; `gpu wall` = std::chrono around upload + kernel + download. Speedups compare CPU medians against `gpu kernel`.*

*`_batch100` rows are per-image figures over a pipelined 100-image batch (throughput in parentheses); their CPU columns use 5 timed batch runs.*
<!-- BENCH:END -->

## Tiled vs naive convolution, by the numbers

For a 5×5 kernel (radius r=2) with 16×16 tiles, per block:

* naive: 256 threads × 25 reads = **6400 global-memory reads**
* tiled: one cooperative load of the (16+2r)×(16+2r) = 20×20 input tile = **400 global reads** (the 16×16 core plus the r-wide halo ring), then all 6400 window reads hit shared memory

That is a **16× reduction in global traffic** (6400/400), bought with 484 bytes of shared memory and one `__syncthreads()`. The naive kernel is kept on purpose: modern L1/L2 caches absorb part of the redundancy, so the *measured* gap (see the benchmark's `gaussian5x5_tiled` vs `gaussian5x5_naive` rows) is real but smaller than 16× — the difference between transaction arithmetic and delivered performance is itself the lesson.

## How to run everything

```sh
# any machine (no GPU needed)
make cpu            # build CPU tests/bench/tools
make test-cpu       # run the local test suite
make bench-cpu      # CPU-only benchmark → results/cpu_baseline_<host>.csv
make tools && ./bin/make_test_image demo.png   # synthetic test PNG

# GPU machine (or free Colab T4 — see colab/README.md)
make gpu ARCH=sm_75        # build test_gpu + bench_gpu (default: sm_75+sm_86 fatbin)
make test-gpu ARCH=sm_75   # GPU-vs-CPU correctness table
make bench-gpu ARCH=sm_75  # canonical CSV → results/results_<gpu>.csv
make examples && ./bin/pipeline_demo           # PNG → gray → blur → equalize

# after copying the Colab CSV into results/
make readme-results  # renders CSV into README.md + results/summary.md
```

## Repository structure

```
include/
  cuda_imgproc.hpp             umbrella header (CPU-only TUs get host code; nvcc adds kernels)
  cuda_imgproc/
    core/cuda_check.cuh        CUDA_CHECK error-handling macro
    core/gpu_buffer.cuh        GpuBuffer<T> move-only RAII device allocation
    core/image.hpp             host Image container, PNG I/O, synthetic generators
    core/filters.hpp           the filter bank weights (shared by CPU and GPU paths)
    kernels/color.cuh          stage 1: rgb_to_gray, rgb_to_hsv
    kernels/augment.cuh        stage 2: flips, rotate, resize (+ bilinear device fn)
    kernels/convolution.cuh    stage 3: tiled + naive convolution, __constant__ bank
    kernels/histogram.cuh      stage 4: histogram, Blelloch scan, remap
    cpu/ops.hpp                single-thread references (identical arithmetic)
    cpu/ops_omp.hpp            OpenMP variants (identical results, all cores)
third_party/                   stb_image.h, stb_image_write.h (vendored, commit-pinned)
tests/                         test_cpu.cpp (local), test_gpu.cu (GPU vs CPU)
bench/                         bench_cpu.cpp, bench_gpu.cu (canonical CSV writer)
examples/pipeline_demo.cu      PNG → gray → gaussian → equalize demo
tools/make_test_image.cpp      synthetic gradient+noise PNG generator
scripts/render_results.py      CSV → README/results tables (idempotent)
colab/                         one-cell T4 runner + instructions
results/                       committed benchmark CSVs + generated summary.md
```

## Limitations & future work

* **Transfers dominate single-image latency.** Kernels are microseconds; PCIe copies are milliseconds. CUDA streams overlapping H2D/compute/D2H across a batch would hide most of that and is the natural next chapter.
* **uint8 only, 1 or 3 channels, no strides/ROI.** Deliberate: format generality multiplies code without adding concepts.
* **FP16 / Tensor Cores unexplored.** Convolution as WMMA matrix ops (or FP16x2 arithmetic) is the modern performance path and a planned extension.
* **No Python bindings.** nanobind + a zero-copy PyTorch tensor interface would let the kernels drop into real training pipelines.
* **Single-GPU, synchronous host API.** Multi-GPU splitting and a stream-carrying async API are out of scope for the teaching core.

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 Anmol Goyal.
