// test_gpu — GPU-vs-CPU correctness suite. Requires a CUDA device; built and
// run on the benchmark machine (Colab T4) via `make test-gpu ARCH=sm_75`.
//
// Method: for every op, generate a seeded pseudo-random image (std::mt19937,
// identical bytes on every platform), run the single-thread CPU reference and
// the GPU kernel, and compare outputs. Tolerances:
//
//   exact (0)  rgb_to_gray, flips, histogram EQ, tiled-vs-naive convolution —
//              these paths use only arithmetic both compilers must round
//              identically (integer moves, IEEE fma chains, double division).
//   0, ±1 ok   convolution vs CPU: same fma accumulation order on both sides
//              should be bit-exact; if a compiler contracts differently under
//              fast-math a ±1 gray level is tolerated and the achieved max
//              diff is printed — the table documents which one holds.
//   ±1         rotate, resize, rgb_to_hsv: float rounding paths (fast-math
//              reciprocals, non-fused interpolation) may legitimately differ
//              in the last bit. Hue is compared circularly (H lives on a
//              mod-180 ring: 0 and 179 are neighbors).
//
// Sizes: 256x256 (whole blocks) and 1023x1021 (prime-ish: exercises every
// boundary check — ragged 1D tails, partial 2D tiles, halo clamps), plus one
// larger histogram case that overflows the 4096-block grid cap so the
// grid-stride loop actually strides.
//
// Prints a per-op table with max abs diff; exit code != 0 on any failure.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "cuda_imgproc.hpp"

struct Row {
    std::string op;
    std::string size;
    int max_diff;
    int tol;
    std::string note;
};
static std::vector<Row> g_rows;
static bool g_all_pass = true;

static void record(const std::string& op, int w, int h, int max_diff, int tol,
                   const std::string& note = "") {
    g_rows.push_back({op, std::to_string(w) + "x" + std::to_string(h), max_diff, tol, note});
    if (max_diff > tol) g_all_pass = false;
}

static int max_abs_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b,
                        bool hue_circular_stride3 = false) {
    if (a.size() != b.size()) return 255;
    int m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        // Hue channel of interleaved HSV: distance on the mod-180 ring.
        if (hue_circular_stride3 && i % 3 == 0) d = d < 180 - d ? d : 180 - d;
        if (d > m) m = d;
    }
    return m;
}

static void run_suite(int w, int h) {
    const cig::Image rgb = cig::random_image(w, h, 3, 42);

    // --- stage 1: color conversion ---------------------------------------
    const cig::Image gray_cpu = cig::cpu::rgb_to_gray(rgb);
    {
        const cig::Image gray_gpu = cig::rgb_to_gray(rgb);
        record("rgb_to_gray", w, h, max_abs_diff(gray_gpu.data, gray_cpu.data), 0,
               "bit-exact required");
    }
    {
        cig::Image hsv_cpu(w, h, 3);
        cig::cpu::rgb_to_hsv(rgb.data.data(), hsv_cpu.data.data(), w, h);
        const cig::Image hsv_gpu = cig::rgb_to_hsv(rgb);
        record("rgb_to_hsv", w, h, max_abs_diff(hsv_gpu.data, hsv_cpu.data, true), 1,
               "±1; hue compared mod 180");
    }

    // --- stage 2: augmentation -------------------------------------------
    {
        const cig::Image want = cig::cpu::flip_horizontal(rgb);
        const cig::Image got = cig::flip_horizontal(rgb);
        record("flip_horizontal", w, h, max_abs_diff(got.data, want.data), 0,
               "bit-exact required");
    }
    {
        const cig::Image want = cig::cpu::flip_vertical(rgb);
        const cig::Image got = cig::flip_vertical(rgb);
        record("flip_vertical", w, h, max_abs_diff(got.data, want.data), 0,
               "bit-exact required");
    }
    {
        const float theta = 0.5235988f;  // 30 degrees
        const cig::Image want = cig::cpu::rotate(rgb, theta);
        const cig::Image got = cig::rotate(rgb, theta);
        record("rotate(30°)", w, h, max_abs_diff(got.data, want.data), 1, "±1 float path");
    }
    {
        const int dw = w * 2 / 5 + 1, dh = h * 3 / 7 + 1;  // irregular downscale
        const cig::Image want = cig::cpu::resize(rgb, dw, dh);
        const cig::Image got = cig::resize(rgb, dw, dh);
        record("resize down", w, h, max_abs_diff(got.data, want.data), 1, "±1 float path");
    }
    {
        const int dw = w * 5 / 4 + 3, dh = h * 9 / 8 + 1;  // irregular upscale
        const cig::Image want = cig::cpu::resize(rgb, dw, dh);
        const cig::Image got = cig::resize(rgb, dw, dh);
        record("resize up", w, h, max_abs_diff(got.data, want.data), 1, "±1 float path");
    }

    // --- stage 3: convolution (on the grayscale plane) --------------------
    {
        cig::Image want(w, h, 1);
        cig::cpu::convolve2d(gray_cpu.data.data(), want.data.data(), w, h,
                             cig::Filter::gaussian5x5);
        const cig::Image tiled = cig::convolve2d(gray_cpu, cig::Filter::gaussian5x5, true);
        const cig::Image naive = cig::convolve2d(gray_cpu, cig::Filter::gaussian5x5, false);
        const int d_tiled = max_abs_diff(tiled.data, want.data);
        const int d_naive = max_abs_diff(naive.data, want.data);
        record("gaussian5x5 tiled", w, h, d_tiled, 1,
               d_tiled == 0 ? "bit-exact" : "±1 fp contraction");
        record("gaussian5x5 naive", w, h, d_naive, 1,
               d_naive == 0 ? "bit-exact" : "±1 fp contraction");
        record("tiled vs naive", w, h, max_abs_diff(tiled.data, naive.data), 0,
               "same op order: bit-exact");
    }
    {
        cig::Image want(w, h, 1);
        cig::cpu::convolve2d(gray_cpu.data.data(), want.data.data(), w, h,
                             cig::Filter::sobel_x);
        const cig::Image got = cig::convolve2d(gray_cpu, cig::Filter::sobel_x, true);
        const int d = max_abs_diff(got.data, want.data);
        record("sobel_x tiled", w, h, d, 1, d == 0 ? "bit-exact" : "±1 fp contraction");
    }

    // --- stage 4: histogram equalization ----------------------------------
    {
        cig::Image want(w, h, 1);
        cig::cpu::equalize_hist(gray_cpu.data.data(), want.data.data(), gray_cpu.n_pixels());
        const cig::Image got = cig::equalize_hist(gray_cpu);
        record("equalize_hist", w, h, max_abs_diff(got.data, want.data), 0,
               "bit-exact required");
    }
}

// GpuBuffer move-semantics smoke test (compile + runtime behavior of the RAII
// wrapper itself).
static void test_gpu_buffer_semantics() {
    cig::GpuBuffer<int> a(1000);
    const int* p = a.get();
    cig::GpuBuffer<int> b(std::move(a));
    const bool moved = b.get() == p && b.size() == 1000 && a.get() == nullptr && a.size() == 0;
    cig::GpuBuffer<int> c;
    c = std::move(b);
    const bool assigned = c.get() == p && b.get() == nullptr;
    record("gpu_buffer move semantics", 0, 0, (moved && assigned) ? 0 : 255, 0, "RAII");
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::fprintf(stderr, "test_gpu: no usable CUDA device\n");
        return 2;
    }
    std::printf("device: %s (%d SMs, sm_%d%d)\n\n", prop.name, prop.multiProcessorCount,
                prop.major, prop.minor);

    try {
        test_gpu_buffer_semantics();
        run_suite(256, 256);
        run_suite(1023, 1021);

        // Grid-stride path: 2100x2100 = 4.41M pixels -> 17227 blocks, capped
        // at 4096, so every histogram thread loops 4-5 times.
        {
            const int w = 2100, h = 2100;
            const cig::Image g = cig::random_image(w, h, 1, 42);
            cig::Image want(w, h, 1);
            cig::cpu::equalize_hist(g.data.data(), want.data.data(), g.n_pixels());
            const cig::Image got = cig::equalize_hist(g);
            record("equalize_hist (grid-stride)", w, h, max_abs_diff(got.data, want.data), 0,
                   "bit-exact required");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_gpu: CUDA failure: %s\n", e.what());
        return 2;
    }

    std::printf("%-28s %-12s %-9s %-4s %-6s %s\n", "op", "size", "max_diff", "tol", "result",
                "note");
    std::printf("%.100s\n",
                "-----------------------------------------------------------------------------"
                "-----------------------");
    for (const Row& r : g_rows) {
        std::printf("%-28s %-12s %-9d %-4d %-6s %s\n", r.op.c_str(), r.size.c_str(),
                    r.max_diff, r.tol, r.max_diff <= r.tol ? "PASS" : "FAIL", r.note.c_str());
    }
    std::printf("\n%s\n", g_all_pass ? "all GPU tests passed" : "GPU TESTS FAILED");
    return g_all_pass ? 0 : 1;
}
