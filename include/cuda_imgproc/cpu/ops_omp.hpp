#pragma once
// OpenMP-parallel CPU implementations.
//
// Same arithmetic as cpu/ops.hpp (results must be byte-identical — the test
// suite checks), parallelized with `#pragma omp parallel for schedule(dynamic)`
// over image *rows*. Rows are a natural work unit here: each row is
// independent, long enough to amortize scheduling overhead, and contiguous in
// memory so each thread streams cache lines instead of fighting over them.
// schedule(dynamic) load-balances rows across cores, which matters once the
// OS steals time slices from some of them.
//
// These numbers are the honest multicore baseline for the benchmark: the GPU
// should be compared against all the CPU parallelism you could get for free,
// not just one core.
//
// Build with -fopenmp (already in the Makefile's CXXFLAGS, and passed through
// -Xcompiler for nvcc-compiled benchmark binaries).

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../core/filters.hpp"
#include "../core/image.hpp"
#include "ops.hpp"

namespace cig {
namespace cpu_omp {

// ------------------------------------------------------------------ stage 1

inline void rgb_to_gray(const std::uint8_t* rgb, std::uint8_t* gray, int w, int h) {
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        // Row base offsets; arithmetic per pixel matches cpu::rgb_to_gray.
        const std::size_t row = static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const std::size_t i = row + x;
            const float r = rgb[3 * i + 0];
            const float g = rgb[3 * i + 1];
            const float b = rgb[3 * i + 2];
            const float yv = std::fmaf(0.299f, r, std::fmaf(0.587f, g, 0.114f * b));
            gray[i] = static_cast<std::uint8_t>(yv + 0.5f);
        }
    }
}

inline void rgb_to_hsv(const std::uint8_t* rgb, std::uint8_t* hsv, int w, int h) {
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        // Delegate one row at a time to the single-thread reference: a row of
        // an interleaved image is itself a w x 1 image.
        const std::size_t off = static_cast<std::size_t>(y) * w * 3;
        cpu::rgb_to_hsv(rgb + off, hsv + off, w, 1);
    }
}

// ------------------------------------------------------------------ stage 2

inline void flip_horizontal(const std::uint8_t* src, std::uint8_t* dst, int w, int h, int ch) {
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t d = (static_cast<std::size_t>(y) * w + x) * ch;
            const std::size_t s = (static_cast<std::size_t>(y) * w + (w - 1 - x)) * ch;
            for (int c = 0; c < ch; ++c) dst[d + c] = src[s + c];
        }
    }
}

inline void flip_vertical(const std::uint8_t* src, std::uint8_t* dst, int w, int h, int ch) {
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t d = (static_cast<std::size_t>(y) * w + x) * ch;
            const std::size_t s = (static_cast<std::size_t>(h - 1 - y) * w + x) * ch;
            for (int c = 0; c < ch; ++c) dst[d + c] = src[s + c];
        }
    }
}

inline void rotate(const std::uint8_t* src, std::uint8_t* dst, int w, int h, int ch,
                   float theta) {
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float xs = ct * dx + st * dy + cx;
            const float ys = -st * dx + ct * dy + cy;
            for (int c = 0; c < ch; ++c) {
                dst[(static_cast<std::size_t>(y) * w + x) * ch + c] =
                    cpu::detail::bilinear_sample(src, w, h, ch, c, xs, ys);
            }
        }
    }
}

inline void resize(const std::uint8_t* src, int sw, int sh, std::uint8_t* dst, int dw, int dh,
                   int ch) {
    const float sx = static_cast<float>(sw) / static_cast<float>(dw);
    const float sy = static_cast<float>(sh) / static_cast<float>(dh);
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const float xs = static_cast<float>(x) * sx;
            const float ys = static_cast<float>(y) * sy;
            for (int c = 0; c < ch; ++c) {
                dst[(static_cast<std::size_t>(y) * dw + x) * ch + c] =
                    cpu::detail::bilinear_sample(src, sw, sh, ch, c, xs, ys);
            }
        }
    }
}

// ------------------------------------------------------------------ stage 3

inline void convolve2d(const std::uint8_t* src, std::uint8_t* dst, int w, int h,
                       const float* wgt, int k) {
    const int r = k / 2;
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) {
                const int sy = cpu::detail::clampi(y + i - r, 0, h - 1);
                for (int j = 0; j < k; ++j) {
                    const int sx = cpu::detail::clampi(x + j - r, 0, w - 1);
                    acc = std::fmaf(
                        wgt[i * k + j],
                        static_cast<float>(src[static_cast<std::size_t>(sy) * w + sx]), acc);
                }
            }
            acc = cpu::detail::clampf(acc, 0.0f, 255.0f);
            dst[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(acc + 0.5f);
        }
    }
}

inline void convolve2d(const std::uint8_t* src, std::uint8_t* dst, int w, int h, Filter f) {
    convolve2d(src, dst, w, h, filter_weights(f), filter_size(f));
}

// ------------------------------------------------------------------ stage 4

inline void equalize_hist(const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
    // Histogram accumulation: integer adds are associative, so an OpenMP
    // array reduction (per-thread private histograms merged at the barrier —
    // the same trick the GPU plays with per-block shared-memory histograms)
    // gives bit-identical totals to the serial loop.
    unsigned int hist[256] = {0};
    const long long nn = static_cast<long long>(n);
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : hist[:256])
    for (long long i = 0; i < nn; ++i) ++hist[src[i]];

    // CDF + LUT: 256 entries, not worth parallelizing. Identical to
    // cpu::equalize_hist.
    unsigned int cdf[256];
    unsigned int running = 0;
    for (int v = 0; v < 256; ++v) {
        running += hist[v];
        cdf[v] = running;
    }
    unsigned int cdf_min = 0;
    for (int v = 0; v < 256; ++v) {
        if (cdf[v] != 0) {
            cdf_min = cdf[v];
            break;
        }
    }
    std::uint8_t lut[256];
    const unsigned int denom = static_cast<unsigned int>(n) - cdf_min;
    for (int v = 0; v < 256; ++v) {
        if (denom == 0) {
            lut[v] = static_cast<std::uint8_t>(v);
        } else {
            double num = static_cast<double>(cdf[v]) - static_cast<double>(cdf_min);
            if (num < 0.0) num = 0.0;
            lut[v] = static_cast<std::uint8_t>(255.0 * num / static_cast<double>(denom) + 0.5);
        }
    }

#pragma omp parallel for schedule(dynamic, 4096)
    for (long long i = 0; i < nn; ++i) dst[i] = lut[src[i]];
}

}  // namespace cpu_omp
}  // namespace cig
