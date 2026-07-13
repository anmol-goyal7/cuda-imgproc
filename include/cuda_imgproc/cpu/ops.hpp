#pragma once
// Single-threaded CPU reference implementations.
//
// These are the ground truth the GPU kernels are tested against, so the rule
// of this file is: *identical arithmetic, not similar arithmetic*. Each op
// mirrors its GPU kernel expression-for-expression — same float literals,
// same operation order, same rounding — so the test suite can demand exact
// (or documented ±1) agreement. Where bit-exactness across two different
// compilers is otherwise not guaranteed (fused multiply-add contraction is
// compiler-discretionary), both sides call fma explicitly: IEEE 754 defines
// fusedMultiplyAdd exactly, so std::fmaf here and fmaf() in device code must
// produce identical bits.
//
// Everything is deliberately naive: row-major loops, no SIMD intrinsics, no
// cache tiling. This is the "one pixel after another" baseline the benchmark
// speedups are measured against (with -O3 -march=native doing what it can).

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../core/filters.hpp"
#include "../core/image.hpp"

namespace cig {
namespace cpu {

namespace detail {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Bilinear interpolation over the 4 integer pixels around (xs, ys), channel c
// of an interleaved image. Coordinates outside the image are clamped to the
// edge. Mirrors bilinear_sample() in kernels/augment.cuh term for term.
inline std::uint8_t bilinear_sample(const std::uint8_t* src, int w, int h, int ch, int c,
                                    float xs, float ys) {
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

}  // namespace detail

// ------------------------------------------------------------------ stage 1

// BT.601 luma: Y = 0.299 R + 0.587 G + 0.114 B, round to nearest.
// Evaluated as fmaf(0.299,R, fmaf(0.587,G, 0.114*B)) on both CPU and GPU —
// see the file comment for why the explicit fma chain matters.
inline void rgb_to_gray(const std::uint8_t* rgb, std::uint8_t* gray, int w, int h) {
    const std::size_t n = static_cast<std::size_t>(w) * h;
    for (std::size_t i = 0; i < n; ++i) {
        const float r = rgb[3 * i + 0];
        const float g = rgb[3 * i + 1];
        const float b = rgb[3 * i + 2];
        const float y = std::fmaf(0.299f, r, std::fmaf(0.587f, g, 0.114f * b));
        gray[i] = static_cast<std::uint8_t>(y + 0.5f);
    }
}

// RGB -> HSV with OpenCV's uint8 encoding: H in [0,180) (degrees halved),
// S and V scaled to [0,255].
inline void rgb_to_hsv(const std::uint8_t* rgb, std::uint8_t* hsv, int w, int h) {
    const std::size_t n = static_cast<std::size_t>(w) * h;
    for (std::size_t i = 0; i < n; ++i) {
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
            if (hh >= 180) hh -= 180;                      // guard the 359.9°/2 rounding wrap
            h8 = static_cast<std::uint8_t>(hh);
        }

        hsv[3 * i + 0] = h8;
        hsv[3 * i + 1] = s8;
        hsv[3 * i + 2] = static_cast<std::uint8_t>(v);
    }
}

// ------------------------------------------------------------------ stage 2

inline void flip_horizontal(const std::uint8_t* src, std::uint8_t* dst, int w, int h, int ch) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t d = (static_cast<std::size_t>(y) * w + x) * ch;
            const std::size_t s = (static_cast<std::size_t>(y) * w + (w - 1 - x)) * ch;
            for (int c = 0; c < ch; ++c) dst[d + c] = src[s + c];
        }
    }
}

inline void flip_vertical(const std::uint8_t* src, std::uint8_t* dst, int w, int h, int ch) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t d = (static_cast<std::size_t>(y) * w + x) * ch;
            const std::size_t s = (static_cast<std::size_t>(h - 1 - y) * w + x) * ch;
            for (int c = 0; c < ch; ++c) dst[d + c] = src[s + c];
        }
    }
}

// Rotation by theta (radians, counter-clockwise) about the image center
// ((W-1)/2, (H-1)/2), via inverse mapping: for each *output* pixel, rotate
// its coordinates backwards to find where it came from in the source, then
// interpolate. Out-of-range sources clamp to the edge.
inline void rotate(const std::uint8_t* src, std::uint8_t* dst, int w, int h, int ch,
                   float theta) {
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float xs = ct * dx + st * dy + cx;
            const float ys = -st * dx + ct * dy + cy;
            for (int c = 0; c < ch; ++c) {
                dst[(static_cast<std::size_t>(y) * w + x) * ch + c] =
                    detail::bilinear_sample(src, w, h, ch, c, xs, ys);
            }
        }
    }
}

// Bilinear resize. Output pixel (x, y) samples source (x*sx, y*sy) with
// sx = srcW/dstW, sy = srcH/dstH.
inline void resize(const std::uint8_t* src, int sw, int sh, std::uint8_t* dst, int dw, int dh,
                   int ch) {
    const float sx = static_cast<float>(sw) / static_cast<float>(dw);
    const float sy = static_cast<float>(sh) / static_cast<float>(dh);
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const float xs = static_cast<float>(x) * sx;
            const float ys = static_cast<float>(y) * sy;
            for (int c = 0; c < ch; ++c) {
                dst[(static_cast<std::size_t>(y) * dw + x) * ch + c] =
                    detail::bilinear_sample(src, sw, sh, ch, c, xs, ys);
            }
        }
    }
}

// ------------------------------------------------------------------ stage 3

// Single-channel 2D convolution with a k*k kernel (k odd), clamp-to-edge
// borders. Accumulates in float via explicit fma, innermost loop over kernel
// columns — the GPU kernels use the exact same order, which is what makes
// exact output comparison meaningful.
inline void convolve2d(const std::uint8_t* src, std::uint8_t* dst, int w, int h,
                       const float* wgt, int k) {
    const int r = k / 2;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) {
                const int sy = detail::clampi(y + i - r, 0, h - 1);
                for (int j = 0; j < k; ++j) {
                    const int sx = detail::clampi(x + j - r, 0, w - 1);
                    acc = std::fmaf(
                        wgt[i * k + j],
                        static_cast<float>(src[static_cast<std::size_t>(sy) * w + sx]), acc);
                }
            }
            acc = detail::clampf(acc, 0.0f, 255.0f);
            dst[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(acc + 0.5f);
        }
    }
}

inline void convolve2d(const std::uint8_t* src, std::uint8_t* dst, int w, int h, Filter f) {
    convolve2d(src, dst, w, h, filter_weights(f), filter_size(f));
}

// ------------------------------------------------------------------ stage 4

// Histogram equalization (single channel), the standard cdf_min formulation:
//   lut[v] = round(255 * (cdf[v] - cdf_min) / (N - cdf_min))
// where cdf is the inclusive histogram CDF and cdf_min its first nonzero
// entry. The LUT arithmetic is done in double on both CPU and GPU: unlike
// float, double division is unaffected by --use_fast_math, so both sides
// round identically and the outputs match exactly.
inline void equalize_hist(const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
    unsigned int hist[256] = {0};
    for (std::size_t i = 0; i < n; ++i) ++hist[src[i]];

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
            // Degenerate input: every pixel has the same value. The formula
            // is 0/0; map identically instead.
            lut[v] = static_cast<std::uint8_t>(v);
        } else {
            double num = static_cast<double>(cdf[v]) - static_cast<double>(cdf_min);
            if (num < 0.0) num = 0.0;  // bins below the first occupied one (unused)
            lut[v] = static_cast<std::uint8_t>(255.0 * num / static_cast<double>(denom) + 0.5);
        }
    }

    for (std::size_t i = 0; i < n; ++i) dst[i] = lut[src[i]];
}

// -------------------------------------------------- Image-level convenience

inline Image rgb_to_gray(const Image& img) {
    Image out(img.width, img.height, 1);
    rgb_to_gray(img.data.data(), out.data.data(), img.width, img.height);
    return out;
}

inline Image rgb_to_hsv(const Image& img) {
    Image out(img.width, img.height, 3);
    rgb_to_hsv(img.data.data(), out.data.data(), img.width, img.height);
    return out;
}

inline Image flip_horizontal(const Image& img) {
    Image out(img.width, img.height, img.channels);
    flip_horizontal(img.data.data(), out.data.data(), img.width, img.height, img.channels);
    return out;
}

inline Image flip_vertical(const Image& img) {
    Image out(img.width, img.height, img.channels);
    flip_vertical(img.data.data(), out.data.data(), img.width, img.height, img.channels);
    return out;
}

inline Image rotate(const Image& img, float theta) {
    Image out(img.width, img.height, img.channels);
    rotate(img.data.data(), out.data.data(), img.width, img.height, img.channels, theta);
    return out;
}

inline Image resize(const Image& img, int dw, int dh) {
    Image out(dw, dh, img.channels);
    resize(img.data.data(), img.width, img.height, out.data.data(), dw, dh, img.channels);
    return out;
}

// Multi-channel images are convolved channel-plane by channel-plane (the
// kernel itself is single-channel); split/merge happens here on the host.
inline Image convolve2d(const Image& img, Filter f) {
    Image out(img.width, img.height, img.channels);
    if (img.channels == 1) {
        convolve2d(img.data.data(), out.data.data(), img.width, img.height, f);
        return out;
    }
    const std::size_t n = img.n_pixels();
    std::vector<std::uint8_t> plane(n), filtered(n);
    for (int c = 0; c < img.channels; ++c) {
        for (std::size_t i = 0; i < n; ++i) plane[i] = img.data[i * img.channels + c];
        convolve2d(plane.data(), filtered.data(), img.width, img.height, f);
        for (std::size_t i = 0; i < n; ++i) out.data[i * img.channels + c] = filtered[i];
    }
    return out;
}

inline Image equalize_hist(const Image& img) {
    Image out(img.width, img.height, img.channels);
    // Defined for single-channel images (equalizing RGB channels
    // independently shifts hues; convert to gray or HSV first).
    equalize_hist(img.data.data(), out.data.data(), img.n_pixels() * img.channels);
    return out;
}

}  // namespace cpu
}  // namespace cig
