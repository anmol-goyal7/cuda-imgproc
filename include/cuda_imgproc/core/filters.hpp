#pragma once
// The convolution filter bank — single source of truth for filter weights.
//
// Both the CPU reference implementation and the GPU constant-memory upload
// read weights from this table. That is what makes exact CPU-vs-GPU output
// comparison possible: identical inputs, identical weights, identical
// accumulation order (see cpu/ops.hpp and kernels/convolution.cuh).
//
// This header is deliberately CUDA-free so host-only translation units can
// use it.

#include <stdexcept>

namespace cig {

// Largest filter this library supports; sizes the __constant__ array on the
// GPU side. 7 leaves headroom over the 5x5 gaussian without wasting the
// (64 KB total) constant memory budget.
constexpr int kMaxFilterSize = 7;

enum class Filter {
    gaussian5x5,
    sobel_x,
    sobel_y,
    laplacian,
    sharpen,
};

inline int filter_size(Filter f) {
    return f == Filter::gaussian5x5 ? 5 : 3;
}

// Row-major k*k weight table for each filter.
inline const float* filter_weights(Filter f) {
    // Gaussian blur, sigma = 1.0. Weights are G(x,y) = exp(-(x^2+y^2)/(2*sigma^2))
    // evaluated at integer offsets x,y in [-2,2], then normalized to sum 1
    // (normalizer S = sum of the 25 raw values = 6.168924081028881). The
    // resulting doubles are rounded to float and hardcoded so every build, on
    // every platform, convolves with bit-identical weights.
    static const float gaussian5x5[25] = {
        0.00296901679f, 0.0133062098f, 0.0219382308f, 0.0133062098f, 0.00296901679f,
        0.0133062098f,  0.0596342944f, 0.0983203277f, 0.0596342944f, 0.0133062098f,
        0.0219382308f,  0.0983203277f, 0.162102818f,  0.0983203277f, 0.0219382308f,
        0.0133062098f,  0.0596342944f, 0.0983203277f, 0.0596342944f, 0.0133062098f,
        0.00296901679f, 0.0133062098f, 0.0219382308f, 0.0133062098f, 0.00296901679f,
    };
    // Horizontal gradient (responds to vertical edges).
    static const float sobel_x[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f,
    };
    // Vertical gradient (responds to horizontal edges).
    static const float sobel_y[9] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f,
    };
    // Discrete Laplacian: second derivative, responds to any edge/corner.
    static const float laplacian[9] = {
        0.0f,  1.0f, 0.0f,
        1.0f, -4.0f, 1.0f,
        0.0f,  1.0f, 0.0f,
    };
    // Identity + Laplacian-based edge boost.
    static const float sharpen[9] = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f,
    };

    switch (f) {
        case Filter::gaussian5x5: return gaussian5x5;
        case Filter::sobel_x:     return sobel_x;
        case Filter::sobel_y:     return sobel_y;
        case Filter::laplacian:   return laplacian;
        case Filter::sharpen:     return sharpen;
    }
    throw std::invalid_argument("filter_weights: unknown filter");
}

inline const char* filter_name(Filter f) {
    switch (f) {
        case Filter::gaussian5x5: return "gaussian5x5";
        case Filter::sobel_x:     return "sobel_x";
        case Filter::sobel_y:     return "sobel_y";
        case Filter::laplacian:   return "laplacian";
        case Filter::sharpen:     return "sharpen";
    }
    return "unknown";
}

}  // namespace cig
