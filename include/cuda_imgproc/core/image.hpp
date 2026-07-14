#pragma once
// Host-side image container + PNG I/O + synthetic image generators.
//
// The GPU side of this library never sees this type: kernels operate on raw
// device pointers wrapped in GpuBuffer<uint8_t>. Image is the host staging
// area — a dumb, contiguous, row-major, channel-interleaved byte array —
// so a host<->device transfer is always a single flat cudaMemcpy.
//
// Layout: data[(y * width + x) * channels + c], matching what stb produces
// and what every kernel in this library assumes.

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Vendored stb headers (third_party/), pinned to commit
// 31c1ad37456438565541f4919958214b6e762fb4 of github.com/nothings/stb
// (stb_image v2.30, stb_image_write v1.16).
//
// *_STATIC gives every translation unit its own internal-linkage copy of the
// implementation, which is the only way a header-only library can bundle stb
// without forcing exactly one TU to define STB_IMAGE_IMPLEMENTATION. The cost
// (duplicated object code per TU) is irrelevant for this project's handful of
// single-TU programs. The diagnostic pragmas silence warnings that vendored
// code triggers under -Wall -Wextra; they are not ours to fix.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace cig {

namespace detail {
// Shared precondition check for the public API. Image ops fed the wrong shape
// (a grayscale image into rgb_to_gray, an even filter size, ...) would corrupt
// memory rather than fail, so the host-side entry points validate and throw.
inline void require(bool cond, const char* msg) {
    if (!cond) {
        throw std::invalid_argument(msg);
    }
}
}  // namespace detail

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;                // 1 = grayscale, 3 = interleaved RGB
    std::vector<std::uint8_t> data;  // size = width * height * channels

    Image() = default;
    Image(int w, int h, int c)
        : width(w),
          height(h),
          channels(c),
          data(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
               static_cast<std::size_t>(c)) {}

    std::size_t size() const { return data.size(); }
    std::size_t n_pixels() const {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }

    std::uint8_t& at(int x, int y, int c = 0) {
        return data[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
    std::uint8_t at(int x, int y, int c = 0) const {
        return data[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
};

// Loads a PNG (or anything else stb groks). desired_channels = 0 keeps the
// file's channel count; 1 or 3 forces a conversion inside stb.
inline Image load_png(const std::string& path, int desired_channels = 0) {
    int w = 0, h = 0, c = 0;
    std::uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &c, desired_channels);
    if (pixels == nullptr) {
        throw std::runtime_error("load_png: failed to load '" + path +
                                 "': " + stbi_failure_reason());
    }
    if (desired_channels != 0) {
        c = desired_channels;
    }
    Image img(w, h, c);
    img.data.assign(pixels, pixels + img.size());
    stbi_image_free(pixels);
    return img;
}

inline void save_png(const std::string& path, const Image& img) {
    const int ok = stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                                  img.data.data(), img.width * img.channels);
    if (ok == 0) {
        throw std::runtime_error("save_png: failed to write '" + path + "'");
    }
}

// Deterministic synthetic test image: smooth color gradients (exercise
// interpolation), additive noise (exercise histograms/filters), and a few
// bright disks (visible structure, so blur/equalization results can be judged
// by eye). Used by tools/make_test_image and the pipeline example.
inline Image make_synthetic(int w, int h, int channels = 3, std::uint32_t seed = 42) {
    Image img(w, h, channels);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(-24, 24);

    const int cx = w / 2, cy = h / 2;
    const int r_disk = (w < h ? w : h) / 6;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Base gradients, per channel.
            int base[3];
            base[0] = (w > 1) ? x * 255 / (w - 1) : 0;  // R: left->right ramp
            base[1] = (h > 1) ? y * 255 / (h - 1) : 0;  // G: top->bottom ramp
            base[2] = (base[0] + base[1]) / 2;          // B: diagonal ramp

            // Two disks: one bright, one dark, offset from center.
            const long long dx1 = x - (cx - w / 5), dy1 = y - (cy - h / 5);
            const long long dx2 = x - (cx + w / 5), dy2 = y - (cy + h / 5);
            const bool bright = dx1 * dx1 + dy1 * dy1 < 1LL * r_disk * r_disk;
            const bool dark = dx2 * dx2 + dy2 * dy2 < 1LL * r_disk * r_disk;

            for (int c = 0; c < channels; ++c) {
                int v = base[c < 3 ? c : 2];
                if (bright) v += 90;
                if (dark) v -= 90;
                v += noise(rng);
                img.at(x, y, c) = static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    }
    return img;
}

// Uniform pseudo-random image; the seeded workload for tests and benchmarks
// (mt19937 is fully specified by the C++ standard, so the same seed produces
// the same image on every platform — CPU reference and GPU kernel see
// identical inputs everywhere).
inline Image random_image(int w, int h, int channels, std::uint32_t seed = 42) {
    Image img(w, h, channels);
    std::mt19937 rng(seed);
    // Note: uniform_int_distribution<uint8_t> is undefined behavior per the
    // standard (char-sized types are not permitted), hence int + cast.
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& px : img.data) {
        px = static_cast<std::uint8_t>(dist(rng));
    }
    return img;
}

}  // namespace cig
