// test_cpu — local correctness suite for the CPU reference implementations.
//
// Tiny handcrafted fixtures with outputs computed by hand (the arithmetic is
// spelled out in comments next to each expected value), plus a few structural
// properties (involutions, identities, histogram flattening). No framework:
// prints PASS/FAIL per test, exits nonzero on any failure.
//
// Run: make test-cpu

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_imgproc.hpp"

static int g_failures = 0;

static void report(bool ok, const char* name, const std::string& detail = "") {
    if (ok) {
        std::printf("PASS %s\n", name);
    } else {
        std::printf("FAIL %s (%s)\n", name, detail.c_str());
        ++g_failures;
    }
}

// Compares two byte buffers; on mismatch, describes the first differing index.
static bool equal_bytes(const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want,
                        std::string* detail) {
    if (got.size() != want.size()) {
        *detail = "size " + std::to_string(got.size()) + " != " + std::to_string(want.size());
        return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            *detail = "index " + std::to_string(i) + ": got " + std::to_string(got[i]) + ", want " +
                      std::to_string(want[i]);
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------------- stage 1

static void test_rgb_to_gray() {
    // 4x4 RGB image; expected Y computed by hand from
    // y = 0.299 R + 0.587 G + 0.114 B, rounded via (uint8)(y + 0.5).
    // All chosen values sit far from rounding boundaries, so hand decimal
    // arithmetic and float arithmetic agree.
    // Formatter guard: fixture tables are hand-aligned.
    // clang-format off
    const std::vector<std::uint8_t> rgb = {
        255, 0,   0,   0,   255, 0,   0,   0,   255, 255, 255, 255,  // row 0
        0,   0,   0,   128, 128, 128, 100, 100, 100, 50,  100, 150,  // row 1
        10,  20,  30,  200, 150, 100, 25,  50,  75,  240, 10,  3,    // row 2
        1,   1,   1,   254, 253, 252, 60,  120, 240, 90,  45,  180,  // row 3
    };
    const std::vector<std::uint8_t> want = {
        76,  // (255,0,0):   76.245        -> 76
        150, // (0,255,0):   149.685       -> 150
        29,  // (0,0,255):   29.07         -> 29
        255, // (255,255,255): 255.0       -> 255
        0,   // (0,0,0):     0.0           -> 0
        128, // (128,128,128): 128.0       -> 128
        100, // (100,100,100): 100.0       -> 100
        91,  // (50,100,150): 14.95+58.7+17.1  = 90.75  -> 91
        18,  // (10,20,30):   2.99+11.74+3.42  = 18.15  -> 18
        159, // (200,150,100): 59.8+88.05+11.4 = 159.25 -> 159
        45,  // (25,50,75):   7.475+29.35+8.55 = 45.375 -> 45
        78,  // (240,10,3):   71.76+5.87+0.342 = 77.972 -> 78
        1,   // (1,1,1):      1.0           -> 1
        253, // (254,253,252): 75.946+148.511+28.728 = 253.185 -> 253
        116, // (60,120,240): 17.94+70.44+27.36 = 115.74 -> 116
        74,  // (90,45,180):  26.91+26.415+20.52 = 73.845 -> 74
    };
    // clang-format on
    std::vector<std::uint8_t> got(16);
    cig::cpu::rgb_to_gray(rgb.data(), got.data(), 4, 4);
    std::string d;
    report(equal_bytes(got, want, &d), "rgb_to_gray fixture", d);
}

static void test_rgb_to_hsv() {
    // Expected values follow OpenCV's uint8 encoding: H = round(hue°/2),
    // S = round(255*(V-min)/V), V = max(R,G,B). Hand computations:
    //   (255,0,0):    hue 0    -> H 0;   S 255; V 255
    //   (0,255,0):    hue 120  -> H 60;  S 255; V 255
    //   (0,0,255):    hue 240  -> H 120; S 255; V 255
    //   (0,0,0):      degenerate: H 0, S 0, V 0
    //   (255,255,255): diff 0:  H 0; S 0; V 255
    //   (255,128,0):  hue 60*128/255 = 30.118 -> H round(15.059+0.44...) = 15; S 255; V 255
    //   (100,150,200): max B: hue 240+60*(100-150)/100 = 210 -> H 105;
    //                  S round(255*100/200) = 128; V 200
    //   (200,100,150): max R: hue 60*(100-150)/100 = -30 -> +360 = 330 -> H 165; S 128; V 200
    // clang-format off
    const std::vector<std::uint8_t> rgb = {
        255, 0,   0,   0,   255, 0,   0,   0,   255, 0,   0,   0,
        255, 255, 255, 255, 128, 0,   100, 150, 200, 200, 100, 150,
    };
    const std::vector<std::uint8_t> want = {
        0,   255, 255, 60,  255, 255, 120, 255, 255, 0,   0,   0,
        0,   0,   255, 15,  255, 255, 105, 128, 200, 165, 128, 200,
    };
    // clang-format on
    std::vector<std::uint8_t> got(24);
    cig::cpu::rgb_to_hsv(rgb.data(), got.data(), 4, 2);
    std::string d;
    report(equal_bytes(got, want, &d), "rgb_to_hsv fixture", d);
}

// --------------------------------------------------------------- stage 2

static void test_flips() {
    // 4x4 single channel, values 0..15 laid out row-major:
    //    0  1  2  3        3  2  1  0        12 13 14 15
    //    4  5  6  7   H->  7  6  5  4   V->   8  9 10 11
    //    8  9 10 11       11 10  9  8         4  5  6  7
    //   12 13 14 15       15 14 13 12         0  1  2  3
    std::vector<std::uint8_t> src(16);
    for (int i = 0; i < 16; ++i) src[i] = static_cast<std::uint8_t>(i);
    // clang-format off
    const std::vector<std::uint8_t> want_h = { 3,  2,  1,  0,   7,  6,  5,  4,
                                              11, 10,  9,  8,  15, 14, 13, 12};
    const std::vector<std::uint8_t> want_v = {12, 13, 14, 15,   8,  9, 10, 11,
                                               4,  5,  6,  7,   0,  1,  2,  3};
    // clang-format on

    std::vector<std::uint8_t> got(16);
    std::string d;
    cig::cpu::flip_horizontal(src.data(), got.data(), 4, 4, 1);
    report(equal_bytes(got, want_h, &d), "flip_horizontal fixture", d);
    cig::cpu::flip_vertical(src.data(), got.data(), 4, 4, 1);
    report(equal_bytes(got, want_v, &d), "flip_vertical fixture", d);

    // 2x2 RGB: horizontal flip must move whole pixels, not scramble channels.
    //   (10,11,12)(20,21,22)      (20,21,22)(10,11,12)
    //   (30,31,32)(40,41,42)  ->  (40,41,42)(30,31,32)
    // clang-format off
    const std::vector<std::uint8_t> rgb      = {10, 11, 12, 20, 21, 22,
                                                30, 31, 32, 40, 41, 42};
    const std::vector<std::uint8_t> want_rgb = {20, 21, 22, 10, 11, 12,
                                                40, 41, 42, 30, 31, 32};
    // clang-format on
    std::vector<std::uint8_t> got_rgb(12);
    cig::cpu::flip_horizontal(rgb.data(), got_rgb.data(), 2, 2, 3);
    report(equal_bytes(got_rgb, want_rgb, &d), "flip_horizontal 3-channel fixture", d);

    // Involution: flipping twice restores the original.
    const cig::Image img = cig::random_image(33, 21, 3, 7);
    const cig::Image hh = cig::cpu::flip_horizontal(cig::cpu::flip_horizontal(img));
    const cig::Image vv = cig::cpu::flip_vertical(cig::cpu::flip_vertical(img));
    report(equal_bytes(hh.data, img.data, &d), "flip_horizontal involution", d);
    report(equal_bytes(vv.data, img.data, &d), "flip_vertical involution", d);
}

static void test_rotate() {
    const cig::Image img = cig::random_image(16, 12, 3, 11);
    std::string d;

    // theta = 0: xs = (x-cx)+cx. All quantities are half-integers, exactly
    // representable in float, so the mapping is exact and fx = fy = 0.
    const cig::Image r0 = cig::cpu::rotate(img, 0.0f);
    report(equal_bytes(r0.data, img.data, &d), "rotate(0) is identity", d);

    // theta = pi: equivalent to flipping both axes. float(pi) is ~8.7e-8 off
    // from pi, which perturbs source coordinates by <1e-6 px; after bilinear
    // interpolation and rounding the result is still exact.
    const cig::Image rp = cig::cpu::rotate(img, 3.14159265358979323846f);
    const cig::Image fboth = cig::cpu::flip_vertical(cig::cpu::flip_horizontal(img));
    report(equal_bytes(rp.data, fboth.data, &d), "rotate(pi) equals flip both axes", d);
}

static void test_resize() {
    std::string d;

    // Identity resize: sx = sy = 1, so xs = x exactly -> pure copy.
    const cig::Image img = cig::random_image(37, 23, 3, 5);
    const cig::Image same = cig::cpu::resize(img, 37, 23);
    report(equal_bytes(same.data, img.data, &d), "resize identity", d);

    // 2x1 -> 4x1 upscale, sx = 0.5, center-aligned: xs = (x+0.5)*0.5 - 0.5,
    // i.e. sample points -0.25, 0.25, 0.75, 1.25.
    //   xs=-0.25 -> clamped to 0.0 -> 10;   xs=0.25 -> 0.75*10 + 0.25*30 = 15;
    //   xs= 0.75 -> 0.25*10 + 0.75*30 = 25; xs=1.25 -> clamped to 1.0 -> 30.
    const std::vector<std::uint8_t> src = {10, 30};
    const std::vector<std::uint8_t> want = {10, 15, 25, 30};
    std::vector<std::uint8_t> got(4);
    cig::cpu::resize(src.data(), 2, 1, got.data(), 4, 1, 1);
    report(equal_bytes(got, want, &d), "resize 2x1 -> 4x1 fixture", d);
}

// --------------------------------------------------------------- stage 3

static void test_convolve() {
    std::string d;

    // Sharpen ([[0,-1,0],[-1,5,-1],[0,-1,0]]) on a 4x4 with one hot pixel.
    // Integer weights on integer pixels: float arithmetic is exact, so these
    // are true hand computations. Borders clamp to the edge.
    //   input:            expected (out = 5*c - up - down - left - right):
    //   10 10 10 10       10   0  10  10     e.g. (1,1): 5*50-4*10      = 210
    //   10 50 10 10        0 210   0  10          (1,0): 5*10-10-50-10-10 = -30 -> 0
    //   10 10 10 10       10   0  10  10          (0,0): 5*10-10-10-10-10 =  10
    //   10 10 10 10       10  10  10  10
    const std::vector<std::uint8_t> src = {10, 10, 10, 10, 10, 50, 10, 10,
                                           10, 10, 10, 10, 10, 10, 10, 10};
    const std::vector<std::uint8_t> want = {10, 0, 10, 10, 0,  210, 0,  10,
                                            10, 0, 10, 10, 10, 10,  10, 10};
    std::vector<std::uint8_t> got(16);
    cig::cpu::convolve2d(src.data(), got.data(), 4, 4, cig::Filter::sharpen);
    report(equal_bytes(got, want, &d), "convolve2d sharpen fixture", d);

    // Laplacian sums to 0 -> constant image maps to all zeros.
    const std::vector<std::uint8_t> flat(16, 77);
    const std::vector<std::uint8_t> zeros(16, 0);
    cig::cpu::convolve2d(flat.data(), got.data(), 4, 4, cig::Filter::laplacian);
    report(equal_bytes(got, zeros, &d), "convolve2d laplacian of constant is zero", d);

    // Gaussian weights sum to ~1 (the table is rounded to float). Running the
    // actual fmaf accumulation over a constant-100 image gives 99.99999237...;
    // + 0.5 truncates to 100, so with clamp-to-edge borders a constant image
    // must come back unchanged.
    const std::vector<std::uint8_t> flat100(64, 100);
    std::vector<std::uint8_t> got100(64);
    cig::cpu::convolve2d(flat100.data(), got100.data(), 8, 8, cig::Filter::gaussian5x5);
    report(equal_bytes(got100, flat100, &d), "convolve2d gaussian preserves constant", d);
}

// --------------------------------------------------------------- stage 4

static void test_equalize() {
    std::string d;

    // 4x4 ramp 0..15: hist[v] = 1, inclusive cdf[v] = v+1, cdf_min = 1,
    // lut[v] = round(255*(v+1-1)/(16-1)) = round(17*v) = 17*v exactly.
    std::vector<std::uint8_t> ramp(16);
    for (int i = 0; i < 16; ++i) ramp[i] = static_cast<std::uint8_t>(i);
    std::vector<std::uint8_t> want(16);
    for (int i = 0; i < 16; ++i) want[i] = static_cast<std::uint8_t>(17 * i);
    std::vector<std::uint8_t> got(16);
    cig::cpu::equalize_hist(ramp.data(), got.data(), 16);
    report(equal_bytes(got, want, &d), "equalize_hist ramp fixture", d);

    // Constant image: N - cdf_min = 0 -> degenerate path maps identically.
    const std::vector<std::uint8_t> flat(64, 123);
    std::vector<std::uint8_t> gotf(64);
    cig::cpu::equalize_hist(flat.data(), gotf.data(), 64);
    report(equal_bytes(gotf, flat, &d), "equalize_hist constant image is identity", d);

    // Property: equalization flattens the histogram. Squeeze a random image
    // into [96,160) and check the output CDF is closer to the ideal linear
    // ramp than the input CDF (L1 distance over the 256 normalized bins).
    cig::Image img = cig::random_image(64, 64, 1, 42);
    for (auto& p : img.data) p = static_cast<std::uint8_t>(96 + p % 64);
    cig::Image eq(64, 64, 1);
    cig::cpu::equalize_hist(img.data.data(), eq.data.data(), img.n_pixels());

    auto cdf_l1_from_uniform = [](const std::vector<std::uint8_t>& px) {
        std::vector<double> cdf(256, 0.0);
        for (std::uint8_t v : px) cdf[v] += 1.0;
        double run = 0.0, dist = 0.0;
        for (int v = 0; v < 256; ++v) {
            run += cdf[v] / static_cast<double>(px.size());
            dist += std::fabs(run - static_cast<double>(v + 1) / 256.0);
        }
        return dist;
    };
    const double before = cdf_l1_from_uniform(img.data);
    const double after = cdf_l1_from_uniform(eq.data);
    report(after < before, "equalize_hist flattens CDF",
           "L1(before)=" + std::to_string(before) + " L1(after)=" + std::to_string(after));
}

// --------------------------------------------------------------- OpenMP

// The OpenMP variants must be byte-identical to the single-thread reference:
// every op is either purely per-pixel (row partitioning cannot change
// results) or, for the histogram, an order-independent integer reduction.
static void test_omp_matches_single_thread() {
    const int w = 129, h = 67;  // odd sizes: exercise ragged row splits
    const cig::Image rgb = cig::random_image(w, h, 3, 13);
    const cig::Image gray1 = cig::cpu::rgb_to_gray(rgb);
    std::string d;

    {
        cig::Image got(w, h, 1);
        cig::cpu_omp::rgb_to_gray(rgb.data.data(), got.data.data(), w, h);
        report(equal_bytes(got.data, gray1.data, &d), "omp rgb_to_gray matches", d);
    }
    {
        cig::Image want(w, h, 3), got(w, h, 3);
        cig::cpu::rgb_to_hsv(rgb.data.data(), want.data.data(), w, h);
        cig::cpu_omp::rgb_to_hsv(rgb.data.data(), got.data.data(), w, h);
        report(equal_bytes(got.data, want.data, &d), "omp rgb_to_hsv matches", d);
    }
    {
        cig::Image want(w, h, 3), got(w, h, 3);
        cig::cpu::flip_horizontal(rgb.data.data(), want.data.data(), w, h, 3);
        cig::cpu_omp::flip_horizontal(rgb.data.data(), got.data.data(), w, h, 3);
        report(equal_bytes(got.data, want.data, &d), "omp flip_horizontal matches", d);
        cig::cpu::flip_vertical(rgb.data.data(), want.data.data(), w, h, 3);
        cig::cpu_omp::flip_vertical(rgb.data.data(), got.data.data(), w, h, 3);
        report(equal_bytes(got.data, want.data, &d), "omp flip_vertical matches", d);
    }
    {
        const float theta = 0.6f;
        cig::Image want(w, h, 3), got(w, h, 3);
        cig::cpu::rotate(rgb.data.data(), want.data.data(), w, h, 3, theta);
        cig::cpu_omp::rotate(rgb.data.data(), got.data.data(), w, h, 3, theta);
        report(equal_bytes(got.data, want.data, &d), "omp rotate matches", d);
    }
    {
        cig::Image want(50, 90, 3), got(50, 90, 3);
        cig::cpu::resize(rgb.data.data(), w, h, want.data.data(), 50, 90, 3);
        cig::cpu_omp::resize(rgb.data.data(), w, h, got.data.data(), 50, 90, 3);
        report(equal_bytes(got.data, want.data, &d), "omp resize matches", d);
    }
    {
        cig::Image want(w, h, 1), got(w, h, 1);
        cig::cpu::convolve2d(gray1.data.data(), want.data.data(), w, h, cig::Filter::gaussian5x5);
        cig::cpu_omp::convolve2d(gray1.data.data(), got.data.data(), w, h,
                                 cig::Filter::gaussian5x5);
        report(equal_bytes(got.data, want.data, &d), "omp convolve2d matches", d);
    }
    {
        cig::Image want(w, h, 1), got(w, h, 1);
        cig::cpu::equalize_hist(gray1.data.data(), want.data.data(), gray1.n_pixels());
        cig::cpu_omp::equalize_hist(gray1.data.data(), got.data.data(), gray1.n_pixels());
        report(equal_bytes(got.data, want.data, &d), "omp equalize_hist matches", d);
    }
}

// --------------------------------------------------------------- I/O

static void test_png_roundtrip() {
    const char* path = "test_roundtrip.png";  // CWD: works without a bin/ dir
    const cig::Image img = cig::random_image(25, 17, 3, 99);
    cig::save_png(path, img);
    const cig::Image back = cig::load_png(path);
    std::remove(path);
    std::string d;
    bool ok = back.width == 25 && back.height == 17 && back.channels == 3;
    if (!ok) {
        d = "dimensions mismatch";
    } else {
        ok = equal_bytes(back.data, img.data, &d);
    }
    report(ok, "png save/load roundtrip", d);
}

int main() {
    // The library throws (bad input, failed I/O); an escaped exception should
    // read as a failed run, not a std::terminate backtrace.
    try {
        test_rgb_to_gray();
        test_rgb_to_hsv();
        test_flips();
        test_rotate();
        test_resize();
        test_convolve();
        test_equalize();
        test_omp_matches_single_thread();
        test_png_roundtrip();
    } catch (const std::exception& e) {
        std::printf("FAIL unhandled exception: %s\n", e.what());
        ++g_failures;
    }

    if (g_failures != 0) {
        std::printf("\n%d test(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall CPU tests passed\n");
    return 0;
}
