// pipeline_demo — the whole library in one pipeline, on the GPU:
//
//     PNG -> rgb_to_gray -> gaussian5x5 blur -> histogram equalization
//
// writing each stage's output next to the input so the effect of every kernel
// can be eyeballed:
//
//     ./bin/pipeline_demo [input.png]
//
// With no argument it generates the synthetic test image (gradients + noise +
// disks — same generator as tools/make_test_image) so the demo runs without
// any assets. Outputs: out_gray.png, out_gaussian.png, out_equalized.png.
//
// This file uses only the Image-level convenience API, where every call is
// upload -> kernel(s) -> download. That is the simplest correct usage — and
// deliberately not the fastest: three ops pay three round trips over PCIe.
// The GpuBuffer-level API (see tests/ and bench/) keeps data resident on the
// device across ops; the benchmark's gpu_wall vs gpu_kernel columns quantify
// exactly what those round trips cost.

#include <cstdio>
#include <exception>
#include <string>

#include "cuda_imgproc.hpp"

int main(int argc, char** argv) {
    try {
        cig::Image rgb;
        if (argc > 1) {
            rgb = cig::load_png(argv[1], 3);  // force 3 channels
            std::printf("loaded %s (%dx%d)\n", argv[1], rgb.width, rgb.height);
        } else {
            rgb = cig::make_synthetic(1024, 768, 3);
            std::printf("no input given — using synthetic 1024x768 test image\n");
        }

        // Stage 1: color conversion.
        const cig::Image gray = cig::rgb_to_gray(rgb);
        cig::save_png("out_gray.png", gray);
        std::printf("out_gray.png       (BT.601 luma)\n");

        // Stage 3: 5x5 gaussian blur, shared-memory tiled kernel.
        const cig::Image blurred = cig::convolve2d(gray, cig::Filter::gaussian5x5);
        cig::save_png("out_gaussian.png", blurred);
        std::printf("out_gaussian.png   (gaussian 5x5, sigma=1)\n");

        // Stage 4: histogram equalization (3 kernel launches).
        const cig::Image equalized = cig::equalize_hist(blurred);
        cig::save_png("out_equalized.png", equalized);
        std::printf("out_equalized.png  (histogram equalization)\n");

        return 0;
    } catch (const std::exception& e) {
        // Every CUDA call in the library funnels errors here via CUDA_CHECK.
        std::fprintf(stderr, "pipeline_demo failed: %s\n", e.what());
        return 1;
    }
}
