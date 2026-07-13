// make_test_image — writes the deterministic synthetic test PNG (gradients +
// noise + two disks) used as input for examples/pipeline_demo.
//
//     ./bin/make_test_image [out.png] [width] [height]

#include <cstdio>
#include <cstdlib>
#include <string>

#include "cuda_imgproc.hpp"

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "test_image.png";
    const int w = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int h = argc > 3 ? std::atoi(argv[3]) : 768;
    if (w <= 0 || h <= 0) {
        std::fprintf(stderr, "usage: %s [out.png] [width] [height]\n", argv[0]);
        return 1;
    }

    const cig::Image img = cig::make_synthetic(w, h, 3);
    cig::save_png(path, img);
    std::printf("wrote %s (%dx%d RGB)\n", path.c_str(), w, h);
    return 0;
}
