# cuda-imgproc

Header-only C++17/CUDA image-processing library built from first principles — every kernel written to be read, benchmarked against CPU baselines it must match bit-for-bit.

**Status: under construction.** The library is landing in reviewable stages (core → CPU baselines → GPU kernels → tests → benchmarks); see the commit history. The full README arrives with the final stage.

Build system is a plain Makefile: `make cpu && make test-cpu` runs the local CPU suite; GPU targets require `nvcc`.

License: MIT (see [LICENSE](LICENSE)).
