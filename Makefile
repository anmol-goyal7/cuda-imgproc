# cuda-imgproc — plain Makefile, no CMake by design.
#
# Targets:
#   cpu             build all CPU-only binaries that exist (tests, bench, tools)
#   gpu             build all CUDA binaries (needs nvcc; skipped gracefully otherwise)
#   test-cpu        build + run the local CPU test suite
#   test-gpu        build + run the GPU-vs-CPU correctness suite (GPU machine only)
#   bench-cpu       build + run the CPU-only benchmark
#   bench-gpu       build + run the full benchmark, writes results/*.csv (GPU machine)
#   examples        build the example programs (CUDA)
#   tools           build helper tools (synthetic test-image generator)
#   readme-results  render results/*.csv into README.md + results/summary.md
#   clean           remove build outputs
#
# The library itself is header-only: the only translation units are under
# tests/, bench/, examples/ and tools/. Source lists below are wildcard-driven
# so this same Makefile works at every commit of the repo's history.

CXX      := g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra -Iinclude -Ithird_party

NVCC := nvcc
# Default fatbinary: Turing (sm_75, the Colab T4) and Ampere (sm_86) SASS, plus
# compute_75 PTX so newer GPUs can JIT. Override with a single arch:
#   make gpu ARCH=sm_75
ARCH ?=
ifeq ($(strip $(ARCH)),)
GENCODE := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=compute_75
else
GENCODE := -arch=$(ARCH)
endif
# -Xcompiler -fopenmp: bench_gpu and test_gpu time/run the OpenMP CPU baselines
# from inside the same binary, so nvcc's host pass needs OpenMP enabled too.
NVCCFLAGS := -O3 --use_fast_math -std=c++17 -Iinclude -Ithird_party -Xcompiler -Wall -Xcompiler -fopenmp $(GENCODE)

# Header-only library: every binary depends on every header.
HDRS := $(shell find include -type f \( -name '*.hpp' -o -name '*.cuh' \) 2>/dev/null) \
        $(wildcard third_party/*.h)

NVCC_PATH := $(shell command -v $(NVCC) 2>/dev/null)

# ---------------------------------------------------------------- CPU section

CPU_BINS :=
ifneq ($(wildcard tests/test_cpu.cpp),)
CPU_BINS += bin/test_cpu
endif
ifneq ($(wildcard bench/bench_cpu.cpp),)
CPU_BINS += bin/bench_cpu
endif
ifneq ($(wildcard tools/make_test_image.cpp),)
CPU_BINS += bin/make_test_image
endif

.PHONY: all cpu gpu test-cpu test-gpu bench-cpu bench-gpu examples tools readme-results clean

all: cpu gpu

cpu: $(CPU_BINS)
ifeq ($(strip $(CPU_BINS)),)
	@echo "cpu: no CPU sources present yet — nothing to build"
else
	@echo "cpu: built $(CPU_BINS)"
endif

test-cpu: cpu
ifneq ($(wildcard tests/test_cpu.cpp),)
	./bin/test_cpu
else
	@echo "test-cpu: tests/test_cpu.cpp not present yet — nothing to run"
endif

bench-cpu: cpu
ifneq ($(wildcard bench/bench_cpu.cpp),)
	./bin/bench_cpu
else
	@echo "bench-cpu: bench/bench_cpu.cpp not present yet — nothing to run"
endif

tools: cpu

bin/test_cpu: tests/test_cpu.cpp $(HDRS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/bench_cpu: bench/bench_cpu.cpp $(HDRS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/make_test_image: tools/make_test_image.cpp $(HDRS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

# ---------------------------------------------------------------- GPU section

ifeq ($(NVCC_PATH),)

gpu test-gpu bench-gpu examples:
	@echo "$@: nvcc not found — GPU targets skipped"

else

GPU_BINS :=
ifneq ($(wildcard tests/test_gpu.cu),)
GPU_BINS += bin/test_gpu
endif
ifneq ($(wildcard bench/bench_gpu.cu),)
GPU_BINS += bin/bench_gpu
endif

gpu: $(GPU_BINS)
ifeq ($(strip $(GPU_BINS)),)
	@echo "gpu: no CUDA sources present yet — nothing to build"
else
	@echo "gpu: built $(GPU_BINS)"
endif

test-gpu: bin/test_gpu
	./bin/test_gpu

bench-gpu: bin/bench_gpu
	./bin/bench_gpu

examples: $(if $(wildcard examples/pipeline_demo.cu),bin/pipeline_demo,)
ifeq ($(wildcard examples/pipeline_demo.cu),)
	@echo "examples: examples/pipeline_demo.cu not present yet — nothing to build"
endif

bin/test_gpu: tests/test_gpu.cu $(HDRS) | bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/bench_gpu: bench/bench_gpu.cu $(HDRS) | bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/pipeline_demo: examples/pipeline_demo.cu $(HDRS) | bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

endif

# ---------------------------------------------------------------- misc

readme-results:
	python3 scripts/render_results.py

bin:
	@mkdir -p bin

clean:
	rm -rf bin out_*.png test_image.png test_roundtrip.png
