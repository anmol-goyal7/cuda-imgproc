#pragma once
// CUDA_CHECK — turn silent CUDA failures into loud C++ exceptions.
//
// Every CUDA runtime call returns a cudaError_t, and almost everybody's first
// GPU bug is ignoring one. Worse, kernel launches are asynchronous and return
// void: a launch that dies (bad grid geometry, illegal address, ...) parks its
// error code inside the runtime, where it silently poisons the *next* runtime
// call. The discipline this library follows everywhere:
//
//     CUDA_CHECK(cudaMalloc(&p, bytes));          // check every runtime call
//     my_kernel<<<grid, block>>>(...);            // launch (returns void)
//     CUDA_CHECK(cudaGetLastError());             // check the launch itself
//
// cudaGetLastError() also *clears* the sticky error, so failures are reported
// at the line that caused them, not three calls downstream.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cig {
namespace detail {

// Out-of-line so the macro below stays a single expression statement.
inline void cuda_check_impl(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + " — " + expr + " failed: " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace detail
}  // namespace cig

// do/while(0) makes the macro behave like a single statement inside
// unbraced if/else — the standard macro-hygiene idiom.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        ::cig::detail::cuda_check_impl((call), #call, __FILE__, __LINE__); \
    } while (0)
