#pragma once
// GpuBuffer<T> — move-only RAII ownership of a device allocation, modeled on
// std::unique_ptr.
//
// Raw cudaMalloc/cudaFree pairs are how GPU code leaks: any early return or
// exception between the two skips the free, and device memory — unlike host
// memory — is not reclaimed until the process (or CUDA context) dies. Tying
// the allocation's lifetime to a C++ object scope removes that whole bug
// class.
//
// Why move-only? Copying a GpuBuffer would either have to alias the pointer
// (double free on destruction) or implicitly run a device-to-device copy
// (hiding a potentially large memory transfer behind an innocent '='). Both
// are wrong defaults, so copying is deleted and ownership *transfers*, exactly
// like std::unique_ptr.

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "cuda_check.cuh"

namespace cig {

template <typename T>
class GpuBuffer {
public:
    GpuBuffer() = default;

    // Allocates n elements of device memory. cudaMalloc hands back *device*
    // addresses: dereferencing get() on the host is undefined behavior — the
    // pointer only has meaning inside kernels and cudaMemcpy calls.
    explicit GpuBuffer(std::size_t n) : size_(n) {
        if (n != 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        }
    }

    // Destructor must be noexcept (throwing during stack unwinding calls
    // std::terminate), so the cudaFree result is deliberately ignored. If the
    // context is already torn down at exit, cudaFree fails harmlessly here.
    ~GpuBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Moves steal the pointer and leave the source empty, so exactly one
    // object ever owns (and frees) the allocation.
    GpuBuffer(GpuBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }

    // Synchronous host<->device transfers over PCIe. These block until the
    // copy completes, which is exactly what the benchmarks exploit to time
    // "wall clock including transfers" separately from kernel-only time.
    // Both bounds-check against the allocation: an oversized cudaMemcpy would
    // silently corrupt whatever the driver placed after this buffer.
    void copy_from_host(const T* src, std::size_t n) {
        if (n > size_) {
            throw std::out_of_range("GpuBuffer::copy_from_host: n exceeds allocation");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* dst, std::size_t n) const {
        if (n > size_) {
            throw std::out_of_range("GpuBuffer::copy_to_host: n exceeds allocation");
        }
        CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace cig
