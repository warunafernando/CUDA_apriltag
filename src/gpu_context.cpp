#include "gpu_context.h"

#include <algorithm>
#include <stdexcept>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

GpuContext::GpuContext(int device_id) : device_id_(device_id) {
    checkCuda(cudaSetDevice(device_id_), "cudaSetDevice failed");
    checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags failed");
}

GpuContext::~GpuContext() {
    for (void* ptr : allocations_) {
        cudaFree(ptr);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void* GpuContext::allocDevice(size_t bytes) {
    void* ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc failed");
    allocations_.push_back(ptr);
    return ptr;
}

void GpuContext::freeDevice(void* ptr) {
    if (!ptr) return;
    checkCuda(cudaFree(ptr), "cudaFree failed");
    allocations_.erase(
        std::remove(allocations_.begin(), allocations_.end(), ptr),
        allocations_.end());
}


