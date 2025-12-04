#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

// Simple RAII wrapper around CUDA streams and device buffers.
// This is intentionally generic so it can be reused by the image
// preprocessor and AprilTag detector.

class GpuContext {
public:
    explicit GpuContext(int device_id = 0);
    ~GpuContext();

    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;

    cudaStream_t stream() const { return stream_; }

    // Allocate a device buffer of given size (in bytes).
    void* allocDevice(size_t bytes);

    // Free a device buffer previously returned by allocDevice.
    void freeDevice(void* ptr);

    int deviceId() const { return device_id_; }

private:
    int device_id_{0};
    cudaStream_t stream_{nullptr};
    std::vector<void*> allocations_;
};


