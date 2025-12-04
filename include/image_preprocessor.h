#pragma once

#include "gpu_context.h"

#include <opencv2/core.hpp>

// GPU image preprocessor:
//  - Upload pinned host frame to device
//  - Convert to grayscale
//  - (Optionally) undistort using camera intrinsics
//  - (Optionally) decimate to a lower working resolution
//
// This matches the requirements document's "Raw -> Gray, Undistortion + decimation"
// stage. The implementation here is a simplified, but fully functional, CUDA
// version suitable as a starting point on Jetson Orin NX.

struct CameraIntrinsics {
    float fx{0.f}, fy{0.f}, cx{0.f}, cy{0.f};
    float k1{0.f}, k2{0.f}, p1{0.f}, p2{0.f}, k3{0.f};
};

// Per-frame timing breakdown for the image preprocessor (milliseconds).
struct PreprocessTimings {
    float memcpy_host_ms{0.f};  // host memcpy into pinned buffer
    float h2d_ms{0.f};          // host-to-device copy
    float bgr2gray_ms{0.f};     // BGR->Gray kernel
    float undistort_ms{0.f};    // undistortion kernel (0 if disabled)
    float decimate_ms{0.f};     // decimation kernel (0 if disabled)
    float total_ms{0.f};        // total GPU-side time (approx)
};

class ImagePreprocessor {
public:
    ImagePreprocessor(GpuContext& ctx,
                      int width,
                      int height,
                      int decimation = 1,
                      const CameraIntrinsics* intr = nullptr);

    // Upload a raw frame (BGR8 or GRAY8 cv::Mat) into pinned memory and
    // preprocess it on the GPU. Returns a device pointer to a GRAY8 image
    // of size (workingWidth x workingHeight).
    // Optionally fills 'timings' with per-stage timing information.
    unsigned char* preprocess(const cv::Mat& frame,
                              PreprocessTimings* timings = nullptr);

    int inputWidth() const { return in_width_; }
    int inputHeight() const { return in_height_; }
    int workingWidth() const { return work_width_; }
    int workingHeight() const { return work_height_; }

private:
    GpuContext& ctx_;
    int in_width_;
    int in_height_;
    int work_width_;
    int work_height_;
    int decimation_;
    bool use_undistort_;

    // Host pinned buffer for the incoming frame (BGR or Gray)
    unsigned char* host_pinned_{nullptr};
    size_t host_pitch_{0};

    // Device buffers
    unsigned char* d_raw_{nullptr};
    unsigned char* d_gray_{nullptr};
    unsigned char* d_undistorted_{nullptr};  // Undistorted gray image

    CameraIntrinsics intr_{};

    void allocateBuffers();
};


