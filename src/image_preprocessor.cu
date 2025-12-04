#include "image_preprocessor.h"

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Simple BGR -> Gray conversion
__global__ void bgrToGrayKernel(const unsigned char* bgr, int bgr_stride,
                                unsigned char* gray, int gray_stride,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char* src = bgr + y * bgr_stride + 3 * x;
    unsigned char* dst = gray + y * gray_stride + x;

    float b = static_cast<float>(src[0]);
    float g = static_cast<float>(src[1]);
    float r = static_cast<float>(src[2]);

    *dst = static_cast<unsigned char>(0.114f * b + 0.587f * g + 0.299f * r);
}

// Simple decimation by integer factor (nearest neighbour)
__global__ void decimateKernel(const unsigned char* src, int src_stride,
                               unsigned char* dst, int dst_stride,
                               int in_w, int in_h, int factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = in_w / factor;
    int out_h = in_h / factor;
    if (x >= out_w || y >= out_h) return;

    int sx = x * factor;
    int sy = y * factor;

    const unsigned char* s = src + sy * src_stride + sx;
    unsigned char* d = dst + y * dst_stride + x;
    *d = *s;
}

// Undistortion kernel using Brown-Conrady distortion model
// Maps undistorted coordinates to distorted coordinates and samples
__global__ void undistortKernel(const unsigned char* src, int src_stride,
                                unsigned char* dst, int dst_stride,
                                int width, int height,
                                float fx, float fy, float cx, float cy,
                                float k1, float k2, float p1, float p2, float k3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Normalized coordinates (undistorted)
    float xu = (static_cast<float>(x) - cx) / fx;
    float yu = (static_cast<float>(y) - cy) / fy;

    // Apply inverse distortion model to get distorted coordinates
    float r2 = xu * xu + yu * yu;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    // Radial distortion
    float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    
    // Tangential distortion
    float xd = xu * radial + 2.0f * p1 * xu * yu + p2 * (r2 + 2.0f * xu * xu);
    float yd = yu * radial + p1 * (r2 + 2.0f * yu * yu) + 2.0f * p2 * xu * yu;

    // Convert back to pixel coordinates (distorted)
    float x_distorted = xd * fx + cx;
    float y_distorted = yd * fy + cy;

    // Bilinear interpolation
    int x0 = static_cast<int>(floorf(x_distorted));
    int y0 = static_cast<int>(floorf(y_distorted));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Check bounds
    if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) {
        // Out of bounds - use nearest neighbor or zero
        dst[y * dst_stride + x] = 0;
        return;
    }

    // Fractional parts
    float fx_frac = x_distorted - static_cast<float>(x0);
    float fy_frac = y_distorted - static_cast<float>(y0);

    // Sample four neighbors
    unsigned char val00 = src[y0 * src_stride + x0];
    unsigned char val01 = src[y0 * src_stride + x1];
    unsigned char val10 = src[y1 * src_stride + x0];
    unsigned char val11 = src[y1 * src_stride + x1];

    // Bilinear interpolation
    float val = (1.0f - fx_frac) * (1.0f - fy_frac) * static_cast<float>(val00) +
                fx_frac * (1.0f - fy_frac) * static_cast<float>(val01) +
                (1.0f - fx_frac) * fy_frac * static_cast<float>(val10) +
                fx_frac * fy_frac * static_cast<float>(val11);

    dst[y * dst_stride + x] = static_cast<unsigned char>(val + 0.5f);
}

}  // namespace

ImagePreprocessor::ImagePreprocessor(GpuContext& ctx,
                                     int width,
                                     int height,
                                     int decimation,
                                     const CameraIntrinsics* intr)
    : ctx_(ctx),
      in_width_(width),
      in_height_(height),
      decimation_(decimation <= 0 ? 1 : decimation),
      use_undistort_(intr != nullptr && 
                     (intr->k1 != 0.0f || intr->k2 != 0.0f || intr->p1 != 0.0f || 
                      intr->p2 != 0.0f || intr->k3 != 0.0f)) {
    if (intr) {
        intr_ = *intr;
    }
    work_width_ = in_width_ / decimation_;
    work_height_ = in_height_ / decimation_;
    allocateBuffers();
}

void ImagePreprocessor::allocateBuffers() {
    // Host pinned buffer for raw frame
    size_t bytes = static_cast<size_t>(in_width_) * in_height_ * 3;  // BGR
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&host_pinned_),
                            bytes, cudaHostAllocDefault),
              "cudaHostAlloc failed");

    // Device raw (BGR) and gray (full res)
    d_raw_ = static_cast<unsigned char*>(
        ctx_.allocDevice(bytes));
    d_gray_ = static_cast<unsigned char*>(
        ctx_.allocDevice(static_cast<size_t>(in_width_) * in_height_));
    
    // If undistortion is enabled, allocate undistorted buffer
    if (use_undistort_) {
        d_undistorted_ = static_cast<unsigned char*>(
            ctx_.allocDevice(static_cast<size_t>(in_width_) * in_height_));
    } else {
        d_undistorted_ = nullptr;
    }
}

unsigned char* ImagePreprocessor::preprocess(const cv::Mat& frame,
                                             PreprocessTimings* timings) {
    if (frame.cols != in_width_ || frame.rows != in_height_) {
        throw std::runtime_error("Input frame size mismatch");
    }

    cv::Mat bgr;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = frame;
    }

    double t0 = cv::getTickCount();

    // Copy to pinned host buffer
    std::memcpy(host_pinned_, bgr.data, static_cast<size_t>(in_width_) * in_height_ * 3);
    
    double t1 = cv::getTickCount();
    if (timings) {
        timings->memcpy_host_ms = static_cast<float>((t1 - t0) * 1000.0 / cv::getTickFrequency());
    }

    // Async H2D copy
    cudaEvent_t ev_h2d_start, ev_h2d_end;
    cudaEventCreate(&ev_h2d_start);
    cudaEventCreate(&ev_h2d_end);
    cudaEventRecord(ev_h2d_start, ctx_.stream());
    
    checkCuda(cudaMemcpyAsync(d_raw_, host_pinned_,
                               static_cast<size_t>(in_width_) * in_height_ * 3,
                               cudaMemcpyHostToDevice, ctx_.stream()),
              "cudaMemcpyAsync H2D failed");

    cudaEventRecord(ev_h2d_end, ctx_.stream());
    cudaEventSynchronize(ev_h2d_end);
    
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_end);
    if (timings) {
        timings->h2d_ms = h2d_ms;
    }

    // BGR -> Gray
    dim3 block(16, 16);
    dim3 grid((in_width_ + block.x - 1) / block.x,
              (in_height_ + block.y - 1) / block.y);

    cudaEvent_t ev_bgr2gray_start, ev_bgr2gray_end;
    cudaEventCreate(&ev_bgr2gray_start);
    cudaEventCreate(&ev_bgr2gray_end);
    cudaEventRecord(ev_bgr2gray_start, ctx_.stream());

    bgrToGrayKernel<<<grid, block, 0, ctx_.stream()>>>(
        d_raw_, in_width_ * 3,
        d_gray_, in_width_,
        in_width_, in_height_);

    cudaEventRecord(ev_bgr2gray_end, ctx_.stream());
    cudaEventSynchronize(ev_bgr2gray_end);
    
    float bgr2gray_ms = 0.0f;
    cudaEventElapsedTime(&bgr2gray_ms, ev_bgr2gray_start, ev_bgr2gray_end);
    if (timings) {
        timings->bgr2gray_ms = bgr2gray_ms;
    }

    // Undistortion (if enabled)
    unsigned char* gray_for_decimate = d_gray_;
    cudaEvent_t ev_undistort_start = nullptr, ev_undistort_end = nullptr;
    if (use_undistort_) {
        cudaEventCreate(&ev_undistort_start);
        cudaEventCreate(&ev_undistort_end);
        cudaEventRecord(ev_undistort_start, ctx_.stream());

        undistortKernel<<<grid, block, 0, ctx_.stream()>>>(
            d_gray_, in_width_,
            d_undistorted_, in_width_,
            in_width_, in_height_,
            intr_.fx, intr_.fy, intr_.cx, intr_.cy,
            intr_.k1, intr_.k2, intr_.p1, intr_.p2, intr_.k3);

        cudaEventRecord(ev_undistort_end, ctx_.stream());
        cudaEventSynchronize(ev_undistort_end);
        
        float undistort_ms = 0.0f;
        cudaEventElapsedTime(&undistort_ms, ev_undistort_start, ev_undistort_end);
        if (timings) {
            timings->undistort_ms = undistort_ms;
        }
        
        gray_for_decimate = d_undistorted_;
    }

    // Decimation
    int out_w = work_width_;
    int out_h = work_height_;
    dim3 block_decim(16, 16);
    dim3 grid_decim((out_w + block_decim.x - 1) / block_decim.x,
                    (out_h + block_decim.y - 1) / block_decim.y);

    cudaEvent_t ev_decim_start, ev_decim_end;
    cudaEventCreate(&ev_decim_start);
    cudaEventCreate(&ev_decim_end);
    cudaEventRecord(ev_decim_start, ctx_.stream());

    // Allocate decimated output buffer if needed
    static unsigned char* d_decimated_ = nullptr;
    static int last_decim_w = 0, last_decim_h = 0;
    if (d_decimated_ == nullptr || last_decim_w != out_w || last_decim_h != out_h) {
        if (d_decimated_ != nullptr) {
            ctx_.freeDevice(d_decimated_);
        }
        d_decimated_ = static_cast<unsigned char*>(
            ctx_.allocDevice(static_cast<size_t>(out_w) * out_h));
        last_decim_w = out_w;
        last_decim_h = out_h;
    }

    decimateKernel<<<grid_decim, block_decim, 0, ctx_.stream()>>>(
        gray_for_decimate, in_width_,
        d_decimated_, out_w,
        in_width_, in_height_, decimation_);

    cudaEventRecord(ev_decim_end, ctx_.stream());
    cudaEventSynchronize(ev_decim_end);
    
    float decim_ms = 0.0f;
    cudaEventElapsedTime(&decim_ms, ev_decim_start, ev_decim_end);
    if (timings) {
        timings->decimate_ms = decim_ms;
    }

    // Clean up events
    cudaEventDestroy(ev_h2d_start);
    cudaEventDestroy(ev_h2d_end);
    cudaEventDestroy(ev_bgr2gray_start);
    cudaEventDestroy(ev_bgr2gray_end);
    
    if (use_undistort_ && ev_undistort_start != nullptr && ev_undistort_end != nullptr) {
        cudaEventDestroy(ev_undistort_start);
        cudaEventDestroy(ev_undistort_end);
    }
    
    cudaEventDestroy(ev_decim_start);
    cudaEventDestroy(ev_decim_end);

    if (timings) {
        timings->total_ms = h2d_ms + bgr2gray_ms + (use_undistort_ ? timings->undistort_ms : 0.0f) + decim_ms;
    }

    return d_decimated_;
}
