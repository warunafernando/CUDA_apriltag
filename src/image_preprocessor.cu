#include "image_preprocessor.h"

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cstring>

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
      use_undistort_(intr != nullptr) {
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
}

unsigned char* ImagePreprocessor::preprocess(const cv::Mat& frame,
                                             PreprocessTimings* timings) {
    if (frame.cols != in_width_ || frame.rows != in_height_) {
        throw std::runtime_error("Input frame size mismatch");
    }

    cv::Mat bgr;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 3) {
        bgr = frame;
    } else {
        throw std::runtime_error("Unsupported input format");
    }

    size_t bytes = static_cast<size_t>(in_width_) * in_height_ * 3;

    auto t0 = cv::getTickCount();
    std::memcpy(host_pinned_, bgr.data, bytes);
    auto t1 = cv::getTickCount();

    // Upload to device
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);

    cudaEventRecord(e0, ctx_.stream());
    checkCuda(cudaMemcpyAsync(d_raw_, host_pinned_, bytes,
                              cudaMemcpyHostToDevice, ctx_.stream()),
              "cudaMemcpyAsync (H2D) failed");
    cudaEventRecord(e1, ctx_.stream());

    dim3 block(16, 16);
    dim3 grid((in_width_ + block.x - 1) / block.x,
              (in_height_ + block.y - 1) / block.y);

    bgrToGrayKernel<<<grid, block, 0, ctx_.stream()>>>(
        d_raw_, in_width_ * 3, d_gray_, in_width_,
        in_width_, in_height_);
    cudaEventRecord(e2, ctx_.stream());

    checkCuda(cudaGetLastError(), "bgrToGrayKernel launch failed");

    // For now, undistortion is not implemented; this is a good
    // place to hook in an undistort kernel using intr_.

    unsigned char* out_dev = d_gray_;

    if (decimation_ != 1) {
        // Allocate a decimated buffer lazily
        static unsigned char* d_decimated = nullptr;
        static int dec_w = 0, dec_h = 0;
        if (!d_decimated || dec_w != work_width_ || dec_h != work_height_) {
            if (d_decimated) {
                ctx_.freeDevice(d_decimated);
            }
            d_decimated = static_cast<unsigned char*>(
                ctx_.allocDevice(static_cast<size_t>(work_width_) * work_height_));
            dec_w = work_width_;
            dec_h = work_height_;
        }

        dim3 grid_dec((work_width_ + block.x - 1) / block.x,
                      (work_height_ + block.y - 1) / block.y);

        decimateKernel<<<grid_dec, block, 0, ctx_.stream()>>>(
            d_gray_, in_width_, d_decimated, work_width_,
            in_width_, in_height_, decimation_);
        cudaEventRecord(e3, ctx_.stream());

        checkCuda(cudaGetLastError(), "decimateKernel launch failed");
        out_dev = d_decimated;
    } else {
        cudaEventRecord(e3, ctx_.stream());
    }

    cudaEventSynchronize(e3);

    if (timings) {
        double memcpy_ms = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        float h2d_ms = 0.f, bgr_ms = 0.f, dec_ms = 0.f;
        cudaEventElapsedTime(&h2d_ms, e0, e1);
        cudaEventElapsedTime(&bgr_ms, e1, e2);
        if (decimation_ != 1) {
            cudaEventElapsedTime(&dec_ms, e2, e3);
        }
        timings->memcpy_host_ms = static_cast<float>(memcpy_ms);
        timings->h2d_ms = h2d_ms;
        timings->bgr2gray_ms = bgr_ms;
        timings->decimate_ms = dec_ms;
        timings->total_ms = h2d_ms + bgr_ms + dec_ms;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);

    return out_dev;
}


