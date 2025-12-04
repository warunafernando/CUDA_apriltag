// Sub-pixel corner refinement using gradient-based method
// Improves corner accuracy for better pose estimation

#include <cuda_runtime.h>
#include <cmath>

namespace {

// Refine corner position using gradient information
// Uses Harris corner response or gradient-based refinement
__device__ float2 refineCornerSubpixel(const unsigned char* gray,
                                       int width,
                                       int height,
                                       int stride,
                                       float2 corner,
                                       int window_size = 5) {
    int cx = __float2int_rn(corner.x);
    int cy = __float2int_rn(corner.y);
    
    // Clamp to valid range
    if (cx < window_size || cx >= width - window_size ||
        cy < window_size || cy >= height - window_size) {
        return corner;
    }
    
    // Compute gradients in window around corner
    float gxx = 0.0f, gyy = 0.0f, gxy = 0.0f;
    float gx_sum = 0.0f, gy_sum = 0.0f;
    
    for (int dy = -window_size; dy <= window_size; ++dy) {
        for (int dx = -window_size; dx <= window_size; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                // Compute gradients using Sobel
                int gx = -gray[(y-1) * stride + (x-1)] - 2*gray[y * stride + (x-1)] - gray[(y+1) * stride + (x-1)] +
                         gray[(y-1) * stride + (x+1)] + 2*gray[y * stride + (x+1)] + gray[(y+1) * stride + (x+1)];
                
                int gy = -gray[(y-1) * stride + (x-1)] - 2*gray[(y-1) * stride + x] - gray[(y-1) * stride + (x+1)] +
                         gray[(y+1) * stride + (x-1)] + 2*gray[(y+1) * stride + x] + gray[(y+1) * stride + (x+1)];
                
                float fx = dx;
                float fy = dy;
                float weight = expf(-(fx*fx + fy*fy) / (2.0f * window_size * window_size));
                
                gxx += gx * gx * weight;
                gyy += gy * gy * weight;
                gxy += gx * gy * weight;
                gx_sum += gx * weight;
                gy_sum += gy * weight;
            }
        }
    }
    
    // Compute corner shift using gradient information
    float det = gxx * gyy - gxy * gxy;
    if (fabsf(det) < 1e-6f) {
        return corner;  // Singular matrix, return original
    }
    
    // Solve for corner offset
    float dx = -(gyy * gx_sum - gxy * gy_sum) / det;
    float dy = -(-gxy * gx_sum + gxx * gy_sum) / det;
    
    // Clamp offset to reasonable range
    dx = fmaxf(-1.0f, fminf(1.0f, dx));
    dy = fmaxf(-1.0f, fminf(1.0f, dy));
    
    return make_float2(corner.x + dx, corner.y + dy);
}

// GPU kernel for sub-pixel corner refinement
__global__ void refineCornersSubpixelKernel(const unsigned char* gray,
                                            int width,
                                            int height,
                                            int stride,
                                            float2* corners,
                                            int num_quads,
                                            bool enable_refinement) {
    int quad_idx = blockIdx.x;
    if (quad_idx >= num_quads) return;
    
    if (!enable_refinement) return;
    
    float2* quad_corners = &corners[quad_idx * 4];
    
    // Refine each corner
    for (int i = 0; i < 4; ++i) {
        quad_corners[i] = refineCornerSubpixel(gray, width, height, stride, 
                                               quad_corners[i], 5);
    }
}

} // namespace

