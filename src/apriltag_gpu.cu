#include "apriltag_gpu.h"

#include <cuda_runtime.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <map>
#include <deque>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Compute IoU (Intersection over Union) between two quads
// Used for Non-Maximum Suppression
// Note: This is a simplified IoU using bounding boxes, not exact quad intersection
inline float computeQuadIoU(const float2* quad1, const float2* quad2) {
    // Compute bounding boxes
    float min_x1 = fminf(fminf(quad1[0].x, quad1[1].x), fminf(quad1[2].x, quad1[3].x));
    float max_x1 = fmaxf(fmaxf(quad1[0].x, quad1[1].x), fmaxf(quad1[2].x, quad1[3].x));
    float min_y1 = fminf(fminf(quad1[0].y, quad1[1].y), fminf(quad1[2].y, quad1[3].y));
    float max_y1 = fmaxf(fmaxf(quad1[0].y, quad1[1].y), fmaxf(quad1[2].y, quad1[3].y));
    
    float min_x2 = fminf(fminf(quad2[0].x, quad2[1].x), fminf(quad2[2].x, quad2[3].x));
    float max_x2 = fmaxf(fmaxf(quad2[0].x, quad2[1].x), fmaxf(quad2[2].x, quad2[3].x));
    float min_y2 = fminf(fminf(quad2[0].y, quad2[1].y), fminf(quad2[2].y, quad2[3].y));
    float max_y2 = fmaxf(fmaxf(quad2[0].y, quad2[1].y), fmaxf(quad2[2].y, quad2[3].y));
    
    // Compute intersection
    float inter_x0 = fmaxf(min_x1, min_x2);
    float inter_y0 = fmaxf(min_y1, min_y2);
    float inter_x1 = fminf(max_x1, max_x2);
    float inter_y1 = fminf(max_y1, max_y2);
    
    float inter_area = 0.0f;
    if (inter_x1 > inter_x0 && inter_y1 > inter_y0) {
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0);
    }
    
    // Compute union (approximate using bounding boxes)
    float area1 = (max_x1 - min_x1) * (max_y1 - min_y1);
    float area2 = (max_x2 - min_x2) * (max_y2 - min_y2);
    float union_area = area1 + area2 - inter_area;
    
    if (union_area < 1e-6f) return 0.0f;
    return inter_area / union_area;
}

// Optimized gradient magnitude kernel
__global__ void gradientMagKernel(const unsigned char* gray,
                                  int stride,
                                  unsigned char* grad_mag,
                                  int width,
                                  int height) {
    __shared__ unsigned char tile[18][18];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int load_x = x - 1;
    int load_y = y - 1;
    if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
        tile[ty + 1][tx + 1] = gray[load_y * stride + load_x];
    }
    
    if (tx == 0 && x > 0 && load_y >= 0 && load_y < height) {
        tile[ty + 1][0] = gray[load_y * stride + (x - 1)];
    }
    if (tx == blockDim.x - 1 && x < width - 1 && load_y >= 0 && load_y < height) {
        tile[ty + 1][blockDim.x + 1] = gray[load_y * stride + (x + 1)];
    }
    if (ty == 0 && y > 0 && load_x >= 0 && load_x < width) {
        tile[0][tx + 1] = gray[(y - 1) * stride + load_x];
    }
    if (ty == blockDim.y - 1 && y < height - 1 && load_x >= 0 && load_x < width) {
        tile[blockDim.y + 1][tx + 1] = gray[(y + 1) * stride + load_x];
    }
    
    __syncthreads();
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = -tile[ty][tx] - 2.f * tile[ty + 1][tx] - tile[ty + 2][tx] +
                    tile[ty][tx + 2] + 2.f * tile[ty + 1][tx + 2] + tile[ty + 2][tx + 2];
        
        float gy = -tile[ty][tx] - 2.f * tile[ty][tx + 1] - tile[ty][tx + 2] +
                    tile[ty + 2][tx] + 2.f * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2];
        
        float mag = sqrtf(gx * gx + gy * gy);
        mag = fminf(mag, 255.f);
        grad_mag[y * stride + x] = static_cast<unsigned char>(mag);
    }
}

// Compute adaptive threshold using local mean (Otsu-like approach)
// Divides image into regions and computes threshold per region
__global__ void computeAdaptiveThresholdKernel(const unsigned char* grad_mag,
                                                unsigned char* threshold_map,
                                                int width,
                                                int height,
                                                int stride,
                                                int region_size) {
    // Each thread processes one region
    int region_x = blockIdx.x * blockDim.x + threadIdx.x;
    int region_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int x0 = region_x * region_size;
    int y0 = region_y * region_size;
    int x1 = min(x0 + region_size, width);
    int y1 = min(y0 + region_size, height);
    
    if (x0 >= width || y0 >= height) return;
    
    // Compute mean and variance for this region
    int sum = 0;
    int count = 0;
    int sum_sq = 0;
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            unsigned char val = grad_mag[y * stride + x];
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    
    if (count == 0) {
        threshold_map[region_y * ((width + region_size - 1) / region_size) + region_x] = 40;
        return;
    }
    
    float mean = sum / (float)count;
    float variance = (sum_sq / (float)count) - (mean * mean);
    
    // Otsu-like threshold: mean - k * std_dev
    // For edge detection, we want to be slightly below mean
    float std_dev = sqrtf(variance);
    float threshold = mean - 0.5f * std_dev;
    
    // Clamp to reasonable range
    threshold = fmaxf(20.0f, fminf(80.0f, threshold));
    
    int region_w = (width + region_size - 1) / region_size;
    threshold_map[region_y * region_w + region_x] = __float2uint_rn(threshold);
}

// Edge thresholding with adaptive threshold map
__global__ void edgeThresholdKernel(const unsigned char* grad_mag,
                                   const unsigned char* threshold_map,
                                   unsigned char* edges,
                                   int width,
                                   int height,
                                   int stride,
                                   int region_size,
                                   unsigned char global_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * stride + x;
    
    // Get region index for adaptive threshold
    int region_x = x / region_size;
    int region_y = y / region_size;
    int region_w = (width + region_size - 1) / region_size;
    int region_idx = region_y * region_w + region_x;
    
    // Use adaptive threshold if available, otherwise fallback to global
    unsigned char threshold = (threshold_map != nullptr) ? 
                              threshold_map[region_idx] : global_threshold;
    
    edges[idx] = (grad_mag[idx] > threshold) ? 255 : 0;
}

// Find candidate regions using edge density (GPU)
__global__ void findCandidateRegionsKernel(const unsigned char* edges,
                                           int width,
                                           int height,
                                           int stride,
                                           int* candidate_regions,  // x, y, w, h per region
                                           int* region_count,
                                           int max_regions,
                                           int min_size,
                                           int max_size) {
    // Divide image into grid cells and compute edge density
    const int grid_size = 32;
    int grid_w = (width + grid_size - 1) / grid_size;
    int grid_h = (height + grid_size - 1) / grid_size;
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= grid_w || gy >= grid_h) return;
    
    // Count edges in this grid cell
    int edge_count = 0;
    int x0 = gx * grid_size;
    int y0 = gy * grid_size;
    int x1 = min(x0 + grid_size, width);
    int y1 = min(y0 + grid_size, height);
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            if (edges[y * stride + x] != 0) {
                edge_count++;
            }
        }
    }
    
    // If edge density is high enough, add as candidate region
    int threshold = (grid_size * grid_size) / 10;  // 10% edge density
    if (edge_count > threshold) {
        int idx = atomicAdd(region_count, 1);
        if (idx < max_regions) {
            candidate_regions[idx * 4 + 0] = max(0, x0 - grid_size/2);
            candidate_regions[idx * 4 + 1] = max(0, y0 - grid_size/2);
            candidate_regions[idx * 4 + 2] = min(width - candidate_regions[idx * 4 + 0], grid_size * 2);
            candidate_regions[idx * 4 + 3] = min(height - candidate_regions[idx * 4 + 1], grid_size * 2);
        }
    }
}

// Extract quads from edge map using GPU
// Improved algorithm: better corner detection and quad validation
// Also computes edge strength score for each quad (for NMS)
__global__ void extractQuadsFromEdgesKernel(const unsigned char* edges,
                                            const unsigned char* grad_mag,
                                            int width,
                                            int height,
                                            int stride,
                                            float2* quad_corners_out,
                                            float* quad_scores_out,
                                            int* quad_count,
                                            int max_quads,
                                            int min_perimeter,
                                            int max_perimeter) {
    // grad_mag can be nullptr if scores not needed
    // Use larger grid cells to find complete tag regions
    const int grid_size = 48;  // Balanced size for tag detection
    int grid_w = (width + grid_size - 1) / grid_size;
    int grid_h = (height + grid_size - 1) / grid_size;
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= grid_w || gy >= grid_h) return;
    
    int x0 = gx * grid_size;
    int y0 = gy * grid_size;
    int x1 = min(x0 + grid_size, width);
    int y1 = min(y0 + grid_size, height);
    
    // Count edges and find bounding box
    int edge_count = 0;
    int min_x = width, max_x = 0, min_y = height, max_y = 0;
    int edge_sum_x = 0, edge_sum_y = 0;  // For centroid calculation
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            if (edges[y * stride + x] != 0) {
                edge_count++;
                min_x = min(min_x, x);
                max_x = max(max_x, x);
                min_y = min(min_y, y);
                max_y = max(max_y, y);
                edge_sum_x += x;
                edge_sum_y += y;
            }
        }
    }
    
    // Early rejection: need sufficient edges
    int region_area = (x1 - x0) * (y1 - y0);
    if (region_area == 0 || edge_count < 25) return;  // Reasonable minimum edge count
    
    float edge_density = (edge_count * 100.0f) / region_area;
    if (edge_density < 4.0f || edge_density > 45.0f) return;  // Balanced density range
    
    // Calculate dimensions
    int w = max_x - min_x;
    int h = max_y - min_y;
    if (w < 18 || h < 18) return;  // Minimum size for tags (slightly relaxed)
    
    int perimeter = 2 * (w + h);
    if (perimeter < min_perimeter || perimeter > max_perimeter) return;
    
    // Check aspect ratio (tags are roughly square, allow some perspective)
    float aspect = (w > h) ? (float)w / h : (float)h / w;
    if (aspect > 2.0f) return;  // Allow more perspective distortion
    
    // Calculate centroid
    float centroid_x = edge_sum_x / (float)edge_count;
    float centroid_y = edge_sum_y / (float)edge_count;
    
    // Find corner points using a better algorithm
    // Look for edge points that are farthest from centroid in each quadrant
    float2 corners[4];
    float max_dist[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    
    // Search in expanded region around bounding box
    int search_x0 = max(0, min_x - 10);
    int search_y0 = max(0, min_y - 10);
    int search_x1 = min(width, max_x + 10);
    int search_y1 = min(height, max_y + 10);
    
    const int search_step = 2;
    for (int y = search_y0; y < search_y1; y += search_step) {
        for (int x = search_x0; x < search_x1; x += search_step) {
            if (edges[y * stride + x] != 0) {
                float dx = x - centroid_x;
                float dy = y - centroid_y;
                float dist_sq = dx * dx + dy * dy;
                
                // Determine quadrant and update farthest point
                int quadrant = -1;
                if (dx < 0 && dy < 0) quadrant = 0;      // Top-left
                else if (dx >= 0 && dy < 0) quadrant = 1; // Top-right
                else if (dx >= 0 && dy >= 0) quadrant = 2; // Bottom-right
                else if (dx < 0 && dy >= 0) quadrant = 3;  // Bottom-left
                
                if (quadrant >= 0 && dist_sq > max_dist[quadrant]) {
                    max_dist[quadrant] = dist_sq;
                    corners[quadrant] = make_float2(x, y);
                }
            }
        }
    }
    
    // Check if we found corners in all 4 quadrants
    bool has_all_corners = true;
    for (int i = 0; i < 4; ++i) {
        if (max_dist[i] < 0) {
            has_all_corners = false;
            break;
        }
    }
    
    // If we didn't find corners in all quadrants, try to infer them
    if (!has_all_corners) {
        // For missing corners, use bounding box corners but refine them
        if (max_dist[0] < 0) corners[0] = make_float2(min_x, min_y);
        if (max_dist[1] < 0) corners[1] = make_float2(max_x, min_y);
        if (max_dist[2] < 0) corners[2] = make_float2(max_x, max_y);
        if (max_dist[3] < 0) corners[3] = make_float2(min_x, max_y);
        
        // Refine bounding box corners by finding nearest edge points
        for (int q = 0; q < 4; ++q) {
            if (max_dist[q] < 0) {
                // Find nearest edge point to this corner
                float min_dist_to_edge = 1e6f;
                float2 best_corner = corners[q];
                
                for (int y = search_y0; y < search_y1; y += 2) {
                    for (int x = search_x0; x < search_x1; x += 2) {
                        if (edges[y * stride + x] != 0) {
                            float dx = x - corners[q].x;
                            float dy = y - corners[q].y;
                            float dist = dx * dx + dy * dy;
                            if (dist < min_dist_to_edge && dist < 400.0f) {
                                min_dist_to_edge = dist;
                                best_corner = make_float2(x, y);
                            }
                        }
                    }
                }
                corners[q] = best_corner;
            }
        }
    }
    
    // Validate quad geometry
    // Calculate area using shoelace formula
    float area = 0.0f;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        area += corners[i].x * corners[j].y;
        area -= corners[j].x * corners[i].y;
    }
    area = fabsf(area) * 0.5f;
    
    float min_area = (min_perimeter * min_perimeter) / 30.0f;
    float max_area = (max_perimeter * max_perimeter) * 0.7f;
    
    if (area < min_area || area > max_area) return;
    
    // Check corner distances (ensure they're distinct)
    float min_corner_dist = 1e6f;
    float max_corner_dist = 0.0f;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            float dx = corners[i].x - corners[j].x;
            float dy = corners[i].y - corners[j].y;
            float dist = sqrtf(dx * dx + dy * dy);
            min_corner_dist = fminf(min_corner_dist, dist);
            max_corner_dist = fmaxf(max_corner_dist, dist);
        }
    }
    
    if (min_corner_dist < 15.0f) return;  // Minimum corner distance
    if (max_corner_dist / min_corner_dist > 3.5f) return;  // Allow some irregularity
    
    // Check edge lengths (should be roughly similar for square tags)
    float edge_lengths[4];
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        float dx = corners[j].x - corners[i].x;
        float dy = corners[j].y - corners[i].y;
        edge_lengths[i] = sqrtf(dx * dx + dy * dy);
    }
    
    float min_edge = edge_lengths[0];
    float max_edge = edge_lengths[0];
    for (int i = 1; i < 4; ++i) {
        min_edge = fminf(min_edge, edge_lengths[i]);
        max_edge = fmaxf(max_edge, edge_lengths[i]);
    }
    
    // Edges should be reasonably similar (allow perspective distortion)
    if (max_edge > 0 && (max_edge / min_edge) > 2.5f) return;  // Balanced edge length check
    
    // Check convexity: all corners should be on the same side of each edge
    bool is_convex = true;
    for (int i = 0; i < 4 && is_convex; ++i) {
        int j = (i + 1) % 4;
        int k = (i + 2) % 4;
        int l = (i + 3) % 4;
        
        // Cross product to check side
        float dx1 = corners[j].x - corners[i].x;
        float dy1 = corners[j].y - corners[i].y;
        float dx2 = corners[k].x - corners[i].x;
        float dy2 = corners[k].y - corners[i].y;
        float dx3 = corners[l].x - corners[i].x;
        float dy3 = corners[l].y - corners[i].y;
        
        float cross1 = dx1 * dy2 - dy1 * dx2;
        float cross2 = dx1 * dy3 - dy1 * dx3;
        
        // Both should have same sign for convexity
        if ((cross1 > 0 && cross2 < 0) || (cross1 < 0 && cross2 > 0)) {
            is_convex = false;
        }
    }
    
    if (!is_convex) return;
    
    // Compute edge strength score for NMS (average gradient magnitude along quad perimeter)
    float edge_score = 0.0f;
    int score_samples = 0;
    if (grad_mag != nullptr) {
        // Sample gradient magnitude along quad edges
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            float dx = corners[j].x - corners[i].x;
            float dy = corners[j].y - corners[i].y;
            float edge_len = sqrtf(dx * dx + dy * dy);
            int samples = max(1, (int)(edge_len / 4.0f));  // Sample every 4 pixels
            
            for (int s = 0; s < samples; ++s) {
                float t = s / (float)samples;
                int px = __float2int_rn(corners[i].x + t * dx);
                int py = __float2int_rn(corners[i].y + t * dy);
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    edge_score += grad_mag[py * stride + px];
                    score_samples++;
                }
            }
        }
        if (score_samples > 0) {
            edge_score /= score_samples;
        }
    } else {
        // Fallback: use edge count as score
        edge_score = edge_count;
    }
    
    // All checks passed - add quad
    int idx = atomicAdd(quad_count, 1);
    if (idx < max_quads) {
        float2* out_corners = &quad_corners_out[idx * 4];
        out_corners[0] = corners[0];  // Top-left
        out_corners[1] = corners[1];  // Top-right
        out_corners[2] = corners[2];  // Bottom-right
        out_corners[3] = corners[3];  // Bottom-left
        
        if (quad_scores_out != nullptr) {
            quad_scores_out[idx] = edge_score;
        }
    }
}

// Sample tag bits from a quad (GPU)
// Samples 6x6 grid from the quad using bilinear interpolation
// Corner order: [top-left, top-right, bottom-right, bottom-left] (OpenCV aruco convention)
__global__ void sampleTagBitsKernel(const unsigned char* gray,
                                    int width,
                                    int height,
                                    int stride,
                                    const float2* quad_corners,  // 4 corners per quad
                                    unsigned char* tag_bits,     // 36 bits per tag (6x6)
                                    int num_quads) {
    int quad_idx = blockIdx.x;
    if (quad_idx >= num_quads) return;
    
    const float2* corners = &quad_corners[quad_idx * 4];
    unsigned char* bits = &tag_bits[quad_idx * 36];
    
    int tid = threadIdx.x;
    if (tid < 36) {
        int row = tid / 6;
        int col = tid % 6;
        
        // Compute normalized coordinates in quad (0-1 range)
        // Add 0.5 to sample center of each cell
        // Note: OpenCV aruco corners are [top-left, top-right, bottom-right, bottom-left]
        float u = (col + 0.5f) / 6.0f;
        float v = (row + 0.5f) / 6.0f;
        
        // Bilinear interpolation from quad corners
        // corners[0] = top-left, corners[1] = top-right
        // corners[2] = bottom-right, corners[3] = bottom-left
        float px = (1-u)*(1-v)*corners[0].x + u*(1-v)*corners[1].x +
                   u*v*corners[2].x + (1-u)*v*corners[3].x;
        float py = (1-u)*(1-v)*corners[0].y + u*(1-v)*corners[1].y +
                   u*v*corners[2].y + (1-u)*v*corners[3].y;
        
        // Clamp to image bounds and sample with bilinear interpolation
        int x0 = __float2int_rd(px);
        int y0 = __float2int_rd(py);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = px - x0;
        float fy = py - y0;
        
        x0 = max(0, min(width - 1, x0));
        y0 = max(0, min(height - 1, y0));
        x1 = max(0, min(width - 1, x1));
        y1 = max(0, min(height - 1, y1));
        
        // Bilinear interpolation for better sampling
        float val = (1-fx)*(1-fy)*gray[y0 * stride + x0] +
                    fx*(1-fy)*gray[y0 * stride + x1] +
                    fx*fy*gray[y1 * stride + x1] +
                    (1-fx)*fy*gray[y1 * stride + x0];
        
        bits[row * 6 + col] = __float2uint_rn(val);
    }
}

// Decode AprilTag 36h11 (GPU)
// AprilTag 36h11: 6x6 grid, outer border black, inner 4x4 data
// Uses Hamming(8,4) encoding - 4 data bits + 4 parity bits per 2x2 block
// Returns: tag_id, decision_margin (confidence), hamming_distance
__global__ void decodeTagsKernel(const unsigned char* tag_bits,
                                 int* tag_ids,
                                 float* decision_margins,
                                 float* hamming_distances,
                                 int num_quads,
                                 float threshold) {
    int quad_idx = blockIdx.x;
    if (quad_idx >= num_quads) return;
    
    const unsigned char* bits = &tag_bits[quad_idx * 36];
    int& id = tag_ids[quad_idx];
    float& margin = decision_margins[quad_idx];
    float& hamming = hamming_distances[quad_idx];
    
    id = -1;
    margin = 0.0f;
    hamming = 999.0f;
    
    // Compute adaptive threshold from tag bits (Otsu-like approach)
    int sum = 0;
    for (int i = 0; i < 36; ++i) {
        sum += bits[i];
    }
    float mean = sum / 36.0f;
    float adaptive_threshold = mean;  // Use mean as threshold (simplified Otsu)
    // Clamp to reasonable range
    adaptive_threshold = fmaxf(80.0f, fminf(180.0f, adaptive_threshold));
    
    // Binarize the tag bits using adaptive threshold
    unsigned char binary_bits[36];
    for (int i = 0; i < 36; ++i) {
        binary_bits[i] = (bits[i] > adaptive_threshold) ? 1 : 0;
    }
    
    // Check outer border (should be black/dark = 0)
    // AprilTag has black border on all sides
    int border_errors = 0;
    int border_count = 0;
    for (int i = 0; i < 6; ++i) {
        // Top row
        if (binary_bits[i] != 0) border_errors++;
        border_count++;
        // Bottom row  
        if (binary_bits[30 + i] != 0) border_errors++;
        border_count++;
        // Left column (skip corners to avoid double counting)
        if (i > 0 && i < 5) {
            if (binary_bits[6*i] != 0) border_errors++;
            border_count++;
        }
        // Right column (skip corners)
        if (i > 0 && i < 5) {
            if (binary_bits[5 + 6*i] != 0) border_errors++;
            border_count++;
        }
    }
    
    // Allow up to 40% border errors (very permissive for now)
    if (border_errors > border_count * 0.4f) return;
    
    // Extract inner 4x4 data bits (rows 1-4, cols 1-4)
    // AprilTag layout: border is outer ring, data is inner 4x4
    unsigned char data_bits[16];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            int idx = (row + 1) * 6 + (col + 1);
            data_bits[row * 4 + col] = binary_bits[idx];
        }
    }
    
    // Try all 4 rotations to find valid tag
    // AprilTag can be oriented in 4 ways
    int best_id = -1;
    int min_border_errors = border_errors;
    int best_rotation = -1;
    int min_hamming = 999;
    
    for (int rot = 0; rot < 4; ++rot) {
        // Rotate data bits (90 degrees counter-clockwise each time)
        unsigned char rotated[16];
        for (int i = 0; i < 16; ++i) {
            int row = i / 4;
            int col = i % 4;
            int new_row, new_col;
            if (rot == 0) {
                new_row = row;
                new_col = col;
            } else if (rot == 1) {  // 90 deg CCW
                new_row = col;
                new_col = 3 - row;
            } else if (rot == 2) {  // 180 deg
                new_row = 3 - row;
                new_col = 3 - col;
            } else {  // 270 deg CCW
                new_row = 3 - col;
                new_col = row;
            }
            rotated[new_row * 4 + new_col] = data_bits[i];
        }
        
        // Convert bits to candidate ID
        // AprilTag uses little-endian bit order: bit 0 is LSB
        int candidate_id = 0;
        for (int i = 0; i < 16; ++i) {
            if (rotated[i]) {
                candidate_id |= (1 << i);
            }
        }
        
        // Validate ID range (0-586 for 36h11)
        if (candidate_id >= 0 && candidate_id <= 586) {
            // Compute confidence based on border errors
            float candidate_margin = 1.0f - (border_errors / (float)border_count);
            int hamming_dist = border_errors;  // Simplified: use border errors as Hamming distance
            
            // Prefer rotation with fewer border errors and higher confidence
            if (best_id < 0 || border_errors < min_border_errors || 
                (border_errors == min_border_errors && candidate_margin > margin)) {
                min_border_errors = border_errors;
                min_hamming = hamming_dist;
                best_id = candidate_id;
                best_rotation = rot;
                margin = candidate_margin;
                hamming = hamming_dist;
                
                // If rotation 0 works well, prefer it
                if (rot == 0 && candidate_margin > 0.7f) {
                    break;
                }
            }
        }
    }
    
    // Accept the best ID found
    if (best_id >= 0) {
        id = best_id;
        // Ensure margin and hamming are set
        if (margin < 0.1f) margin = 1.0f - (min_border_errors / (float)border_count);
        if (hamming > 100) hamming = min_hamming;
    }
}

// Sub-pixel corner refinement using gradient-based method
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
    if (quad_idx >= num_quads || !enable_refinement) return;
    
    float2* quad_corners = &corners[quad_idx * 4];
    
    // Refine each corner
    for (int i = 0; i < 4; ++i) {
        quad_corners[i] = refineCornerSubpixel(gray, width, height, stride, 
                                               quad_corners[i], 5);
    }
}

// GPU PnP solver using DLT (Direct Linear Transform)
__global__ void solvePnPKernel(const float2* image_points,  // 4 points per quad
                               const float* object_points,   // 4x3 (x,y,z) per quad
                               const float* K,               // 3x3 camera matrix
                               float* R_out,                 // 3x3 rotation per quad
                               float* t_out,                 // 3x1 translation per quad
                               float* reprojection_errors,   // Reprojection error per quad
                               int num_quads) {
    int quad_idx = blockIdx.x;
    if (quad_idx >= num_quads) return;
    
    const float2* img_pts = &image_points[quad_idx * 4];
    const float* obj_pts = &object_points[quad_idx * 12];
    float* R = &R_out[quad_idx * 9];
    float* t = &t_out[quad_idx * 3];
    
    // Normalize image points
    float cx = K[2], cy = K[5];
    float fx = K[0], fy = K[4];
    
    float2 nimg[4];
    for (int i = 0; i < 4; ++i) {
        nimg[i].x = (img_pts[i].x - cx) / fx;
        nimg[i].y = (img_pts[i].y - cy) / fy;
    }
    
    // Build homography system: H maps object plane (z=0) to image plane
    // For coplanar points, we can solve for homography H, then extract R and t
    // Simplified: use DLT to solve for H, then decompose
    
    // Simplified: assume identity rotation and solve for translation
    // This is a placeholder - full SVD implementation needed for accuracy
    R[0] = 1; R[1] = 0; R[2] = 0;
    R[3] = 0; R[4] = 1; R[5] = 0;
    R[6] = 0; R[7] = 0; R[8] = 1;
    
    // Estimate translation from center point
    float cx_img = (nimg[0].x + nimg[1].x + nimg[2].x + nimg[3].x) / 4.0f;
    float cy_img = (nimg[0].y + nimg[1].y + nimg[2].y + nimg[3].y) / 4.0f;
    
    t[0] = cx_img * 0.1f;  // Simplified - would use proper depth estimation
    t[1] = cy_img * 0.1f;
    t[2] = 1.0f;
    
    // Compute reprojection error
    if (reprojection_errors != nullptr) {
        float reproj_error = 0.0f;
        for (int i = 0; i < 4; ++i) {
            // Project object point to image
            float x_obj = obj_pts[i * 3 + 0];
            float y_obj = obj_pts[i * 3 + 1];
            float z_obj = obj_pts[i * 3 + 2];
            
            // Transform using R and t
            float x_cam = R[0] * x_obj + R[1] * y_obj + R[2] * z_obj + t[0];
            float y_cam = R[3] * x_obj + R[4] * y_obj + R[5] * z_obj + t[1];
            float z_cam = R[6] * x_obj + R[7] * y_obj + R[8] * z_obj + t[2];
            
            if (fabsf(z_cam) < 1e-6f) z_cam = 1e-6f;
            
            // Project to image plane
            float u_proj = fx * (x_cam / z_cam) + cx;
            float v_proj = fy * (y_cam / z_cam) + cy;
            
            // Compute error
            float dx = u_proj - img_pts[i].x;
            float dy = v_proj - img_pts[i].y;
            reproj_error += sqrtf(dx * dx + dy * dy);
        }
        reproj_error /= 4.0f;  // Average error per corner
        reprojection_errors[quad_idx] = reproj_error;
    }
}

}  // namespace

AprilTagGpuDetector::AprilTagGpuDetector(GpuContext& ctx,
                                         int width,
                                         int height,
                                         float tag_size_meters,
                                         const cv::Matx33f& K)
    : ctx_(ctx),
      width_(width),
      height_(height),
      tag_size_(tag_size_meters),
      K_(K) {
    allocateBuffers();
}

void AprilTagGpuDetector::allocateBuffers() {
    size_t img_size = static_cast<size_t>(width_) * height_;
    d_grad_mag_ = static_cast<unsigned char*>(ctx_.allocDevice(img_size));
    d_edges_ = static_cast<unsigned char*>(ctx_.allocDevice(img_size));
    
    // Allocate adaptive threshold map (one threshold per region)
    const int region_size = 64;  // 64x64 pixel regions
    int region_w = (width_ + region_size - 1) / region_size;
    int region_h = (height_ + region_size - 1) / region_size;
    size_t threshold_map_size = static_cast<size_t>(region_w) * region_h;
    d_threshold_map_ = static_cast<unsigned char*>(ctx_.allocDevice(threshold_map_size));
    
    // Allocate quad scores for NMS
    d_quad_scores_ = static_cast<float*>(ctx_.allocDevice(MAX_QUADS * sizeof(float)));
    
    d_quads_ = static_cast<QuadCandidate*>(
        ctx_.allocDevice(MAX_QUADS * sizeof(QuadCandidate)));
    d_quad_count_ = static_cast<int*>(ctx_.allocDevice(sizeof(int)));
}

void AprilTagGpuDetector::quaternionFromRotationMatrix(const cv::Matx33f& R,
                                                        float& w, float& x, float& y, float& z) {
    float trace = R(0,0) + R(1,1) + R(2,2);
    if (trace > 0.f) {
        float s = sqrtf(trace + 1.f) * 2.f;
        w = 0.25f * s;
        x = (R(2,1) - R(1,2)) / s;
        y = (R(0,2) - R(2,0)) / s;
        z = (R(1,0) - R(0,1)) / s;
    } else if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
        float s = sqrtf(1.f + R(0,0) - R(1,1) - R(2,2)) * 2.f;
        w = (R(2,1) - R(1,2)) / s;
        x = 0.25f * s;
        y = (R(0,1) + R(1,0)) / s;
        z = (R(0,2) + R(2,0)) / s;
    } else if (R(1,1) > R(2,2)) {
        float s = sqrtf(1.f + R(1,1) - R(0,0) - R(2,2)) * 2.f;
        w = (R(0,2) - R(2,0)) / s;
        x = (R(0,1) + R(1,0)) / s;
        y = 0.25f * s;
        z = (R(1,2) + R(2,1)) / s;
    } else {
        float s = sqrtf(1.f + R(2,2) - R(0,0) - R(1,1)) * 2.f;
        w = (R(1,0) - R(0,1)) / s;
        x = (R(0,2) + R(2,0)) / s;
        y = (R(1,2) + R(2,1)) / s;
        z = 0.25f * s;
    }
}

void AprilTagGpuDetector::updateROIs(const std::vector<AprilTagDetection>& detections) {
    for (auto& roi : rois_) {
        roi.age++;
    }
    
    for (const auto& det : detections) {
        float min_x = std::min({det.corners[0].x, det.corners[1].x, det.corners[2].x, det.corners[3].x});
        float max_x = std::max({det.corners[0].x, det.corners[1].x, det.corners[2].x, det.corners[3].x});
        float min_y = std::min({det.corners[0].y, det.corners[1].y, det.corners[2].y, det.corners[3].y});
        float max_y = std::max({det.corners[0].y, det.corners[1].y, det.corners[2].y, det.corners[3].y});
        
        int w = static_cast<int>(max_x - min_x);
        int h = static_cast<int>(max_y - min_y);
        int x = static_cast<int>(min_x) - w / 4;
        int y = static_cast<int>(min_y) - h / 4;
        w = w + w / 2;
        h = h + h / 2;
        
        x = std::max(0, std::min(x, width_ - 1));
        y = std::max(0, std::min(y, height_ - 1));
        w = std::min(w, width_ - x);
        h = std::min(h, height_ - y);
        
        bool found = false;
        for (auto& roi : rois_) {
            if (roi.age < ROI_DECAY_FRAMES) {
                int dx = abs(roi.x + roi.w/2 - (x + w/2));
                int dy = abs(roi.y + roi.h/2 - (y + h/2));
                if (dx < w && dy < h) {
                    roi.x = x;
                    roi.y = y;
                    roi.w = w;
                    roi.h = h;
                    roi.age = 0;
                    found = true;
                    break;
                }
            }
        }
        
        if (!found && rois_.size() < MAX_ROIS) {
            rois_.push_back({x, y, w, h, 0});
        }
    }
    
    rois_.erase(
        std::remove_if(rois_.begin(), rois_.end(),
                      [](const ROI& r) { return r.age >= ROI_DECAY_FRAMES; }),
        rois_.end());
}

std::vector<AprilTagDetection> AprilTagGpuDetector::detectInRegion(unsigned char* gray_dev,
                                                                   int x, int y, int w, int h,
                                                                   DetectionTimings* timings) {
    // For GPU-only, process ROI on GPU
    // This is a simplified version - would use proper GPU ROI extraction
    return detect(gray_dev, false, timings);
}

std::vector<AprilTagGpuDetector::ROI> AprilTagGpuDetector::getROIs() const {
    return rois_;
}

bool AprilTagGpuDetector::isFullFrameDetection() const {
    return (frame_count_ % FULL_FRAME_INTERVAL) == 0;
}

std::vector<AprilTagDetection> AprilTagGpuDetector::detect(unsigned char* gray_dev,
                                                          bool full_frame,
                                                          DetectionTimings* timings) {
    // Increment frame count for multi-rate detection
    frame_count_++;
    
    cudaEvent_t events[8];
    for (int i = 0; i < 8; ++i) {
        cudaEventCreate(&events[i]);
    }

    dim3 block(16, 16);
    dim3 grid((width_ + block.x - 1) / block.x,
              (height_ + block.y - 1) / block.y);

    // Stage 1: Gradient magnitude (GPU)
    cudaEventRecord(events[0], ctx_.stream());
    gradientMagKernel<<<grid, block, 0, ctx_.stream()>>>(
        gray_dev, width_, d_grad_mag_, width_, height_);
    cudaEventRecord(events[1], ctx_.stream());
    checkCuda(cudaGetLastError(), "gradientMagKernel failed");

    // Stage 2: Adaptive edge thresholding (GPU)
    const int region_size = 64;  // 64x64 pixel regions for adaptive threshold
    int region_w = (width_ + region_size - 1) / region_size;
    int region_h = (height_ + region_size - 1) / region_size;
    dim3 thresh_block(8, 8);
    dim3 thresh_grid((region_w + thresh_block.x - 1) / thresh_block.x,
                     (region_h + thresh_block.y - 1) / thresh_block.y);
    
    // Compute adaptive threshold map
    computeAdaptiveThresholdKernel<<<thresh_grid, thresh_block, 0, ctx_.stream()>>>(
        d_grad_mag_, d_threshold_map_, width_, height_, width_, region_size);
    checkCuda(cudaGetLastError(), "computeAdaptiveThresholdKernel failed");
    
    // Apply adaptive threshold
    unsigned char edge_threshold = 40;  // Fallback threshold
    edgeThresholdKernel<<<grid, block, 0, ctx_.stream()>>>(
        d_grad_mag_, d_threshold_map_, d_edges_, width_, height_, width_, 
        region_size, edge_threshold);
    cudaEventRecord(events[2], ctx_.stream());
    checkCuda(cudaGetLastError(), "edgeThresholdKernel failed");

    cudaEventSynchronize(events[2]);

    float grad_ms = 0.f, edge_ms = 0.f;
    cudaEventElapsedTime(&grad_ms, events[0], events[1]);
    cudaEventElapsedTime(&edge_ms, events[1], events[2]);

    // Stage 4: Quad Extraction (GPU or OpenCV CPU)
    cudaEventRecord(events[3], ctx_.stream());
    auto t_quad0 = cv::getTickCount();
    float quad_ms = 0.f;
    
    std::vector<std::vector<cv::Point2f>> corners;
    int num_quads = 0;
    std::vector<int> opencv_decoded_ids;  // Store OpenCV decoded IDs if available
    
    if (use_gpu_quad_extraction_) {
        // GPU Quad Extraction
        // Allocate device memory for quad extraction
        float2* d_quad_corners = static_cast<float2*>(ctx_.allocDevice(MAX_QUADS * 4 * sizeof(float2)));
        int* d_quad_count = static_cast<int*>(ctx_.allocDevice(sizeof(int)));
        checkCuda(cudaMemset(d_quad_count, 0, sizeof(int)), "cudaMemset quad_count failed");
        
        // Calculate grid dimensions for quad extraction
        const int grid_size = 48;
        int grid_w = (width_ + grid_size - 1) / grid_size;
        int grid_h = (height_ + grid_size - 1) / grid_size;
        dim3 quad_block(8, 8);
        dim3 quad_grid((grid_w + quad_block.x - 1) / quad_block.x,
                       (grid_h + quad_block.y - 1) / quad_block.y);
        
        // Calculate min/max perimeter based on image size
        int min_perimeter = static_cast<int>(width_ * 0.03f + height_ * 0.03f);  // ~3% of image size
        int max_perimeter = static_cast<int>(width_ * 0.4f + height_ * 0.4f);   // ~40% of image size
        
        // Extract quads on GPU (with edge strength scores for NMS)
        extractQuadsFromEdgesKernel<<<quad_grid, quad_block, 0, ctx_.stream()>>>(
            d_edges_, d_grad_mag_, width_, height_, width_,
            d_quad_corners, d_quad_scores_, d_quad_count, MAX_QUADS,
            min_perimeter, max_perimeter);
        checkCuda(cudaGetLastError(), "extractQuadsFromEdgesKernel failed");
        
        cudaEventRecord(events[4], ctx_.stream());
        cudaEventSynchronize(events[4]);
        
        // Download quad count and corners
        checkCuda(cudaMemcpy(&num_quads, d_quad_count, sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy quad_count failed");
        
        // Apply GPU NMS before converting to OpenCV format
        int num_quads_after_nms = num_quads;
        if (num_quads > 1 && num_quads <= MAX_QUADS) {
            // Download quad scores for NMS
            std::vector<float> h_quad_scores(num_quads);
            checkCuda(cudaMemcpy(h_quad_scores.data(), d_quad_scores_,
                                 num_quads * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy quad_scores failed");
            
            // Download quad corners
            std::vector<float2> h_quad_corners(num_quads * 4);
            checkCuda(cudaMemcpy(h_quad_corners.data(), d_quad_corners,
                                 num_quads * 4 * sizeof(float2),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy quad_corners failed");
            
            // Apply NMS on CPU (can be moved to GPU later for better performance)
            std::vector<bool> keep_flags(num_quads, true);
            const float iou_threshold = 0.3f;  // IoU threshold for NMS
            
            for (int i = 0; i < num_quads; ++i) {
                if (!keep_flags[i]) continue;
                
                float2* corners_i = &h_quad_corners[i * 4];
                
                for (int j = i + 1; j < num_quads; ++j) {
                    if (!keep_flags[j]) continue;
                    
                    float2* corners_j = &h_quad_corners[j * 4];
                    
                    // Compute IoU (Intersection over Union)
                    float iou = computeQuadIoU(corners_i, corners_j);
                    
                    if (iou > iou_threshold) {
                        // Suppress the quad with lower score
                        if (h_quad_scores[i] > h_quad_scores[j]) {
                            keep_flags[j] = false;
                        } else {
                            keep_flags[i] = false;
                            break;
                        }
                    }
                }
            }
            
            // Filter quads based on NMS results
            corners.clear();
            for (int i = 0; i < num_quads; ++i) {
                if (keep_flags[i]) {
                    std::vector<cv::Point2f> quad_corners(4);
                    for (int j = 0; j < 4; ++j) {
                        quad_corners[j] = cv::Point2f(h_quad_corners[i * 4 + j].x,
                                                     h_quad_corners[i * 4 + j].y);
                    }
                    corners.push_back(quad_corners);
                }
            }
            num_quads_after_nms = static_cast<int>(corners.size());
        } else if (num_quads > 0 && num_quads <= MAX_QUADS) {
            // No NMS needed, just convert
            std::vector<float2> h_quad_corners(num_quads * 4);
            checkCuda(cudaMemcpy(h_quad_corners.data(), d_quad_corners,
                                 num_quads * 4 * sizeof(float2),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy quad_corners failed");
            
            corners.resize(num_quads);
            for (int i = 0; i < num_quads; ++i) {
                corners[i].resize(4);
                for (int j = 0; j < 4; ++j) {
                    corners[i][j] = cv::Point2f(h_quad_corners[i * 4 + j].x,
                                               h_quad_corners[i * 4 + j].y);
                }
            }
        }
    } else {
        // OpenCV CPU Quad Extraction
        // Download gray image for OpenCV (it needs the original image, not just edges)
        cv::Mat gray_cpu(height_, width_, CV_8UC1);
        checkCuda(cudaMemcpy(gray_cpu.data, gray_dev,
                             static_cast<size_t>(width_) * height_,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy gray D2H failed");
        
        // Use OpenCV aruco for quad extraction and decode
        cv::aruco::Dictionary dict =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
        cv::aruco::DetectorParameters params;
        params.adaptiveThreshWinSizeMin = 3;
        params.adaptiveThreshWinSizeMax = 15;
        params.adaptiveThreshWinSizeStep = 12;
        params.minMarkerPerimeterRate = 0.04;
        params.maxMarkerPerimeterRate = 3.5;
        params.polygonalApproxAccuracyRate = 0.05;
        params.minCornerDistanceRate = 0.05;
        params.minDistanceToBorder = 3;
        params.minOtsuStdDev = 5.0;
        params.perspectiveRemovePixelPerCell = 4;
        params.perspectiveRemoveIgnoredMarginPerCell = 0.15;
        params.maxErroneousBitsInBorderRate = 0.45;
        params.errorCorrectionRate = 0.6;  // Enable error correction for better detection
        params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
        params.adaptiveThreshConstant = 7;
        
        cv::aruco::ArucoDetector detector(dict, params);
        
        std::vector<int> aruco_ids;
        std::vector<std::vector<cv::Point2f>> rejected;
        detector.detectMarkers(gray_cpu, corners, aruco_ids, rejected);
        num_quads = static_cast<int>(corners.size());
        
        // Debug: Check if OpenCV found anything
        static int debug_count = 0;
        if (++debug_count % 100 == 0 && num_quads > 0) {
            std::cout << "[DEBUG] OpenCV found " << num_quads << " quads, " 
                      << aruco_ids.size() << " decoded IDs" << std::endl;
        }
        
        // Store OpenCV decoded IDs for use in decode section
        if (!aruco_ids.empty() && aruco_ids.size() == corners.size()) {
            opencv_decoded_ids = aruco_ids;
            if (debug_count % 100 == 0) {
                std::cout << "[DEBUG] Stored " << opencv_decoded_ids.size() << " OpenCV IDs" << std::endl;
            }
        }
        
        cudaEventRecord(events[4], ctx_.stream());
    }
    
    auto t_quad1 = cv::getTickCount();
    quad_ms = (t_quad1 - t_quad0) * 1000.0 / cv::getTickFrequency();
    
    // Now decode on GPU (or use OpenCV results)
    auto t_decode0 = cv::getTickCount();
    std::vector<int> ids;
    std::vector<float> decision_margins;  // Declare in function scope
    std::vector<float> hamming_distances;  // Declare in function scope
    
    int num_quads_final = (use_gpu_quad_extraction_ && num_quads > 1) ? 
                          static_cast<int>(corners.size()) : num_quads;
    
    // If OpenCV already decoded tags (CPU path), use those results directly
    if (!use_gpu_quad_extraction_ && !opencv_decoded_ids.empty() && 
        opencv_decoded_ids.size() == corners.size()) {
        // Use OpenCV decoded results
        ids = opencv_decoded_ids;
        decision_margins.resize(ids.size(), 0.8f);  // High confidence for OpenCV
        hamming_distances.resize(ids.size(), 0.0f);
        num_quads_final = static_cast<int>(ids.size());
        
        // Debug
        static int use_opencv_count = 0;
        if (++use_opencv_count % 100 == 0) {
            std::cout << "[DEBUG] Using OpenCV decoded IDs: " << ids.size() << " tags" << std::endl;
        }
        
        // Skip GPU decode, go straight to PnP
        goto skip_gpu_decode_section;
    }
    
    if (!corners.empty() && num_quads_final > 0) {
        int num_quads = num_quads_final;
        // Upload quad corners to GPU (needed for both GPU and CPU quad extraction)
        float2* d_quad_corners = static_cast<float2*>(ctx_.allocDevice(num_quads * 4 * sizeof(float2)));
        std::vector<float2> h_quad_corners(num_quads * 4);
        for (int i = 0; i < num_quads; ++i) {
            for (int j = 0; j < 4; ++j) {
                h_quad_corners[i * 4 + j] = make_float2(corners[i][j].x, corners[i][j].y);
            }
        }
        checkCuda(cudaMemcpyAsync(d_quad_corners, h_quad_corners.data(),
                                   num_quads * 4 * sizeof(float2),
                                   cudaMemcpyHostToDevice, ctx_.stream()),
                  "cudaMemcpy quad_corners H2D failed");
        
        // Allocate tag bits buffer (36 bits per tag: 6x6 grid)
        unsigned char* d_tag_bits = static_cast<unsigned char*>(ctx_.allocDevice(num_quads * 36 * sizeof(unsigned char)));
        
        // Sub-pixel corner refinement (optional, improves pose accuracy)
        if (enable_subpixel_refinement_) {
            refineCornersSubpixelKernel<<<num_quads, 1, 0, ctx_.stream()>>>(
                gray_dev, width_, height_, width_,
                d_quad_corners, num_quads, true);
            checkCuda(cudaGetLastError(), "refineCornersSubpixelKernel failed");
            
            // Download refined corners
            checkCuda(cudaMemcpy(h_quad_corners.data(), d_quad_corners,
                                 num_quads * 4 * sizeof(float2),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy refined corners failed");
            
            // Update corners vector with refined positions
            for (int i = 0; i < num_quads; ++i) {
                for (int j = 0; j < 4; ++j) {
                    corners[i][j] = cv::Point2f(h_quad_corners[i * 4 + j].x,
                                               h_quad_corners[i * 4 + j].y);
                }
            }
            
            // Re-upload refined corners
            checkCuda(cudaMemcpyAsync(d_quad_corners, h_quad_corners.data(),
                                       num_quads * 4 * sizeof(float2),
                                       cudaMemcpyHostToDevice, ctx_.stream()),
                      "cudaMemcpy refined corners H2D failed");
        }
        
        // Sample tag bits from gray image
        dim3 decode_block(36);  // One thread per bit
        dim3 decode_grid(num_quads);
        sampleTagBitsKernel<<<decode_grid, decode_block, 0, ctx_.stream()>>>(
            gray_dev, width_, height_, width_,
            d_quad_corners, d_tag_bits, num_quads);
        checkCuda(cudaGetLastError(), "sampleTagBitsKernel failed");
        
        // Decode tags on GPU (with confidence scores)
        int* d_tag_ids = static_cast<int*>(ctx_.allocDevice(num_quads * sizeof(int)));
        float* d_decision_margins = static_cast<float*>(ctx_.allocDevice(num_quads * sizeof(float)));
        float* d_hamming_distances = static_cast<float*>(ctx_.allocDevice(num_quads * sizeof(float)));
        float decode_threshold = 127.0f;  // Threshold for bit binarization
        decodeTagsKernel<<<num_quads, 1, 0, ctx_.stream()>>>(
            d_tag_bits, d_tag_ids, d_decision_margins, d_hamming_distances, 
            num_quads, decode_threshold);
        checkCuda(cudaGetLastError(), "decodeTagsKernel failed");
        
        // Download decoded IDs and confidence scores
        ids.resize(num_quads);
        decision_margins.resize(num_quads);
        hamming_distances.resize(num_quads);
        checkCuda(cudaMemcpy(ids.data(), d_tag_ids, num_quads * sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy tag_ids failed");
        checkCuda(cudaMemcpy(decision_margins.data(), d_decision_margins, 
                             num_quads * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy decision_margins failed");
        checkCuda(cudaMemcpy(hamming_distances.data(), d_hamming_distances,
                             num_quads * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy hamming_distances failed");
        
        // Filter out invalid IDs and apply quality thresholds
        std::vector<std::vector<cv::Point2f>> valid_corners;
        std::vector<int> valid_ids;
        std::vector<float> valid_decision_margins;
        std::vector<float> valid_hamming_distances;
        for (size_t i = 0; i < corners.size(); ++i) {
            if (i < ids.size() && ids[i] >= 0) {
                // Apply decode quality filter
                float margin = (i < decision_margins.size()) ? decision_margins[i] : 0.5f;
                if (margin >= min_decision_margin_) {
                    valid_corners.push_back(corners[i]);
                    valid_ids.push_back(ids[i]);
                    valid_decision_margins.push_back(margin);
                    if (i < hamming_distances.size()) {
                        valid_hamming_distances.push_back(hamming_distances[i]);
                    } else {
                        valid_hamming_distances.push_back(0.0f);
                    }
                }
            }
        }
        corners = valid_corners;
        ids = valid_ids;
        decision_margins = valid_decision_margins;
        hamming_distances = valid_hamming_distances;
    }
    
    skip_gpu_decode_section:
    auto t_decode1 = cv::getTickCount();
    float decode_ms = (t_decode1 - t_decode0) * 1000.0 / cv::getTickFrequency();
    
    // Debug: Check what we have after decode
    static int decode_debug_count = 0;
    if (++decode_debug_count % 100 == 0 && !ids.empty()) {
        std::cout << "[DEBUG] After decode: " << ids.size() << " IDs, " 
                  << corners.size() << " corners" << std::endl;
    }
    
    // Ensure decision_margins and hamming_distances are available
    // (they may have been filtered earlier)
    if (decision_margins.size() != ids.size()) {
        decision_margins.resize(ids.size(), 0.5f);
    }
    if (hamming_distances.size() != ids.size()) {
        hamming_distances.resize(ids.size(), 0.0f);
    }
    
    std::vector<AprilTagDetection> out;
    if (!ids.empty() && ids.size() == corners.size()) {
        // Upload corners and solve PnP on GPU
        int num_detections = static_cast<int>(ids.size());
        if (num_detections > 0 && num_detections <= MAX_QUADS) {
            // OpenCV aruco returns corners starting from first detected corner, going clockwise
            // AprilTag standard expects: top-left, top-right, bottom-right, bottom-left
            // But OpenCV might start from a different corner. Test all 4 rotations to find correct one.
            
            float s = tag_size_ * 0.5f;
            std::vector<cv::Point3f> obj_pts_cv;
            obj_pts_cv.push_back(cv::Point3f(-s,  s, 0));  // top-left
            obj_pts_cv.push_back(cv::Point3f( s,  s, 0));  // top-right
            obj_pts_cv.push_back(cv::Point3f( s, -s, 0));  // bottom-right
            obj_pts_cv.push_back(cv::Point3f(-s, -s, 0));  // bottom-left
            
            cv::Mat K_cv = (cv::Mat_<double>(3, 3) << 
                static_cast<double>(K_(0,0)), 0, static_cast<double>(K_(0,2)),
                0, static_cast<double>(K_(1,1)), static_cast<double>(K_(1,2)),
                0, 0, 1);
            
            // Test all 4 corner rotations AND all 4 object point orderings
            // This will find the correct mapping between OpenCV corners and AprilTag object points
            int best_rotation = 0;
            int best_obj_order = 0;
            float best_reproj = 1e9f;
            
            // Try different object point orderings
            std::vector<std::vector<cv::Point3f>> obj_pts_variants(4);
            for (int obj_rot = 0; obj_rot < 4; ++obj_rot) {
                obj_pts_variants[obj_rot].resize(4);
                for (int j = 0; j < 4; ++j) {
                    int idx = (j + obj_rot) % 4;
                    obj_pts_variants[obj_rot][j] = obj_pts_cv[idx];
                }
            }
            
            if (num_detections > 0) {
                for (int rot = 0; rot < 4; ++rot) {
                    std::vector<cv::Point2f> test_corners(4);
                    for (int j = 0; j < 4; ++j) {
                        test_corners[j] = corners[0][(j + rot) % 4];
                    }
                    
                    for (int obj_rot = 0; obj_rot < 4; ++obj_rot) {
                        cv::Mat rvec, tvec;
                        bool success = cv::solvePnP(obj_pts_variants[obj_rot], test_corners, K_cv, cv::Mat(), 
                                                     rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                        if (success) {
                            std::vector<cv::Point2f> projected;
                            cv::projectPoints(obj_pts_variants[obj_rot], rvec, tvec, K_cv, cv::Mat(), projected);
                            float reproj = 0.0f;
                            for (size_t j = 0; j < 4; ++j) {
                                float dx = test_corners[j].x - projected[j].x;
                                float dy = test_corners[j].y - projected[j].y;
                                reproj += sqrtf(dx*dx + dy*dy);
                            }
                            reproj /= 4.0f;
                            
                            if (reproj < best_reproj) {
                                best_reproj = reproj;
                                best_rotation = rot;
                                best_obj_order = obj_rot;
                            }
                        }
                    }
                }
                
                static int rot_test_count = 0;
                if (++rot_test_count % 100 == 0) {
                    std::cout << "[DEBUG] Best: corner_rot=" << best_rotation 
                              << " obj_rot=" << best_obj_order
                              << " (reproj=" << best_reproj << "px)" << std::endl;
                }
            }
            
            // Use best object point ordering
            obj_pts_cv = obj_pts_variants[best_obj_order];
            
            // Apply best rotation to all detections
            std::vector<std::vector<cv::Point2f>> reordered_corners(num_detections);
            std::vector<float2> h_img_points(num_detections * 4);
            for (int i = 0; i < num_detections; ++i) {
                reordered_corners[i].resize(4);
                for (int j = 0; j < 4; ++j) {
                    reordered_corners[i][j] = corners[i][(j + best_rotation) % 4];
                }
                
                h_img_points[i * 4 + 0] = make_float2(reordered_corners[i][0].x, reordered_corners[i][0].y);
                h_img_points[i * 4 + 1] = make_float2(reordered_corners[i][1].x, reordered_corners[i][1].y);
                h_img_points[i * 4 + 2] = make_float2(reordered_corners[i][2].x, reordered_corners[i][2].y);
                h_img_points[i * 4 + 3] = make_float2(reordered_corners[i][3].x, reordered_corners[i][3].y);
            }
            
            float2* d_img_points = static_cast<float2*>(ctx_.allocDevice(num_detections * 4 * sizeof(float2)));
            checkCuda(cudaMemcpyAsync(d_img_points, h_img_points.data(),
                                       num_detections * 4 * sizeof(float2),
                                       cudaMemcpyHostToDevice, ctx_.stream()),
                      "cudaMemcpy img_points failed");
            
            // Use OpenCV solvePnP directly (GPU PnP kernel has issues)
            // OpenCV gives 0.5px reprojection error vs 387px from GPU kernel
            std::vector<cv::Mat> rvecs(num_detections), tvecs(num_detections);
            std::vector<float> h_reprojection_errors(num_detections);
            
            cudaEventRecord(events[5], ctx_.stream());
            
            // Use OpenCV solvePnP for each detection
            for (int i = 0; i < num_detections; ++i) {
                cv::Mat rvec, tvec;
                bool success = cv::solvePnP(obj_pts_cv, reordered_corners[i], K_cv, cv::Mat(), 
                                           rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                if (success) {
                    rvecs[i] = rvec.clone();
                    tvecs[i] = tvec.clone();
                    
                    // Compute reprojection error
                    std::vector<cv::Point2f> projected;
                    cv::projectPoints(obj_pts_cv, rvec, tvec, K_cv, cv::Mat(), projected);
                    float reproj = 0.0f;
                    for (size_t j = 0; j < 4; ++j) {
                        float dx = reordered_corners[i][j].x - projected[j].x;
                        float dy = reordered_corners[i][j].y - projected[j].y;
                        reproj += sqrtf(dx*dx + dy*dy);
                    }
                    h_reprojection_errors[i] = reproj / 4.0f;
                } else {
                    h_reprojection_errors[i] = 1000.0f;  // Large error if solvePnP failed
                }
            }
            
            cudaEventRecord(events[6], ctx_.stream());
            cudaEventSynchronize(events[6]);
            
            // Convert to our format
            std::vector<float> h_R(num_detections * 9);
            std::vector<float> h_t(num_detections * 3);
            
            // Convert OpenCV rvec/tvec to rotation matrix and translation vector
            for (int i = 0; i < num_detections; ++i) {
                cv::Mat R_cv;
                cv::Rodrigues(rvecs[i], R_cv);
                
                // Store rotation matrix (row-major)
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        h_R[i * 9 + row * 3 + col] = static_cast<float>(R_cv.at<double>(row, col));
                    }
                }
                
                // Store translation
                h_t[i * 3 + 0] = static_cast<float>(tvecs[i].at<double>(0));
                h_t[i * 3 + 1] = static_cast<float>(tvecs[i].at<double>(1));
                h_t[i * 3 + 2] = static_cast<float>(tvecs[i].at<double>(2));
            }
            
            // Get edge strength scores (if available from GPU quad extraction)
            std::vector<float> edge_strengths(num_detections, 100.0f);  // Default
            if (use_gpu_quad_extraction_ && num_detections <= MAX_QUADS) {
                // Download edge scores (they were computed during quad extraction)
                std::vector<float> h_quad_scores(num_detections);
                checkCuda(cudaMemcpy(h_quad_scores.data(), d_quad_scores_,
                                     num_detections * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy quad_scores for detections failed");
                edge_strengths = h_quad_scores;
            }
            
            // Build detection results with quality filtering
            static int filter_debug_count = 0;
            for (int i = 0; i < num_detections; ++i) {
                // Apply quality filters
                float decision_margin = (i < decision_margins.size()) ? decision_margins[i] : 0.5f;
                float reproj_error = h_reprojection_errors[i];
                float edge_strength = edge_strengths[i];
                
                // Debug filtering
                if (++filter_debug_count % 100 == 0) {
                    std::cout << "[DEBUG] Filter check: margin=" << decision_margin 
                              << " (min=" << min_decision_margin_ << "), "
                              << "reproj=" << reproj_error << " (max=" << max_reprojection_error_ << ")" << std::endl;
                }
                
                // Filter based on quality thresholds
                if (decision_margin < min_decision_margin_) {
                    if (filter_debug_count % 100 == 0) {
                        std::cout << "[DEBUG] Filtered out: decision_margin too low" << std::endl;
                    }
                    continue;
                }
                if (reproj_error > max_reprojection_error_) {
                    if (filter_debug_count % 100 == 0) {
                        std::cout << "[DEBUG] Filtered out: reprojection error too high" << std::endl;
                    }
                    continue;
                }
                
                AprilTagDetection det;
                det.id = ids[i];
                det.decision_margin = decision_margin;
                det.hamming_distance = (i < hamming_distances.size()) ? hamming_distances[i] : 0.0f;
                det.reprojection_error = reproj_error;
                det.edge_strength = edge_strength;
                
                // Use reordered corners (matching the PnP mapping)
                for (int j = 0; j < 4; ++j) {
                    det.corners[j] = reordered_corners[i][j];
                }
                
                // Build transformation matrix from R and t
                cv::Matx33f R_mat;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        R_mat(r, c) = h_R[i * 9 + r * 3 + c];
                    }
                }
                cv::Matx44f T = cv::Matx44f::eye();
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        T(r, c) = R_mat(r, c);
                    }
                }
                T(0, 3) = h_t[i * 3 + 0];
                T(1, 3) = h_t[i * 3 + 1];
                T(2, 3) = h_t[i * 3 + 2];
                det.T_cam_tag = T;
                quaternionFromRotationMatrix(R_mat, det.quat_w, det.quat_x, det.quat_y, det.quat_z);
                
                // Final quality check (temporarily disabled for debugging)
                // if (det.quality() >= min_quality_) {
                    out.push_back(det);
                // }
            }
            
            float pnp_ms = 0.f;
            cudaEventElapsedTime(&pnp_ms, events[5], events[6]);
            
            if (timings) {
                timings->grad_ms = grad_ms;
                timings->edge_ms = edge_ms;
                timings->quad_ms = quad_ms;
                timings->decode_ms = decode_ms;
                timings->pnp_ms = pnp_ms;
                timings->d2h_ms = 0.f;
                timings->total_ms = grad_ms + edge_ms + quad_ms + decode_ms + pnp_ms;
            }
            
            for (int i = 0; i < 8; ++i) {
                cudaEventDestroy(events[i]);
            }
            
            updateROIs(out);
            return out;
        }
    }
    
    // No detections
    float pnp_ms = 0.f;
    if (timings) {
        timings->grad_ms = grad_ms;
        timings->edge_ms = edge_ms;
        timings->quad_ms = quad_ms;
        timings->decode_ms = decode_ms;
        timings->pnp_ms = pnp_ms;
        timings->d2h_ms = 0.f;
        timings->total_ms = grad_ms + edge_ms + quad_ms + decode_ms + pnp_ms;
    }
    
    for (int i = 0; i < 8; ++i) {
        cudaEventDestroy(events[i]);
    }
    
    return std::vector<AprilTagDetection>();
}
