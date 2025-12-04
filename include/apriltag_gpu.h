#pragma once

#include "gpu_context.h"

#include <opencv2/core.hpp>
#include <vector>
#include <map>

// GPU AprilTag detector interface matching the requirements document.
// Implements full GPU pipeline: gradients -> edge map -> quad extraction -> decode -> PnP
// Includes ROI tracking and multi-rate detection for performance

struct AprilTagDetection {
    int id{0};
    float decision_margin{0.f};  // Decode confidence (0-1, higher is better)
    float hamming_distance{0.f}; // Hamming distance from valid code
    float reprojection_error{0.f}; // Pose reprojection error in pixels
    float edge_strength{0.f};      // Edge strength score
    cv::Point2f corners[4];  // image-space corners (sub-pixel refined)
    cv::Matx44f T_cam_tag;   // 4x4 pose of tag in camera frame
    // Quaternion representation (w, x, y, z)
    float quat_w{1.f}, quat_x{0.f}, quat_y{0.f}, quat_z{0.f};
    
    // Quality score (0-1, higher is better)
    float quality() const {
        // decision_margin is already 0-1 where higher is better
        // reprojection_error should be low (good) - normalize to 0-1
        // edge_strength should be high (good) - normalize to 0-1
        float reproj_score = 1.0f - fminf(1.0f, reprojection_error / 10.0f);
        float edge_score = fminf(1.0f, edge_strength / 255.0f);
        return decision_margin * reproj_score * edge_score;
    }
};

// Per-frame timing breakdown for the AprilTag detector (milliseconds).
struct DetectionTimings {
    float grad_ms{0.f};         // gradient magnitude kernel
    float edge_ms{0.f};         // edge thresholding
    float quad_ms{0.f};         // quad extraction
    float decode_ms{0.f};       // tag decode
    float pnp_ms{0.f};          // solvePnP (GPU)
    float d2h_ms{0.f};          // device-to-host copy (if needed)
    float total_ms{0.f};        // total detection time
};

class AprilTagGpuDetector {
public:
    AprilTagGpuDetector(GpuContext& ctx,
                        int width,
                        int height,
                        float tag_size_meters,
                        const cv::Matx33f& K);

    // Run detection on a preprocessed GRAY8 device image.
    //  - gray_dev must be width x height, row-major, 1 byte per pixel.
    //  - full_frame: if true, scan entire frame; if false, only check ROI regions
    // Returns vector of detections on the host.
    // Optionally fills 'timings' with per-stage timing information.
    std::vector<AprilTagDetection> detect(unsigned char* gray_dev,
                                          bool full_frame = true,
                                          DetectionTimings* timings = nullptr);

    // Update ROI regions based on last detections (for multi-rate tracking)
    void updateROIs(const std::vector<AprilTagDetection>& detections);
    
    // Get current ROI regions (for visualization)
    struct ROI {
        int x, y, w, h;
        int age;
    };
    std::vector<ROI> getROIs() const;
    bool isFullFrameDetection() const;

    // Toggle between GPU and OpenCV CPU quad extraction
    void setUseGpuQuadExtraction(bool use_gpu) { use_gpu_quad_extraction_ = use_gpu; }
    bool useGpuQuadExtraction() const { return use_gpu_quad_extraction_; }
    
    // Filtering and quality settings
    void setMinQuality(float min_quality) { min_quality_ = min_quality; }
    void setMaxReprojectionError(float max_error) { max_reprojection_error_ = max_error; }
    void setMinDecisionMargin(float min_margin) { min_decision_margin_ = min_margin; }
    void setEnableSubpixelRefinement(bool enable) { enable_subpixel_refinement_ = enable; }
    void setEnableTemporalFiltering(bool enable) { enable_temporal_filtering_ = enable; }
    void setTemporalFilterAlpha(float alpha) { temporal_filter_alpha_ = alpha; }
    void setTemporalFilterMaxAge(int max_age) { temporal_filter_max_age_ = max_age; }

    // Temporal filtering: track pose history per tag ID
    struct PoseHistory {
        cv::Matx44f T_cam_tag;              // Smoothed pose
        float quat_w{1.f}, quat_x{0.f}, quat_y{0.f}, quat_z{0.f};  // Smoothed quaternion
        int age{0};                          // Frames since last update
        bool initialized{false};             // Has this tag been seen before?
    };

    int width() const { return width_; }
    int height() const { return height_; }

private:
    GpuContext& ctx_;
    int width_;
    int height_;
    float tag_size_;
    cv::Matx33f K_;

    // Device buffers
    unsigned char* d_grad_mag_{nullptr};
    unsigned char* d_edges_{nullptr};
    unsigned char* d_threshold_map_{nullptr};  // For adaptive thresholding
    float* d_quad_scores_{nullptr};  // For NMS (edge strength scores)
    
    // Quad extraction mode
    bool use_gpu_quad_extraction_{true};  // Default to GPU
    
    // Quality filtering settings (relaxed for better detection)
    float min_quality_{0.01f};              // Minimum quality score (0-1) - very low threshold
    float max_reprojection_error_{500.0f};  // Max reprojection error in pixels - very permissive (temporarily high due to calibration)
    float min_decision_margin_{0.05f};       // Minimum decode confidence - very low threshold
    bool enable_subpixel_refinement_{true};  // Enable sub-pixel corner refinement
    bool enable_temporal_filtering_{false};  // Enable temporal filtering (pose smoothing)
    float temporal_filter_alpha_{0.3f};      // EMA alpha for temporal filtering (0-1)
    int temporal_filter_max_age_{30};       // Max frames to keep pose history
    std::map<int, PoseHistory> pose_history_;  // tag_id -> pose history

    // ROI tracking for multi-rate detection (ROI struct defined above in public section)
    std::vector<ROI> rois_;
    static constexpr int MAX_ROIS = 16;
    static constexpr int ROI_DECAY_FRAMES = 20;  // remove ROI after N frames without detection (increased for stability)
    static constexpr int FULL_FRAME_INTERVAL = 30;  // full frame scan every N frames (very aggressive for max speed)
    int frame_count_{0};

    // Quad candidates storage (max 256 quads)
    static constexpr int MAX_QUADS = 256;
    struct QuadCandidate {
        float2 corners[4];
        float score;
    };
    QuadCandidate* d_quads_{nullptr};
    int* d_quad_count_{nullptr};

    void allocateBuffers();
    void quaternionFromRotationMatrix(const cv::Matx33f& R, float& w, float& x, float& y, float& z);
    void applyTemporalFilter(AprilTagDetection& det);  // Apply temporal smoothing to pose
    std::vector<AprilTagDetection> detectInRegion(unsigned char* gray_dev,
                                                   int x, int y, int w, int h,
                                                   DetectionTimings* timings);
};
