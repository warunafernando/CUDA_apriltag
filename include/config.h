#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "image_preprocessor.h"

// Application configuration structure
struct AppConfig {
    // Camera settings
    struct Camera {
        int width{1280};
        int height{720};
        int fps{120};
        int decimation{2};  // Working resolution = width/decimation x height/decimation
        std::string device{"/dev/video0"};  // Camera device path or index
        bool use_v4l2{true};  // Use V4L2 backend
    } camera;
    
    // Tag settings
    struct Tag {
        float size_m{0.165f};  // FRC tag size in meters (6.5 inches)
    } tag;
    
    // Detection settings
    struct Detection {
        float min_quality{0.01f};              // Minimum quality score (0-1)
        float max_reprojection_error{500.0f};  // Max reprojection error (pixels)
        float min_decision_margin{0.05f};      // Minimum decode confidence
        bool enable_subpixel_refinement{true}; // Enable sub-pixel corner refinement
        bool enable_temporal_filtering{false}; // Enable temporal filtering
        bool use_gpu_quad_extraction{false};   // Use GPU quad extraction (false = OpenCV CPU)
    } detection;
    
    // ROI settings
    struct ROI {
        int full_frame_interval{30};  // Full frame scan every N frames
        int decay_frames{20};          // ROI persists for N frames without detection
        int max_rois{16};              // Maximum tracked ROIs
    } roi;
    
    // Test/benchmark settings
    struct Test {
        double duration_seconds{60.0};  // Test duration
        int max_captures{5};             // Max frame captures for documentation
        std::string capture_dir{"captures"};  // Directory for captured frames
    } test;
    
    // File paths
    struct Files {
        std::string intrinsics{"camera_intrinsics.json"};  // Camera intrinsics file
        std::string config{"config.json"};                 // This config file
    } files;
    
    // Default camera intrinsics (used if intrinsics file not found)
    CameraIntrinsics default_intrinsics;
    
    // Load configuration from JSON file
    static bool loadFromJSON(const std::string& filename, AppConfig& config);
    
    // Save configuration to JSON file
    static bool saveToJSON(const std::string& filename, const AppConfig& config);
    
    // Initialize default intrinsics based on camera resolution
    void initDefaultIntrinsics() {
        default_intrinsics.fx = static_cast<float>(camera.width) * 0.8f;
        default_intrinsics.fy = static_cast<float>(camera.height) * 0.8f;
        default_intrinsics.cx = static_cast<float>(camera.width) / 2.0f;
        default_intrinsics.cy = static_cast<float>(camera.height) / 2.0f;
        default_intrinsics.k1 = 0.0f;
        default_intrinsics.k2 = 0.0f;
        default_intrinsics.p1 = 0.0f;
        default_intrinsics.p2 = 0.0f;
        default_intrinsics.k3 = 0.0f;
    }
};

