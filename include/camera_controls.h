#pragma once

#include <opencv2/highgui.hpp>
#include "config.h"
#include <iostream>
#include <vector>
#include <string>

// Camera information structure
struct CameraInfo {
    int index;                          // Camera index
    std::string device_path;            // Device path (e.g., "/dev/video0")
    std::string backend_name;           // Backend name (V4L2, etc.)
    bool is_available;                  // Can be opened
    int width;                          // Current/default width
    int height;                         // Current/default height
    double fps;                         // Current/default FPS
    std::vector<int> supported_widths;  // Supported resolutions (widths)
    std::vector<int> supported_heights;  // Supported resolutions (heights)
    std::vector<double> supported_fps;  // Supported FPS values
};

// Camera feature/capability information
struct CameraFeatures {
    bool supports_brightness{false};
    bool supports_contrast{false};
    bool supports_saturation{false};
    bool supports_exposure{false};
    bool supports_gain{false};
    bool supports_white_balance{false};
    bool supports_sharpness{false};
    bool supports_gamma{false};
    bool supports_auto_exposure{false};
    bool supports_auto_white_balance{false};
    
    // Property ranges (if available)
    double brightness_min{0}, brightness_max{0};
    double contrast_min{0}, contrast_max{0};
    double exposure_min{0}, exposure_max{0};
    double gain_min{0}, gain_max{0};
    double white_balance_min{0}, white_balance_max{0};
};

// ============================================================================
// Camera Detection Functions
// ============================================================================

// Detect all available cameras (try indices 0-15 and /dev/video*)
std::vector<CameraInfo> detectAvailableCameras(bool use_v4l2 = true);

// Detect cameras using V4L2 (check /dev/video* devices)
std::vector<CameraInfo> detectV4L2Cameras();

// Detect cameras by trying index-based opening
std::vector<CameraInfo> detectIndexCameras(int max_index = 15, bool use_v4l2 = true);

// Get camera info for a specific device/index
CameraInfo getCameraInfo(int index, bool use_v4l2 = true);
CameraInfo getCameraInfo(const std::string& device_path, bool use_v4l2 = true);

// Print list of available cameras
void printAvailableCameras(const std::vector<CameraInfo>& cameras);

// ============================================================================
// Camera Features/Capabilities Functions
// ============================================================================

// Get camera features/capabilities
CameraFeatures getCameraFeatures(cv::VideoCapture& cap);

// Check if a specific property is supported
bool isPropertySupported(cv::VideoCapture& cap, int property_id);

// Get property range (min, max, step)
bool getPropertyRange(cv::VideoCapture& cap, int property_id, 
                      double& min_val, double& max_val, double& step);

// Print camera features/capabilities
void printCameraFeatures(const CameraFeatures& features);

// ============================================================================
// Camera Settings Functions
// ============================================================================

// Apply camera control settings from config to OpenCV VideoCapture
void applyCameraControls(cv::VideoCapture& cap, const AppConfig::Camera::Controls& controls);

// Read all camera control values from camera
AppConfig::Camera::Controls readCameraControls(cv::VideoCapture& cap);

// Write camera control values to camera
bool writeCameraControls(cv::VideoCapture& cap, const AppConfig::Camera::Controls& controls);

// Print all camera control values to console
void printCameraControls(cv::VideoCapture& cap);

// Save current camera settings to config
bool saveCameraControlsToConfig(cv::VideoCapture& cap, const std::string& config_file);

// Load camera settings from config and apply
bool loadAndApplyCameraControls(cv::VideoCapture& cap, const std::string& config_file);

// ============================================================================
// Camera Opening Helper Functions
// ============================================================================

// Open camera with automatic detection and fallback
bool openCamera(cv::VideoCapture& cap, const std::string& device_or_index, 
                bool use_v4l2 = true, int width = -1, int height = -1, double fps = -1);

// Open camera with automatic detection from config
bool openCameraFromConfig(cv::VideoCapture& cap, const AppConfig& config);

// Test if camera can be opened
bool testCamera(int index, bool use_v4l2 = true);
bool testCamera(const std::string& device_path, bool use_v4l2 = true);

