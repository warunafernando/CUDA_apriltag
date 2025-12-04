#include "camera_controls.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <glob.h>
#include <dirent.h>

AppConfig::Camera::Controls readCameraControls(cv::VideoCapture& cap) {
    AppConfig::Camera::Controls controls;
    
    if (!cap.isOpened()) {
        std::cerr << "Warning: Cannot read camera controls - camera not opened" << std::endl;
        return controls;
    }
    
    // Read all properties
    controls.brightness = cap.get(cv::CAP_PROP_BRIGHTNESS);
    controls.contrast = cap.get(cv::CAP_PROP_CONTRAST);
    controls.saturation = cap.get(cv::CAP_PROP_SATURATION);
    controls.exposure = cap.get(cv::CAP_PROP_EXPOSURE);
    controls.gain = cap.get(cv::CAP_PROP_GAIN);
    controls.white_balance = cap.get(cv::CAP_PROP_WB_TEMPERATURE);
    if (controls.white_balance <= 0) {
        // Try alternative property
        controls.white_balance = cap.get(cv::CAP_PROP_WHITE_BALANCE_BLUE_U);
    }
    controls.sharpness = cap.get(cv::CAP_PROP_SHARPNESS);
    controls.gamma = cap.get(cv::CAP_PROP_GAMMA);
    
    // Auto exposure: 0.25 = manual, 0.75 = auto
    double auto_exp = cap.get(cv::CAP_PROP_AUTO_EXPOSURE);
    if (auto_exp >= 0.5) {
        controls.auto_exposure = 1;  // Auto
    } else if (auto_exp >= 0) {
        controls.auto_exposure = 0;  // Manual
    } else {
        controls.auto_exposure = -1;  // Unknown/not supported
    }
    
    // Auto white balance: 0 = manual, 1 = auto
    double auto_wb = cap.get(cv::CAP_PROP_AUTO_WB);
    if (auto_wb >= 0.5) {
        controls.auto_white_balance = 1;  // Auto
    } else if (auto_wb >= 0) {
        controls.auto_white_balance = 0;  // Manual
    } else {
        controls.auto_white_balance = -1;  // Unknown/not supported
    }
    
    return controls;
}

void printCameraControls(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        std::cerr << "Warning: Cannot read camera controls - camera not opened" << std::endl;
        return;
    }
    
    std::cout << "\n=== Current Camera Control Values ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Basic properties
    std::cout << "Brightness:     " << cap.get(cv::CAP_PROP_BRIGHTNESS) << std::endl;
    std::cout << "Contrast:       " << cap.get(cv::CAP_PROP_CONTRAST) << std::endl;
    std::cout << "Saturation:     " << cap.get(cv::CAP_PROP_SATURATION) << std::endl;
    std::cout << "Sharpness:      " << cap.get(cv::CAP_PROP_SHARPNESS) << std::endl;
    std::cout << "Gamma:          " << cap.get(cv::CAP_PROP_GAMMA) << std::endl;
    
    // Exposure
    double auto_exp = cap.get(cv::CAP_PROP_AUTO_EXPOSURE);
    std::cout << "Auto Exposure:  " << (auto_exp >= 0.5 ? "Auto" : (auto_exp >= 0 ? "Manual" : "N/A")) 
              << " (" << auto_exp << ")" << std::endl;
    std::cout << "Exposure:       " << cap.get(cv::CAP_PROP_EXPOSURE) << std::endl;
    std::cout << "Gain:           " << cap.get(cv::CAP_PROP_GAIN) << std::endl;
    
    // White balance
    double auto_wb = cap.get(cv::CAP_PROP_AUTO_WB);
    std::cout << "Auto WB:        " << (auto_wb >= 0.5 ? "Auto" : (auto_wb >= 0 ? "Manual" : "N/A")) 
              << " (" << auto_wb << ")" << std::endl;
    double wb_temp = cap.get(cv::CAP_PROP_WB_TEMPERATURE);
    if (wb_temp > 0) {
        std::cout << "WB Temperature: " << wb_temp << " K" << std::endl;
    } else {
        double wb_blue = cap.get(cv::CAP_PROP_WHITE_BALANCE_BLUE_U);
        if (wb_blue > 0) {
            std::cout << "WB Blue:        " << wb_blue << std::endl;
        }
    }
    
    // Additional info
    std::cout << "\n=== Camera Info ===" << std::endl;
    std::cout << "Width:          " << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) << std::endl;
    std::cout << "Height:         " << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) << std::endl;
    std::cout << "FPS:            " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Format:         " << cap.get(cv::CAP_PROP_FORMAT) << std::endl;
    std::cout << "Mode:           " << cap.get(cv::CAP_PROP_MODE) << std::endl;
    std::cout << std::endl;
}

bool saveCameraControlsToConfig(cv::VideoCapture& cap, const std::string& config_file) {
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot read camera controls - camera not opened" << std::endl;
        return false;
    }
    
    // Load existing config
    AppConfig config;
    if (!AppConfig::loadFromJSON(config_file, config)) {
        std::cout << "Config file not found, creating new one..." << std::endl;
        config.initDefaultIntrinsics();
    }
    
    // Read current camera controls
    config.camera.controls = readCameraControls(cap);
    
    // Save to config file
    if (AppConfig::saveToJSON(config_file, config)) {
        std::cout << "Saved current camera settings to " << config_file << std::endl;
        return true;
    } else {
        std::cerr << "Failed to save config to " << config_file << std::endl;
        return false;
    }
}

void applyCameraControls(cv::VideoCapture& cap, const AppConfig::Camera::Controls& controls) {
    if (!cap.isOpened()) {
        std::cerr << "Warning: Cannot apply camera controls - camera not opened" << std::endl;
        return;
    }
    
    // Auto exposure (must be set first, as it affects other exposure-related settings)
    if (controls.auto_exposure >= 0) {
        // CAP_PROP_AUTO_EXPOSURE: 0.25 = manual, 0.75 = auto
        double auto_exp_value = controls.auto_exposure > 0 ? 0.75 : 0.25;
        if (cap.set(cv::CAP_PROP_AUTO_EXPOSURE, auto_exp_value)) {
            std::cout << "  Set auto_exposure: " << (controls.auto_exposure > 0 ? "auto" : "manual") << std::endl;
        }
    }
    
    // Exposure (in milliseconds or relative value, depending on camera)
    if (controls.exposure >= 0) {
        if (cap.set(cv::CAP_PROP_EXPOSURE, controls.exposure)) {
            std::cout << "  Set exposure: " << controls.exposure << std::endl;
        } else {
            // Try alternative property (some cameras use different property IDs)
            cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // Manual mode
            if (cap.set(cv::CAP_PROP_EXPOSURE, controls.exposure)) {
                std::cout << "  Set exposure: " << controls.exposure << std::endl;
            }
        }
    }
    
    // Brightness (0.0-1.0 or absolute value)
    if (controls.brightness >= 0) {
        if (cap.set(cv::CAP_PROP_BRIGHTNESS, controls.brightness)) {
            std::cout << "  Set brightness: " << controls.brightness << std::endl;
        }
    }
    
    // Contrast (0.0-1.0 or absolute value)
    if (controls.contrast >= 0) {
        if (cap.set(cv::CAP_PROP_CONTRAST, controls.contrast)) {
            std::cout << "  Set contrast: " << controls.contrast << std::endl;
        }
    }
    
    // Saturation (0.0-1.0 or absolute value)
    if (controls.saturation >= 0) {
        if (cap.set(cv::CAP_PROP_SATURATION, controls.saturation)) {
            std::cout << "  Set saturation: " << controls.saturation << std::endl;
        }
    }
    
    // Gain
    if (controls.gain >= 0) {
        if (cap.set(cv::CAP_PROP_GAIN, controls.gain)) {
            std::cout << "  Set gain: " << controls.gain << std::endl;
        }
    }
    
    // Auto white balance (must be set before white balance value)
    if (controls.auto_white_balance >= 0) {
        // CAP_PROP_AUTO_WB: 0 = manual, 1 = auto
        if (cap.set(cv::CAP_PROP_AUTO_WB, controls.auto_white_balance > 0 ? 1.0 : 0.0)) {
            std::cout << "  Set auto_white_balance: " << (controls.auto_white_balance > 0 ? "auto" : "manual") << std::endl;
        }
    }
    
    // White balance (temperature in Kelvin or relative value)
    if (controls.white_balance >= 0 && controls.auto_white_balance != 1) {
        // Set auto WB to manual first if not already set
        if (controls.auto_white_balance < 0) {
            cap.set(cv::CAP_PROP_AUTO_WB, 0.0);
        }
        if (cap.set(cv::CAP_PROP_WB_TEMPERATURE, controls.white_balance)) {
            std::cout << "  Set white_balance: " << controls.white_balance << std::endl;
        } else {
            // Try alternative property
            if (cap.set(cv::CAP_PROP_WHITE_BALANCE_BLUE_U, controls.white_balance)) {
                std::cout << "  Set white_balance (blue): " << controls.white_balance << std::endl;
            }
        }
    }
    
    // Sharpness
    if (controls.sharpness >= 0) {
        if (cap.set(cv::CAP_PROP_SHARPNESS, controls.sharpness)) {
            std::cout << "  Set sharpness: " << controls.sharpness << std::endl;
        }
    }
    
    // Gamma
    if (controls.gamma >= 0) {
        if (cap.set(cv::CAP_PROP_GAMMA, controls.gamma)) {
            std::cout << "  Set gamma: " << controls.gamma << std::endl;
        }
    }
}

// ============================================================================
// Camera Detection Functions
// ============================================================================

std::vector<CameraInfo> detectV4L2Cameras() {
    std::vector<CameraInfo> cameras;
    
    // Check /dev/video* devices
    glob_t glob_result;
    if (glob("/dev/video*", GLOB_NOSORT, nullptr, &glob_result) == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            std::string device_path = glob_result.gl_pathv[i];
            
            // Try to open the device
            cv::VideoCapture test_cap;
            bool opened = test_cap.open(device_path, cv::CAP_V4L2);
            
            CameraInfo info;
            info.device_path = device_path;
            info.backend_name = "V4L2";
            info.is_available = opened;
            
            if (opened) {
                info.width = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_WIDTH));
                info.height = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                info.fps = test_cap.get(cv::CAP_PROP_FPS);
                
                // Extract index from path (e.g., "/dev/video0" -> 0)
                try {
                    std::string idx_str = device_path.substr(device_path.find_last_of("video") + 1);
                    info.index = std::stoi(idx_str);
                } catch (...) {
                    info.index = -1;
                }
                
                test_cap.release();
            } else {
                info.index = -1;
                info.width = 0;
                info.height = 0;
                info.fps = 0;
            }
            
            cameras.push_back(info);
        }
        globfree(&glob_result);
    }
    
    return cameras;
}

std::vector<CameraInfo> detectIndexCameras(int max_index, bool use_v4l2) {
    std::vector<CameraInfo> cameras;
    
    for (int i = 0; i <= max_index; ++i) {
        cv::VideoCapture test_cap;
        bool opened = test_cap.open(i, use_v4l2 ? cv::CAP_V4L2 : 0);
        
        CameraInfo info;
        info.index = i;
        info.device_path = "/dev/video" + std::to_string(i);
        info.backend_name = use_v4l2 ? "V4L2" : "Default";
        info.is_available = opened;
        
        if (opened) {
            info.width = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_WIDTH));
            info.height = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            info.fps = test_cap.get(cv::CAP_PROP_FPS);
            test_cap.release();
        } else {
            info.width = 0;
            info.height = 0;
            info.fps = 0;
        }
        
        cameras.push_back(info);
    }
    
    return cameras;
}

std::vector<CameraInfo> detectAvailableCameras(bool use_v4l2) {
    std::vector<CameraInfo> cameras;
    
    // First try V4L2 detection
    if (use_v4l2) {
        std::vector<CameraInfo> v4l2_cameras = detectV4L2Cameras();
        cameras.insert(cameras.end(), v4l2_cameras.begin(), v4l2_cameras.end());
    }
    
    // Also try index-based detection
    std::vector<CameraInfo> index_cameras = detectIndexCameras(15, use_v4l2);
    
    // Merge, avoiding duplicates
    for (const auto& idx_cam : index_cameras) {
        bool found = false;
        for (const auto& existing : cameras) {
            if (existing.index == idx_cam.index && existing.device_path == idx_cam.device_path) {
                found = true;
                break;
            }
        }
        if (!found && idx_cam.is_available) {
            cameras.push_back(idx_cam);
        }
    }
    
    // Sort by index
    std::sort(cameras.begin(), cameras.end(), 
              [](const CameraInfo& a, const CameraInfo& b) { return a.index < b.index; });
    
    return cameras;
}

CameraInfo getCameraInfo(int index, bool use_v4l2) {
    CameraInfo info;
    info.index = index;
    info.device_path = "/dev/video" + std::to_string(index);
    info.backend_name = use_v4l2 ? "V4L2" : "Default";
    
    cv::VideoCapture test_cap;
    info.is_available = test_cap.open(index, use_v4l2 ? cv::CAP_V4L2 : 0);
    
    if (info.is_available) {
        info.width = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        info.height = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        info.fps = test_cap.get(cv::CAP_PROP_FPS);
        test_cap.release();
    } else {
        info.width = 0;
        info.height = 0;
        info.fps = 0;
    }
    
    return info;
}

CameraInfo getCameraInfo(const std::string& device_path, bool use_v4l2) {
    CameraInfo info;
    info.device_path = device_path;
    info.backend_name = use_v4l2 ? "V4L2" : "Default";
    
    cv::VideoCapture test_cap;
    info.is_available = test_cap.open(device_path, use_v4l2 ? cv::CAP_V4L2 : 0);
    
    if (info.is_available) {
        info.index = -1;  // Will try to extract from path
        try {
            size_t pos = device_path.find_last_of("video");
            if (pos != std::string::npos && pos + 1 < device_path.length()) {
                std::string idx_str = device_path.substr(pos + 1);
                info.index = std::stoi(idx_str);
            }
        } catch (...) {
            info.index = -1;
        }
        
        info.width = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        info.height = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        info.fps = test_cap.get(cv::CAP_PROP_FPS);
        test_cap.release();
    } else {
        info.index = -1;
        info.width = 0;
        info.height = 0;
        info.fps = 0;
    }
    
    return info;
}

void printAvailableCameras(const std::vector<CameraInfo>& cameras) {
    std::cout << "\n=== Available Cameras ===" << std::endl;
    
    if (cameras.empty()) {
        std::cout << "No cameras detected." << std::endl;
        return;
    }
    
    int available_count = 0;
    for (const auto& cam : cameras) {
        if (cam.is_available) {
            available_count++;
            std::cout << "  [" << cam.index << "] " << cam.device_path 
                      << " (" << cam.backend_name << ")" << std::endl;
            std::cout << "      Resolution: " << cam.width << "x" << cam.height 
                      << " @ " << cam.fps << " FPS" << std::endl;
        }
    }
    
    std::cout << "\nTotal: " << available_count << " available camera(s) out of " 
              << cameras.size() << " checked" << std::endl;
}

// ============================================================================
// Camera Features/Capabilities Functions
// ============================================================================

CameraFeatures getCameraFeatures(cv::VideoCapture& cap) {
    CameraFeatures features;
    
    if (!cap.isOpened()) {
        return features;
    }
    
    // Check each property support
    features.supports_brightness = isPropertySupported(cap, cv::CAP_PROP_BRIGHTNESS);
    features.supports_contrast = isPropertySupported(cap, cv::CAP_PROP_CONTRAST);
    features.supports_saturation = isPropertySupported(cap, cv::CAP_PROP_SATURATION);
    features.supports_exposure = isPropertySupported(cap, cv::CAP_PROP_EXPOSURE);
    features.supports_gain = isPropertySupported(cap, cv::CAP_PROP_GAIN);
    features.supports_white_balance = isPropertySupported(cap, cv::CAP_PROP_WB_TEMPERATURE);
    features.supports_sharpness = isPropertySupported(cap, cv::CAP_PROP_SHARPNESS);
    features.supports_gamma = isPropertySupported(cap, cv::CAP_PROP_GAMMA);
    features.supports_auto_exposure = isPropertySupported(cap, cv::CAP_PROP_AUTO_EXPOSURE);
    features.supports_auto_white_balance = isPropertySupported(cap, cv::CAP_PROP_AUTO_WB);
    
    // Get property ranges where available
    if (features.supports_brightness) {
        getPropertyRange(cap, cv::CAP_PROP_BRIGHTNESS, 
                        features.brightness_min, features.brightness_max, 
                        features.brightness_min);  // step not used
    }
    if (features.supports_contrast) {
        getPropertyRange(cap, cv::CAP_PROP_CONTRAST, 
                        features.contrast_min, features.contrast_max, 
                        features.contrast_min);
    }
    if (features.supports_exposure) {
        getPropertyRange(cap, cv::CAP_PROP_EXPOSURE, 
                        features.exposure_min, features.exposure_max, 
                        features.exposure_min);
    }
    if (features.supports_gain) {
        getPropertyRange(cap, cv::CAP_PROP_GAIN, 
                        features.gain_min, features.gain_max, 
                        features.gain_min);
    }
    if (features.supports_white_balance) {
        getPropertyRange(cap, cv::CAP_PROP_WB_TEMPERATURE, 
                        features.white_balance_min, features.white_balance_max, 
                        features.white_balance_min);
    }
    
    return features;
}

bool isPropertySupported(cv::VideoCapture& cap, int property_id) {
    if (!cap.isOpened()) {
        return false;
    }
    
    double value = cap.get(property_id);
    // If property is not supported, OpenCV typically returns -1 or 0
    // But we need to check if setting it works
    double original = value;
    bool can_set = cap.set(property_id, value);
    if (can_set) {
        // Restore original value
        cap.set(property_id, original);
    }
    return can_set && value >= 0;
}

bool getPropertyRange(cv::VideoCapture& cap, int property_id, 
                      double& min_val, double& max_val, double& step) {
    if (!cap.isOpened()) {
        return false;
    }
    
    // OpenCV doesn't provide direct range queries, so we try to get current value
    // and test setting different values
    double current = cap.get(property_id);
    if (current < 0) {
        return false;
    }
    
    // Try to find min/max by testing values
    // This is a simplified approach - actual implementation might need
    // backend-specific queries (e.g., v4l2-ctl)
    min_val = 0;
    max_val = current * 2;  // Estimate
    step = 1.0;
    
    return true;
}

void printCameraFeatures(const CameraFeatures& features) {
    std::cout << "\n=== Camera Features/Capabilities ===" << std::endl;
    
    std::cout << "Supported Controls:" << std::endl;
    std::cout << "  Brightness:        " << (features.supports_brightness ? "Yes" : "No");
    if (features.supports_brightness && features.brightness_max > 0) {
        std::cout << " (range: " << features.brightness_min << " - " << features.brightness_max << ")";
    }
    std::cout << std::endl;
    
    std::cout << "  Contrast:          " << (features.supports_contrast ? "Yes" : "No");
    if (features.supports_contrast && features.contrast_max > 0) {
        std::cout << " (range: " << features.contrast_min << " - " << features.contrast_max << ")";
    }
    std::cout << std::endl;
    
    std::cout << "  Saturation:        " << (features.supports_saturation ? "Yes" : "No") << std::endl;
    std::cout << "  Exposure:          " << (features.supports_exposure ? "Yes" : "No");
    if (features.supports_exposure && features.exposure_max > 0) {
        std::cout << " (range: " << features.exposure_min << " - " << features.exposure_max << ")";
    }
    std::cout << std::endl;
    
    std::cout << "  Gain:              " << (features.supports_gain ? "Yes" : "No");
    if (features.supports_gain && features.gain_max > 0) {
        std::cout << " (range: " << features.gain_min << " - " << features.gain_max << ")";
    }
    std::cout << std::endl;
    
    std::cout << "  White Balance:     " << (features.supports_white_balance ? "Yes" : "No");
    if (features.supports_white_balance && features.white_balance_max > 0) {
        std::cout << " (range: " << features.white_balance_min << " - " << features.white_balance_max << " K)";
    }
    std::cout << std::endl;
    
    std::cout << "  Sharpness:         " << (features.supports_sharpness ? "Yes" : "No") << std::endl;
    std::cout << "  Gamma:             " << (features.supports_gamma ? "Yes" : "No") << std::endl;
    std::cout << "  Auto Exposure:     " << (features.supports_auto_exposure ? "Yes" : "No") << std::endl;
    std::cout << "  Auto White Balance: " << (features.supports_auto_white_balance ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Camera Settings Functions (Enhanced)
// ============================================================================

bool writeCameraControls(cv::VideoCapture& cap, const AppConfig::Camera::Controls& controls) {
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot write camera controls - camera not opened" << std::endl;
        return false;
    }
    
    applyCameraControls(cap, controls);
    return true;
}

bool loadAndApplyCameraControls(cv::VideoCapture& cap, const std::string& config_file) {
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot apply camera controls - camera not opened" << std::endl;
        return false;
    }
    
    AppConfig config;
    if (!AppConfig::loadFromJSON(config_file, config)) {
        std::cerr << "Error: Cannot load config from " << config_file << std::endl;
        return false;
    }
    
    applyCameraControls(cap, config.camera.controls);
    return true;
}

// ============================================================================
// Camera Opening Helper Functions
// ============================================================================

bool openCamera(cv::VideoCapture& cap, const std::string& device_or_index, 
                bool use_v4l2, int width, int height, double fps) {
    // Try to determine if it's a device path or index
    bool is_device_path = (device_or_index.find("/dev/video") == 0);
    
    bool opened = false;
    if (is_device_path) {
        opened = cap.open(device_or_index, use_v4l2 ? cv::CAP_V4L2 : 0);
    } else {
        try {
            int index = std::stoi(device_or_index);
            opened = cap.open(index, use_v4l2 ? cv::CAP_V4L2 : 0);
        } catch (...) {
            // Invalid index, try as device path anyway
            opened = cap.open(device_or_index, use_v4l2 ? cv::CAP_V4L2 : 0);
        }
    }
    
    if (!opened) {
        // Fallback: try default backend
        if (is_device_path) {
            opened = cap.open(device_or_index);
        } else {
            try {
                int index = std::stoi(device_or_index);
                opened = cap.open(index);
            } catch (...) {
                opened = false;
            }
        }
    }
    
    if (opened) {
        // Set resolution and FPS if specified
        if (width > 0 && height > 0) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        }
        if (fps > 0) {
            cap.set(cv::CAP_PROP_FPS, fps);
        }
    }
    
    return opened;
}

bool openCameraFromConfig(cv::VideoCapture& cap, const AppConfig& config) {
    return openCamera(cap, config.camera.device, config.camera.use_v4l2,
                     config.camera.width, config.camera.height, config.camera.fps);
}

bool testCamera(int index, bool use_v4l2) {
    cv::VideoCapture test_cap;
    bool opened = test_cap.open(index, use_v4l2 ? cv::CAP_V4L2 : 0);
    if (opened) {
        test_cap.release();
    }
    return opened;
}

bool testCamera(const std::string& device_path, bool use_v4l2) {
    cv::VideoCapture test_cap;
    bool opened = test_cap.open(device_path, use_v4l2 ? cv::CAP_V4L2 : 0);
    if (opened) {
        test_cap.release();
    }
    return opened;
}

