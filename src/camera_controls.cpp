#include "camera_controls.h"
#include <iostream>
#include <iomanip>

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

