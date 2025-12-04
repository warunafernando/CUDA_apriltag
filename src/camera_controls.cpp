#include "camera_controls.h"
#include <iostream>

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

