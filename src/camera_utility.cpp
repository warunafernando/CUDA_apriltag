#include "camera_controls.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <string>

// Camera utility program for detecting cameras, reading features, and managing settings

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [command] [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  detect              - Detect and list all available cameras\n";
    std::cout << "  info <index|path>   - Get detailed info about a specific camera\n";
    std::cout << "  features <index|path> - Get camera features/capabilities\n";
    std::cout << "  read <index|path>   - Read current camera settings\n";
    std::cout << "  write <index|path>  - Write camera settings from config.json\n";
    std::cout << "  save <index|path>   - Read and save camera settings to config.json\n";
    std::cout << "  test <index|path>   - Test if camera can be opened\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --v4l2             - Use V4L2 backend (default: true)\n";
    std::cout << "  --no-v4l2          - Don't use V4L2 backend\n";
    std::cout << "  --config <file>    - Config file path (default: config.json)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " detect\n";
    std::cout << "  " << program_name << " info 0\n";
    std::cout << "  " << program_name << " features /dev/video0\n";
    std::cout << "  " << program_name << " read 0\n";
    std::cout << "  " << program_name << " save 0\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string command = argv[1];
    bool use_v4l2 = true;
    std::string config_file = "config.json";
    
    // Parse options
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--v4l2") {
            use_v4l2 = true;
        } else if (arg == "--no-v4l2") {
            use_v4l2 = false;
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
    }
    
    // Handle commands that don't need a camera
    if (command == "detect") {
        std::cout << "Detecting available cameras..." << std::endl;
        std::vector<CameraInfo> cameras = detectAvailableCameras(use_v4l2);
        printAvailableCameras(cameras);
        return 0;
    }
    
    // Commands that need a camera device/index
    if (argc < 3) {
        std::cerr << "Error: " << command << " requires a camera index or device path" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    std::string device_or_index = argv[2];
    cv::VideoCapture cap;
    
    // Open camera
    bool opened = openCamera(cap, device_or_index, use_v4l2);
    if (!opened) {
        std::cerr << "Error: Failed to open camera: " << device_or_index << std::endl;
        return 1;
    }
    
    std::cout << "Successfully opened camera: " << device_or_index << std::endl;
    
    if (command == "info") {
        CameraInfo info;
        if (device_or_index.find("/dev/video") == 0) {
            info = getCameraInfo(device_or_index, use_v4l2);
        } else {
            try {
                int index = std::stoi(device_or_index);
                info = getCameraInfo(index, use_v4l2);
            } catch (...) {
                std::cerr << "Error: Invalid camera index: " << device_or_index << std::endl;
                cap.release();
                return 1;
            }
        }
        
        std::cout << "\n=== Camera Information ===" << std::endl;
        std::cout << "Index:        " << info.index << std::endl;
        std::cout << "Device Path:  " << info.device_path << std::endl;
        std::cout << "Backend:      " << info.backend_name << std::endl;
        std::cout << "Available:    " << (info.is_available ? "Yes" : "No") << std::endl;
        std::cout << "Resolution:   " << info.width << "x" << info.height << std::endl;
        std::cout << "FPS:          " << info.fps << std::endl;
        
    } else if (command == "features") {
        CameraFeatures features = getCameraFeatures(cap);
        printCameraFeatures(features);
        
    } else if (command == "read") {
        printCameraControls(cap);
        
    } else if (command == "write") {
        std::cout << "Loading camera settings from " << config_file << "..." << std::endl;
        if (loadAndApplyCameraControls(cap, config_file)) {
            std::cout << "Successfully applied camera settings from config." << std::endl;
        } else {
            std::cerr << "Failed to apply camera settings from config." << std::endl;
            cap.release();
            return 1;
        }
        
    } else if (command == "save") {
        std::cout << "Reading current camera settings..." << std::endl;
        if (saveCameraControlsToConfig(cap, config_file)) {
            std::cout << "Successfully saved camera settings to " << config_file << std::endl;
        } else {
            std::cerr << "Failed to save camera settings." << std::endl;
            cap.release();
            return 1;
        }
        
    } else if (command == "test") {
        bool can_open = false;
        if (device_or_index.find("/dev/video") == 0) {
            can_open = testCamera(device_or_index, use_v4l2);
        } else {
            try {
                int index = std::stoi(device_or_index);
                can_open = testCamera(index, use_v4l2);
            } catch (...) {
                std::cerr << "Error: Invalid camera index: " << device_or_index << std::endl;
                return 1;
            }
        }
        
        if (can_open) {
            std::cout << "Camera " << device_or_index << " can be opened successfully." << std::endl;
        } else {
            std::cout << "Camera " << device_or_index << " cannot be opened." << std::endl;
            return 1;
        }
        
    } else {
        std::cerr << "Error: Unknown command: " << command << std::endl;
        printUsage(argv[0]);
        cap.release();
        return 1;
    }
    
    cap.release();
    return 0;
}

