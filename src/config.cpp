#include "config.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace {

// Simple JSON value extractor
float extractFloat(const std::string& content, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return 0.0f;
    
    pos = content.find(":", pos);
    if (pos == std::string::npos) return 0.0f;
    pos++;
    
    while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
    
    size_t end = pos;
    while (end < content.length() && 
           (content[end] == '.' || content[end] == '-' || 
            (content[end] >= '0' && content[end] <= '9') || 
            content[end] == 'e' || content[end] == 'E' || content[end] == '+' || content[end] == '-')) {
        end++;
    }
    
    if (end > pos) {
        return std::stof(content.substr(pos, end - pos));
    }
    return 0.0f;
}

int extractInt(const std::string& content, const std::string& key) {
    return static_cast<int>(extractFloat(content, key));
}

bool extractBool(const std::string& content, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return false;
    
    pos = content.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    
    while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
    
    if (pos < content.length() && content[pos] == 't') {
        // Check for "true"
        if (content.substr(pos, 4) == "true") return true;
    }
    return false;
}

std::string extractString(const std::string& content, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return "";
    
    pos = content.find(":", pos);
    if (pos == std::string::npos) return "";
    pos++;
    
    while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '"')) pos++;
    
    size_t end = pos;
    while (end < content.length() && content[end] != '"' && content[end] != ',' && content[end] != '}') end++;
    
    if (end > pos) {
        return content.substr(pos, end - pos);
    }
    return "";
}

}  // namespace

bool AppConfig::loadFromJSON(const std::string& filename, AppConfig& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    std::string content;
    while (std::getline(file, line)) {
        content += line;
    }
    file.close();
    
    // Load camera settings
    config.camera.width = extractInt(content, "camera_width");
    if (config.camera.width == 0) config.camera.width = 1280;
    
    config.camera.height = extractInt(content, "camera_height");
    if (config.camera.height == 0) config.camera.height = 720;
    
    config.camera.fps = extractInt(content, "camera_fps");
    if (config.camera.fps == 0) config.camera.fps = 120;
    
    config.camera.decimation = extractInt(content, "camera_decimation");
    if (config.camera.decimation == 0) config.camera.decimation = 2;
    
    std::string device = extractString(content, "camera_device");
    if (!device.empty()) config.camera.device = device;
    
    config.camera.use_v4l2 = extractBool(content, "camera_use_v4l2");
    
    // Load camera controls (-1 means use default/auto)
    float brightness = extractFloat(content, "camera_brightness");
    config.camera.controls.brightness = (brightness == 0.0f && content.find("camera_brightness") == std::string::npos) ? -1.0 : brightness;
    
    float contrast = extractFloat(content, "camera_contrast");
    config.camera.controls.contrast = (contrast == 0.0f && content.find("camera_contrast") == std::string::npos) ? -1.0 : contrast;
    
    float saturation = extractFloat(content, "camera_saturation");
    config.camera.controls.saturation = (saturation == 0.0f && content.find("camera_saturation") == std::string::npos) ? -1.0 : saturation;
    
    float exposure = extractFloat(content, "camera_exposure");
    config.camera.controls.exposure = (exposure == 0.0f && content.find("camera_exposure") == std::string::npos) ? -1.0 : exposure;
    
    float gain = extractFloat(content, "camera_gain");
    config.camera.controls.gain = (gain == 0.0f && content.find("camera_gain") == std::string::npos) ? -1.0 : gain;
    
    float white_balance = extractFloat(content, "camera_white_balance");
    config.camera.controls.white_balance = (white_balance == 0.0f && content.find("camera_white_balance") == std::string::npos) ? -1.0 : white_balance;
    
    float sharpness = extractFloat(content, "camera_sharpness");
    config.camera.controls.sharpness = (sharpness == 0.0f && content.find("camera_sharpness") == std::string::npos) ? -1.0 : sharpness;
    
    float gamma = extractFloat(content, "camera_gamma");
    config.camera.controls.gamma = (gamma == 0.0f && content.find("camera_gamma") == std::string::npos) ? -1.0 : gamma;
    
    int auto_exposure = extractInt(content, "camera_auto_exposure");
    config.camera.controls.auto_exposure = (auto_exposure == 0 && content.find("camera_auto_exposure") == std::string::npos) ? -1 : auto_exposure;
    
    int auto_white_balance = extractInt(content, "camera_auto_white_balance");
    config.camera.controls.auto_white_balance = (auto_white_balance == 0 && content.find("camera_auto_white_balance") == std::string::npos) ? -1 : auto_white_balance;
    
    // Load tag settings
    config.tag.size_m = extractFloat(content, "tag_size_m");
    if (config.tag.size_m == 0.0f) config.tag.size_m = 0.165f;
    
    // Load detection settings
    config.detection.min_quality = extractFloat(content, "detection_min_quality");
    if (config.detection.min_quality == 0.0f) config.detection.min_quality = 0.01f;
    
    config.detection.max_reprojection_error = extractFloat(content, "detection_max_reprojection_error");
    if (config.detection.max_reprojection_error == 0.0f) config.detection.max_reprojection_error = 500.0f;
    
    config.detection.min_decision_margin = extractFloat(content, "detection_min_decision_margin");
    if (config.detection.min_decision_margin == 0.0f) config.detection.min_decision_margin = 0.05f;
    
    config.detection.enable_subpixel_refinement = extractBool(content, "detection_enable_subpixel_refinement");
    config.detection.enable_temporal_filtering = extractBool(content, "detection_enable_temporal_filtering");
    config.detection.use_gpu_quad_extraction = extractBool(content, "detection_use_gpu_quad_extraction");
    
    // Load ROI settings
    config.roi.full_frame_interval = extractInt(content, "roi_full_frame_interval");
    if (config.roi.full_frame_interval == 0) config.roi.full_frame_interval = 30;
    
    config.roi.decay_frames = extractInt(content, "roi_decay_frames");
    if (config.roi.decay_frames == 0) config.roi.decay_frames = 20;
    
    config.roi.max_rois = extractInt(content, "roi_max_rois");
    if (config.roi.max_rois == 0) config.roi.max_rois = 16;
    
    // Load test settings
    config.test.duration_seconds = extractFloat(content, "test_duration_seconds");
    if (config.test.duration_seconds == 0.0f) config.test.duration_seconds = 60.0f;
    
    config.test.max_captures = extractInt(content, "test_max_captures");
    if (config.test.max_captures == 0) config.test.max_captures = 5;
    
    std::string capture_dir = extractString(content, "test_capture_dir");
    if (!capture_dir.empty()) config.test.capture_dir = capture_dir;
    
    // Load file paths
    std::string intrinsics_file = extractString(content, "files_intrinsics");
    if (!intrinsics_file.empty()) config.files.intrinsics = intrinsics_file;
    
    // Initialize default intrinsics
    config.initDefaultIntrinsics();
    
    return true;
}

bool AppConfig::saveToJSON(const std::string& filename, const AppConfig& config) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << std::fixed << std::setprecision(6);
    file << "{\n";
    file << "  \"camera_width\": " << config.camera.width << ",\n";
    file << "  \"camera_height\": " << config.camera.height << ",\n";
    file << "  \"camera_fps\": " << config.camera.fps << ",\n";
    file << "  \"camera_decimation\": " << config.camera.decimation << ",\n";
    file << "  \"camera_device\": \"" << config.camera.device << "\",\n";
    file << "  \"camera_use_v4l2\": " << (config.camera.use_v4l2 ? "true" : "false") << ",\n";
    file << "  \"camera_brightness\": " << config.camera.controls.brightness << ",\n";
    file << "  \"camera_contrast\": " << config.camera.controls.contrast << ",\n";
    file << "  \"camera_saturation\": " << config.camera.controls.saturation << ",\n";
    file << "  \"camera_exposure\": " << config.camera.controls.exposure << ",\n";
    file << "  \"camera_gain\": " << config.camera.controls.gain << ",\n";
    file << "  \"camera_white_balance\": " << config.camera.controls.white_balance << ",\n";
    file << "  \"camera_sharpness\": " << config.camera.controls.sharpness << ",\n";
    file << "  \"camera_gamma\": " << config.camera.controls.gamma << ",\n";
    file << "  \"camera_auto_exposure\": " << config.camera.controls.auto_exposure << ",\n";
    file << "  \"camera_auto_white_balance\": " << config.camera.controls.auto_white_balance << ",\n";
    file << "  \"tag_size_m\": " << config.tag.size_m << ",\n";
    file << "  \"detection_min_quality\": " << config.detection.min_quality << ",\n";
    file << "  \"detection_max_reprojection_error\": " << config.detection.max_reprojection_error << ",\n";
    file << "  \"detection_min_decision_margin\": " << config.detection.min_decision_margin << ",\n";
    file << "  \"detection_enable_subpixel_refinement\": " << (config.detection.enable_subpixel_refinement ? "true" : "false") << ",\n";
    file << "  \"detection_enable_temporal_filtering\": " << (config.detection.enable_temporal_filtering ? "true" : "false") << ",\n";
    file << "  \"detection_use_gpu_quad_extraction\": " << (config.detection.use_gpu_quad_extraction ? "true" : "false") << ",\n";
    file << "  \"roi_full_frame_interval\": " << config.roi.full_frame_interval << ",\n";
    file << "  \"roi_decay_frames\": " << config.roi.decay_frames << ",\n";
    file << "  \"roi_max_rois\": " << config.roi.max_rois << ",\n";
    file << "  \"test_duration_seconds\": " << config.test.duration_seconds << ",\n";
    file << "  \"test_max_captures\": " << config.test.max_captures << ",\n";
    file << "  \"test_capture_dir\": \"" << config.test.capture_dir << "\",\n";
    file << "  \"files_intrinsics\": \"" << config.files.intrinsics << "\"\n";
    file << "}\n";
    
    file.close();
    return true;
}

