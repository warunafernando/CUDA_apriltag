#include "camera_intrinsics.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/core.hpp>

bool CameraIntrinsicsUtils::loadFromJSON(const std::string& filename, CameraIntrinsics& intr) {
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
    
    // Simple JSON parsing (for basic format)
    // Expected format: {"fx": 1000.0, "fy": 1000.0, ...}
    // For more robust parsing, consider using a JSON library
    
    // Try to extract values using string operations
    auto extractFloat = [&content](const std::string& key) -> float {
        std::string search = "\"" + key + "\"";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return 0.0f;
        
        pos = content.find(":", pos);
        if (pos == std::string::npos) return 0.0f;
        pos++; // Skip ':'
        
        // Skip whitespace
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        
        // Extract number
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
    };
    
    intr.fx = extractFloat("fx");
    intr.fy = extractFloat("fy");
    intr.cx = extractFloat("cx");
    intr.cy = extractFloat("cy");
    intr.k1 = extractFloat("k1");
    intr.k2 = extractFloat("k2");
    intr.p1 = extractFloat("p1");
    intr.p2 = extractFloat("p2");
    intr.k3 = extractFloat("k3");
    
    return true;
}

bool CameraIntrinsicsUtils::saveToJSON(const std::string& filename, const CameraIntrinsics& intr) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << std::fixed << std::setprecision(6);
    file << "{\n";
    file << "  \"fx\": " << intr.fx << ",\n";
    file << "  \"fy\": " << intr.fy << ",\n";
    file << "  \"cx\": " << intr.cx << ",\n";
    file << "  \"cy\": " << intr.cy << ",\n";
    file << "  \"k1\": " << intr.k1 << ",\n";
    file << "  \"k2\": " << intr.k2 << ",\n";
    file << "  \"p1\": " << intr.p1 << ",\n";
    file << "  \"p2\": " << intr.p2 << ",\n";
    file << "  \"k3\": " << intr.k3 << "\n";
    file << "}\n";
    
    file.close();
    return true;
}

cv::Mat CameraIntrinsicsUtils::getCameraMatrix(const CameraIntrinsics& intr) {
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        static_cast<double>(intr.fx), 0.0, static_cast<double>(intr.cx),
        0.0, static_cast<double>(intr.fy), static_cast<double>(intr.cy),
        0.0, 0.0, 1.0);
    return K;
}

cv::Mat CameraIntrinsicsUtils::getDistortionCoeffs(const CameraIntrinsics& intr) {
    cv::Mat dist = (cv::Mat_<double>(5, 1) <<
        static_cast<double>(intr.k1),
        static_cast<double>(intr.k2),
        static_cast<double>(intr.p1),
        static_cast<double>(intr.p2),
        static_cast<double>(intr.k3));
    return dist;
}

CameraIntrinsics CameraIntrinsicsUtils::fromOpenCV(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    CameraIntrinsics intr;
    
    if (camera_matrix.rows == 3 && camera_matrix.cols == 3) {
        intr.fx = static_cast<float>(camera_matrix.at<double>(0, 0));
        intr.fy = static_cast<float>(camera_matrix.at<double>(1, 1));
        intr.cx = static_cast<float>(camera_matrix.at<double>(0, 2));
        intr.cy = static_cast<float>(camera_matrix.at<double>(1, 2));
    }
    
    if (dist_coeffs.rows >= 1) {
        intr.k1 = dist_coeffs.rows > 0 ? static_cast<float>(dist_coeffs.at<double>(0)) : 0.0f;
        intr.k2 = dist_coeffs.rows > 1 ? static_cast<float>(dist_coeffs.at<double>(1)) : 0.0f;
        intr.p1 = dist_coeffs.rows > 2 ? static_cast<float>(dist_coeffs.at<double>(2)) : 0.0f;
        intr.p2 = dist_coeffs.rows > 3 ? static_cast<float>(dist_coeffs.at<double>(3)) : 0.0f;
        intr.k3 = dist_coeffs.rows > 4 ? static_cast<float>(dist_coeffs.at<double>(4)) : 0.0f;
    }
    
    return intr;
}

