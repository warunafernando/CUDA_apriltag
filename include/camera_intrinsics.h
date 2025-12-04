#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "image_preprocessor.h"

// Utility functions for CameraIntrinsics JSON save/load
namespace CameraIntrinsicsUtils {
    bool loadFromJSON(const std::string& filename, CameraIntrinsics& intr);
    bool saveToJSON(const std::string& filename, const CameraIntrinsics& intr);
    cv::Mat getCameraMatrix(const CameraIntrinsics& intr);
    cv::Mat getDistortionCoeffs(const CameraIntrinsics& intr);
    CameraIntrinsics fromOpenCV(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
}
