#include "gpu_context.h"
#include "image_preprocessor.h"
#include "apriltag_gpu.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <sstream>

namespace py = pybind11;

// Camera calibration helper class
class CameraCalibration {
public:
    static bool calibrateFromImages(const std::vector<std::string>& image_paths,
                                    int board_width, int board_height,
                                    float square_size,
                                    std::string output_file) {
        std::vector<std::vector<cv::Point3f>> object_points;
        std::vector<std::vector<cv::Point2f>> image_points;
        
        // Prepare object points (0,0,0), (1,0,0), (2,0,0) ... etc
        std::vector<cv::Point3f> objp;
        for (int i = 0; i < board_height; i++) {
            for (int j = 0; j < board_width; j++) {
                objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
            }
        }
        
        cv::Size image_size;
        bool found_size = false;
        
        // Find chessboard corners in all images
        for (const auto& path : image_paths) {
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Failed to load image: " << path << std::endl;
                continue;
            }
            
            if (!found_size) {
                image_size = img.size();
                found_size = true;
            }
            
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(img, cv::Size(board_width, board_height), corners);
            
            if (found) {
                // Refine corners
                cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
                object_points.push_back(objp);
                image_points.push_back(corners);
            } else {
                std::cerr << "Chessboard not found in: " << path << std::endl;
            }
        }
        
        if (object_points.empty()) {
            std::cerr << "No valid calibration images found!" << std::endl;
            return false;
        }
        
        // Calibrate camera
        cv::Mat camera_matrix, dist_coeffs;
        std::vector<cv::Mat> rvecs, tvecs;
        
        double rms = cv::calibrateCamera(object_points, image_points, image_size,
                                         camera_matrix, dist_coeffs, rvecs, tvecs);
        
        std::cout << "Calibration RMS error: " << rms << " pixels" << std::endl;
        
        // Save calibration to file
        cv::FileStorage fs(output_file, cv::FileStorage::WRITE);
        if (fs.isOpened()) {
            fs << "camera_matrix" << camera_matrix;
            fs << "distortion_coefficients" << dist_coeffs;
            fs << "image_width" << image_size.width;
            fs << "image_height" << image_size.height;
            fs << "rms_error" << rms;
            fs.release();
            std::cout << "Calibration saved to: " << output_file << std::endl;
            return true;
        }
        
        return false;
    }
    
    static bool loadCalibration(const std::string& calibration_file,
                                float& fx, float& fy, float& cx, float& cy,
                                float& k1, float& k2, float& p1, float& p2, float& k3,
                                int& width, int& height) {
        cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open calibration file: " << calibration_file << std::endl;
            return false;
        }
        
        cv::Mat camera_matrix, dist_coeffs;
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs["image_width"] >> width;
        fs["image_height"] >> height;
        fs.release();
        
        if (camera_matrix.empty() || dist_coeffs.empty()) {
            std::cerr << "Invalid calibration data in file" << std::endl;
            return false;
        }
        
        fx = static_cast<float>(camera_matrix.at<double>(0, 0));
        fy = static_cast<float>(camera_matrix.at<double>(1, 1));
        cx = static_cast<float>(camera_matrix.at<double>(0, 2));
        cy = static_cast<float>(camera_matrix.at<double>(1, 2));
        
        k1 = dist_coeffs.rows > 0 ? static_cast<float>(dist_coeffs.at<double>(0)) : 0.0f;
        k2 = dist_coeffs.rows > 1 ? static_cast<float>(dist_coeffs.at<double>(1)) : 0.0f;
        p1 = dist_coeffs.rows > 2 ? static_cast<float>(dist_coeffs.at<double>(2)) : 0.0f;
        p2 = dist_coeffs.rows > 3 ? static_cast<float>(dist_coeffs.at<double>(3)) : 0.0f;
        k3 = dist_coeffs.rows > 4 ? static_cast<float>(dist_coeffs.at<double>(4)) : 0.0f;
        
        return true;
    }
};

class PyCudaAprilTag {
public:
    PyCudaAprilTag(int width,
                   int height,
                   int decimation,
                   float fx, float fy, float cx, float cy,
                   float tag_size_m,
                   float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f, float k3 = 0.0f)
        : ctx_(0),
          tag_size_m_(tag_size_m) {
        // Set camera intrinsics with distortion
        CameraIntrinsics intr;
        intr.fx = fx;
        intr.fy = fy;
        intr.cx = cx;
        intr.cy = cy;
        intr.k1 = k1;
        intr.k2 = k2;
        intr.p1 = p1;
        intr.p2 = p2;
        intr.k3 = k3;
        
        // Create preprocessor with intrinsics
        pre_ = std::make_unique<ImagePreprocessor>(ctx_, width, height, decimation, &intr);
        
        cv::Matx33f K(fx, 0.f, cx,
                      0.f, fy, cy,
                      0.f, 0.f, 1.f);
        det_ = std::make_unique<AprilTagGpuDetector>(
            ctx_, pre_->workingWidth(), pre_->workingHeight(),
            tag_size_m, K);
    }
    
    // Delete copy constructor and assignment to prevent copying
    PyCudaAprilTag(const PyCudaAprilTag&) = delete;
    PyCudaAprilTag& operator=(const PyCudaAprilTag&) = delete;

    py::list detect(py::array_t<uint8_t> frame) {
        py::buffer_info info = frame.request();
        if (info.ndim != 2 && info.ndim != 3) {
            throw std::runtime_error("Expected 2D (gray) or 3D (BGR) array");
        }
        int h = static_cast<int>(info.shape[0]);
        int w = static_cast<int>(info.shape[1]);

        cv::Mat cv_frame;
        if (info.ndim == 2) {
            cv_frame = cv::Mat(h, w, CV_8UC1, info.ptr);
        } else {
            int c = static_cast<int>(info.shape[2]);
            if (c != 3) {
                throw std::runtime_error("Expected 3-channel BGR image");
            }
            cv_frame = cv::Mat(h, w, CV_8UC3, info.ptr);
        }

        unsigned char* d_gray = pre_->preprocess(cv_frame);
        auto dets = det_->detect(d_gray);

        py::list out;
        for (const auto& d : dets) {
            py::dict item;
            item["id"] = d.id;
            item["decision_margin"] = d.decision_margin;
            item["hamming_distance"] = d.hamming_distance;
            item["reprojection_error"] = d.reprojection_error;
            item["edge_strength"] = d.edge_strength;
            item["quality"] = d.quality();
            
            py::list corners;
            for (int i = 0; i < 4; ++i) {
                corners.append(py::make_tuple(d.corners[i].x, d.corners[i].y));
            }
            item["corners"] = corners;
            
            // Pose information
            py::dict pose;
            pose["quat_w"] = d.quat_w;
            pose["quat_x"] = d.quat_x;
            pose["quat_y"] = d.quat_y;
            pose["quat_z"] = d.quat_z;
            
            // Extract translation from transformation matrix
            pose["tx"] = d.T_cam_tag(0, 3);
            pose["ty"] = d.T_cam_tag(1, 3);
            pose["tz"] = d.T_cam_tag(2, 3);
            
            item["pose"] = pose;
            
            out.append(item);
        }
        return out;
    }
    
    // Get camera intrinsics (stored during construction)
    py::dict getIntrinsics() {
        py::dict intrinsics;
        // Return the intrinsics that were set during construction
        // These are stored in the ImagePreprocessor but not directly exposed
        // For now, return empty dict - user should keep track of their calibration
        return intrinsics;
    }
    
    // Set camera intrinsics (update preprocessor)
    void setIntrinsics(float fx, float fy, float cx, float cy,
                      float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f, float k3 = 0.0f) {
        CameraIntrinsics intr;
        intr.fx = fx;
        intr.fy = fy;
        intr.cx = cx;
        intr.cy = cy;
        intr.k1 = k1;
        intr.k2 = k2;
        intr.p1 = p1;
        intr.p2 = p2;
        intr.k3 = k3;
        
        // Recreate preprocessor with new intrinsics
        int width = pre_->inputWidth();
        int height = pre_->inputHeight();
        int decimation = width / pre_->workingWidth();
        pre_ = std::make_unique<ImagePreprocessor>(ctx_, width, height, decimation, &intr);
        
        // Update detector with new camera matrix
        cv::Matx33f K(fx, 0.f, cx,
                      0.f, fy, cy,
                      0.f, 0.f, 1.f);
        det_ = std::make_unique<AprilTagGpuDetector>(
            ctx_, pre_->workingWidth(), pre_->workingHeight(),
            tag_size_m_, K);
    }
    
private:
    GpuContext ctx_;
    std::unique_ptr<ImagePreprocessor> pre_;  // Use pointer to allow reassignment
    std::unique_ptr<AprilTagGpuDetector> det_;
    float tag_size_m_;  // Store tag size for setIntrinsics
};

// Factory function to create detector from calibration file
std::unique_ptr<PyCudaAprilTag> createFromCalibrationFile(const std::string& calibration_file,
                                                           int width, int height,
                                                           int decimation,
                                                           float tag_size_m) {
    float fx, fy, cx, cy, k1, k2, p1, p2, k3;
    int calib_width, calib_height;
    
    if (!CameraCalibration::loadCalibration(calibration_file, fx, fy, cx, cy,
                                            k1, k2, p1, p2, k3,
                                            calib_width, calib_height)) {
        throw std::runtime_error("Failed to load calibration file: " + calibration_file);
    }
    
    // Use calibration dimensions if not specified
    if (width <= 0) width = calib_width;
    if (height <= 0) height = calib_height;
    
    return std::make_unique<PyCudaAprilTag>(width, height, decimation, fx, fy, cx, cy, tag_size_m,
                                           k1, k2, p1, p2, k3);
}

PYBIND11_MODULE(cuda_apriltag_py, m) {
    m.doc() = "GPU-accelerated AprilTag detection with camera calibration support";
    
    // Camera calibration functions
    m.def("calibrate_camera", &CameraCalibration::calibrateFromImages,
          "Calibrate camera from chessboard images",
          py::arg("image_paths"), py::arg("board_width"), py::arg("board_height"),
          py::arg("square_size"), py::arg("output_file"));
    
    m.def("load_calibration", &CameraCalibration::loadCalibration,
          "Load camera calibration from file",
          py::arg("calibration_file"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("k1"), py::arg("k2"), py::arg("p1"), py::arg("p2"), py::arg("k3"),
          py::arg("width"), py::arg("height"));
    
    m.def("create_from_calibration_file", &createFromCalibrationFile,
          "Create detector from calibration file",
          py::arg("calibration_file"), py::arg("width") = 0, py::arg("height") = 0,
          py::arg("decimation") = 2, py::arg("tag_size_m") = 0.165f);
    
    // PyCudaAprilTag class
    py::class_<PyCudaAprilTag>(m, "CudaAprilTag")
        .def(py::init<int, int, int, float, float, float, float, float,
                      float, float, float, float, float>(),
             "Initialize detector with camera intrinsics and distortion",
             py::arg("width"), py::arg("height"), py::arg("decimation"),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
             py::arg("tag_size_m"),
             py::arg("k1") = 0.0f, py::arg("k2") = 0.0f,
             py::arg("p1") = 0.0f, py::arg("p2") = 0.0f, py::arg("k3") = 0.0f)
        .def("detect", &PyCudaAprilTag::detect,
             "Detect AprilTags in image",
             py::arg("frame"))
        .def("get_intrinsics", &PyCudaAprilTag::getIntrinsics,
             "Get camera intrinsics")
        .def("set_intrinsics", &PyCudaAprilTag::setIntrinsics,
             "Update camera intrinsics and distortion coefficients",
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
             py::arg("k1") = 0.0f, py::arg("k2") = 0.0f,
             py::arg("p1") = 0.0f, py::arg("p2") = 0.0f, py::arg("k3") = 0.0f);
}
