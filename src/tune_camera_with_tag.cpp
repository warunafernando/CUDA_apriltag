#include "gpu_context.h"
#include "image_preprocessor.h"
#include "apriltag_gpu.h"
#include "camera_intrinsics.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Camera tuning using AprilTag
// This program allows you to tune camera intrinsics by:
// 1. Detecting an AprilTag in the camera feed
// 2. Using the known tag size and detected corners to estimate intrinsics
// 3. Allowing manual adjustment of intrinsics
// 4. Saving to JSON file

class CameraTuner {
public:
    CameraTuner(int width, int height, float tag_size_m)
        : width_(width), height_(height), tag_size_m_(tag_size_m),
          ctx_(0), pre_(ctx_, width, height, 1, nullptr) {
        // Initialize with default intrinsics
        intr_.fx = static_cast<float>(width) * 0.8f;  // Rough estimate
        intr_.fy = static_cast<float>(height) * 0.8f;
        intr_.cx = static_cast<float>(width) / 2.0f;
        intr_.cy = static_cast<float>(height) / 2.0f;
        
        // Create detector
        cv::Matx33f K(intr_.fx, 0.f, intr_.cx,
                      0.f, intr_.fy, intr_.cy,
                      0.f, 0.f, 1.f);
        detector_ = std::make_unique<AprilTagGpuDetector>(
            ctx_, width, height, tag_size_m, K);
    }
    
    void loadIntrinsics(const std::string& filename) {
        if (CameraIntrinsicsUtils::loadFromJSON(filename, intr_)) {
            std::cout << "Loaded intrinsics from " << filename << std::endl;
            updateDetector();
        } else {
            std::cout << "Failed to load intrinsics, using defaults" << std::endl;
        }
    }
    
    void saveIntrinsics(const std::string& filename) {
        if (CameraIntrinsicsUtils::saveToJSON(filename, intr_)) {
            std::cout << "Saved intrinsics to " << filename << std::endl;
        } else {
            std::cerr << "Failed to save intrinsics" << std::endl;
        }
    }
    
    void updateDetector() {
        cv::Matx33f K(intr_.fx, 0.f, intr_.cx,
                      0.f, intr_.fy, intr_.cy,
                      0.f, 0.f, 1.f);
        detector_ = std::make_unique<AprilTagGpuDetector>(
            ctx_, width_, height_, tag_size_m_, K);
    }
    
    // Estimate intrinsics from detected tag
    bool estimateIntrinsicsFromTag(const std::vector<cv::Point2f>& corners, 
                                   float& estimated_fx, float& estimated_fy,
                                   float& estimated_cx, float& estimated_cy) {
        if (corners.size() != 4) return false;
        
        // Known 3D points of tag (in tag coordinate system)
        float s = tag_size_m_ * 0.5f;
        std::vector<cv::Point3f> obj_points;
        obj_points.push_back(cv::Point3f(-s,  s, 0));  // top-left
        obj_points.push_back(cv::Point3f( s,  s, 0));  // top-right
        obj_points.push_back(cv::Point3f( s, -s, 0));  // bottom-right
        obj_points.push_back(cv::Point3f(-s, -s, 0));  // bottom-left
        
        // Use PnP to estimate pose
        cv::Mat rvec, tvec;
        cv::Mat K_guess = (cv::Mat_<double>(3, 3) <<
            static_cast<double>(intr_.fx), 0.0, static_cast<double>(intr_.cx),
            0.0, static_cast<double>(intr_.fy), static_cast<double>(intr_.cy),
            0.0, 0.0, 1.0);
        
        bool success = cv::solvePnP(obj_points, corners, K_guess, cv::Mat(),
                                   rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
        
        if (!success) return false;
        
        // Project points back and estimate focal length from reprojection
        std::vector<cv::Point2f> projected;
        cv::projectPoints(obj_points, rvec, tvec, K_guess, cv::Mat(), projected);
        
        // Estimate focal length from distance and size
        // f = (pixel_size * distance) / object_size
        // We can estimate from the tag's apparent size
        
        // Calculate average distance to tag
        double distance = cv::norm(tvec);
        
        // Calculate apparent size in pixels
        double pixel_width = cv::norm(cv::Point2f(corners[0]) - cv::Point2f(corners[1]));
        double pixel_height = cv::norm(cv::Point2f(corners[0]) - cv::Point2f(corners[2]));
        double avg_pixel_size = (pixel_width + pixel_height) / 2.0;
        
        // Estimate focal length: f = (pixel_size * distance) / object_size
        if (distance > 0 && avg_pixel_size > 0) {
            double estimated_f = (avg_pixel_size * distance) / tag_size_m_;
            estimated_fx = estimated_fy = static_cast<float>(estimated_f);
        } else {
            estimated_fx = intr_.fx;
            estimated_fy = intr_.fy;
        }
        
        // Principal point: center of detected tag
        cv::Point2f center(0, 0);
        for (const auto& pt : corners) {
            center += pt;
        }
        center *= (1.0f / 4.0f);
        
        estimated_cx = center.x;
        estimated_cy = center.y;
        
        return true;
    }
    
    // Auto-tune using multiple tag detections
    void autoTune(cv::VideoCapture& cap, int num_samples = 20) {
        std::cout << "\n=== Auto-Tuning Camera Intrinsics ===" << std::endl;
        std::cout << "Show the AprilTag to the camera from different angles and distances." << std::endl;
        std::cout << "The system will collect " << num_samples << " samples." << std::endl;
        std::cout << "Press SPACE to capture a sample, 'q' to finish early\n" << std::endl;
        
        std::vector<float> fx_samples, fy_samples, cx_samples, cy_samples;
        int sample_count = 0;
        
        while (sample_count < num_samples) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            
            if (frame.empty()) continue;
            
            // Convert to grayscale for detection
            cv::Mat gray;
            if (frame.channels() == 3) {
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = frame;
            }
            
            // Preprocess
            unsigned char* d_gray = pre_.preprocess(frame);
            
            // Detect tags
            auto detections = detector_->detect(d_gray);
            
            // Draw current frame
            cv::Mat display = frame.clone();
            
            if (!detections.empty()) {
                const auto& det = detections[0];
                
                // Draw tag
                for (int i = 0; i < 4; ++i) {
                    cv::line(display, det.corners[i],
                            det.corners[(i + 1) % 4],
                            cv::Scalar(0, 255, 0), 2);
                }
                cv::putText(display, "Tag detected! Press SPACE to sample",
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(0, 255, 0), 2);
                
                // Convert corners to vector
                std::vector<cv::Point2f> corners(4);
                for (int i = 0; i < 4; ++i) {
                    corners[i] = det.corners[i];
                }
                
                // Estimate intrinsics from this detection
                float fx, fy, cx, cy;
                if (estimateIntrinsicsFromTag(corners, fx, fy, cx, cy)) {
                    fx_samples.push_back(fx);
                    fy_samples.push_back(fy);
                    cx_samples.push_back(cx);
                    cy_samples.push_back(cy);
                    
                    cv::putText(display, 
                               cv::format("Sample %d: fx=%.1f fy=%.1f", sample_count + 1, fx, fy),
                               cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                               cv::Scalar(255, 255, 0), 2);
                }
            } else {
                cv::putText(display, "No tag detected",
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(0, 0, 255), 2);
            }
            
            cv::putText(display, 
                       cv::format("Samples: %d/%d | Press SPACE to capture, 'q' to finish",
                                 sample_count, num_samples),
                       cv::Point(10, display.rows - 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            cv::imshow("Camera Tuning", display);
            
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;
            if (key == ' ' && !detections.empty()) {
                sample_count++;
                std::cout << "Sample " << sample_count << " captured" << std::endl;
            }
        }
        
        // Calculate average intrinsics
        if (!fx_samples.empty()) {
            float avg_fx = 0, avg_fy = 0, avg_cx = 0, avg_cy = 0;
            for (size_t i = 0; i < fx_samples.size(); ++i) {
                avg_fx += fx_samples[i];
                avg_fy += fy_samples[i];
                avg_cx += cx_samples[i];
                avg_cy += cy_samples[i];
            }
            avg_fx /= fx_samples.size();
            avg_fy /= fy_samples.size();
            avg_cx /= cx_samples.size();
            avg_cy /= cy_samples.size();
            
            // Update intrinsics
            intr_.fx = avg_fx;
            intr_.fy = avg_fy;
            intr_.cx = avg_cx;
            intr_.cy = avg_cy;
            
            updateDetector();
            
            std::cout << "\n=== Tuning Results ===" << std::endl;
            std::cout << "fx: " << avg_fx << std::endl;
            std::cout << "fy: " << avg_fy << std::endl;
            std::cout << "cx: " << avg_cx << std::endl;
            std::cout << "cy: " << avg_cy << std::endl;
            std::cout << "Samples used: " << fx_samples.size() << std::endl;
        } else {
            std::cout << "No valid samples collected!" << std::endl;
        }
    }
    
    void interactiveTune(cv::VideoCapture& cap) {
        std::cout << "\n=== Interactive Camera Tuning ===" << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  w/s: Adjust fx (+/-10)" << std::endl;
        std::cout << "  e/d: Adjust fy (+/-10)" << std::endl;
        std::cout << "  r/f: Adjust cx (+/-5)" << std::endl;
        std::cout << "  t/g: Adjust cy (+/-5)" << std::endl;
        std::cout << "  a: Auto-estimate from current tag" << std::endl;
        std::cout << "  SPACE: Save current intrinsics" << std::endl;
        std::cout << "  'q': Quit\n" << std::endl;
        
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            
            if (frame.empty()) continue;
            
            // Preprocess and detect
            unsigned char* d_gray = pre_.preprocess(frame);
            auto detections = detector_->detect(d_gray);
            
            // Draw frame
            cv::Mat display = frame.clone();
            
            // Draw detected tags
            for (const auto& det : detections) {
                for (int i = 0; i < 4; ++i) {
                    cv::line(display, det.corners[i],
                            det.corners[(i + 1) % 4],
                            cv::Scalar(0, 255, 0), 2);
                }
                cv::putText(display, std::to_string(det.id),
                           det.corners[0], cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(255, 0, 0), 2);
                
                // Show reprojection error
                cv::putText(display, 
                           cv::format("Reproj: %.2fpx", det.reprojection_error),
                           cv::Point(det.corners[0].x, det.corners[0].y + 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(0, 255, 255), 2);
            }
            
            // Display current intrinsics
            std::string info = cv::format("fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                                         intr_.fx, intr_.fy, intr_.cx, intr_.cy);
            cv::putText(display, info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            if (!detections.empty()) {
                cv::putText(display, "Tag detected! Adjust intrinsics to minimize reprojection error",
                           cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                           cv::Scalar(0, 255, 0), 2);
            }
            
            cv::imshow("Camera Tuning", display);
            
            int key = cv::waitKey(1) & 0xFF;
            bool updated = false;
            
            switch (key) {
                case 'w': intr_.fx += 10.0f; updated = true; break;
                case 's': intr_.fx -= 10.0f; updated = true; break;
                case 'e': intr_.fy += 10.0f; updated = true; break;
                case 'd': intr_.fy -= 10.0f; updated = true; break;
                case 'r': intr_.cx += 5.0f; updated = true; break;
                case 'f': intr_.cx -= 5.0f; updated = true; break;
                case 't': intr_.cy += 5.0f; updated = true; break;
                case 'g': intr_.cy -= 5.0f; updated = true; break;
                case 'a':
                    if (!detections.empty()) {
                        std::vector<cv::Point2f> corners(4);
                        for (int i = 0; i < 4; ++i) {
                            corners[i] = detections[0].corners[i];
                        }
                        float fx, fy, cx, cy;
                        if (estimateIntrinsicsFromTag(corners, fx, fy, cx, cy)) {
                            intr_.fx = fx;
                            intr_.fy = fy;
                            intr_.cx = cx;
                            intr_.cy = cy;
                            updated = true;
                            std::cout << "Auto-estimated: fx=" << fx << " fy=" << fy 
                                     << " cx=" << cx << " cy=" << cy << std::endl;
                        }
                    }
                    break;
                case ' ':  // Space to save
                    saveIntrinsics("camera_intrinsics.json");
                    std::cout << "Intrinsics saved!" << std::endl;
                    break;
                case 'q':
                    return;
            }
            
            if (updated) {
                updateDetector();
            }
        }
    }
    
    const CameraIntrinsics& getIntrinsics() const { return intr_; }
    void setIntrinsics(const CameraIntrinsics& intr) {
        intr_ = intr;
        updateDetector();
    }

private:
    int width_, height_;
    float tag_size_m_;
    GpuContext ctx_;
    ImagePreprocessor pre_;
    std::unique_ptr<AprilTagGpuDetector> detector_;
    CameraIntrinsics intr_;
};

int main(int argc, char** argv) {
    try {
        int width = 1280;
        int height = 720;
        float tag_size_m = 0.165f;  // Default tag size in meters
        std::string intrinsics_file = "camera_intrinsics.json";
        bool auto_mode = false;
        int num_samples = 20;
        
        // Parse arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--tag-size" && i + 1 < argc) {
                tag_size_m = std::stof(argv[++i]);
            } else if (arg == "--file" && i + 1 < argc) {
                intrinsics_file = argv[++i];
            } else if (arg == "--auto") {
                auto_mode = true;
            } else if (arg == "--samples" && i + 1 < argc) {
                num_samples = std::stoi(argv[++i]);
            }
        }
        
        std::cout << "=== Camera Tuning with AprilTag ===" << std::endl;
        std::cout << "Tag size: " << tag_size_m << " meters" << std::endl;
        std::cout << "Intrinsics file: " << intrinsics_file << std::endl;
        std::cout << "Mode: " << (auto_mode ? "Auto-tune" : "Interactive") << std::endl;
        
        // Open camera
        cv::VideoCapture cap;
        cap.open("/dev/video0", cv::CAP_V4L2);
        if (!cap.isOpened()) {
            cap.open(0, cv::CAP_V4L2);
        }
        if (!cap.isOpened()) {
            cap.open(0);
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera" << std::endl;
            return 1;
        }
        
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        // Create tuner
        CameraTuner tuner(width, height, tag_size_m);
        
        // Try to load existing intrinsics
        tuner.loadIntrinsics(intrinsics_file);
        
        // Run tuning
        if (auto_mode) {
            tuner.autoTune(cap, num_samples);
            tuner.saveIntrinsics(intrinsics_file);
        } else {
            tuner.interactiveTune(cap);
        }
        
        cap.release();
        cv::destroyAllWindows();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

