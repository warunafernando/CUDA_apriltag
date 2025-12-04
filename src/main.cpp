#include "gpu_context.h"
#include "image_preprocessor.h"
#include "apriltag_gpu.h"
#include "camera_intrinsics.h"
#include "config.h"
#include "camera_controls.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <map>
#include <iomanip>

// Simple demo program:
//  - Opens a 720p camera
//  - Streams frames into pinned host memory
//  - Runs the GPU preprocessor + AprilTagGpuDetector
//  - Visualizes the (currently empty) detections and prints FPS
//
// This matches the high-level architecture and can be used as a
// starting point to incrementally port a full AprilTag GPU pipeline.

int main(int argc, char** argv) {
    try {
        // Check for --read-camera flag
        bool read_camera_only = false;
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "--read-camera" || arg == "-r") {
                read_camera_only = true;
            }
        }
        
        // Load configuration
        AppConfig config;
        std::string config_file = "config.json";
        if (!AppConfig::loadFromJSON(config_file, config)) {
            std::cout << "Config file not found, using defaults. Creating " << config_file << "..." << std::endl;
            config.initDefaultIntrinsics();
            AppConfig::saveToJSON(config_file, config);
        } else {
            std::cout << "Loaded configuration from " << config_file << std::endl;
        }
        
        int width = config.camera.width;
        int height = config.camera.height;
        int decimation = config.camera.decimation;

        cv::VideoCapture cap;
        bool use_test_image = false;
        
        if (read_camera_only) {
            // Just read camera settings and exit
            if (config.camera.device.find("/dev/video") == 0) {
                cap.open(config.camera.device, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
            } else {
                int idx = std::stoi(config.camera.device);
                cap.open(idx, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
            }
            if (!cap.isOpened()) {
                cap.open(0, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
            }
            if (!cap.isOpened()) {
                cap.open(0);
            }
            
            if (cap.isOpened()) {
                printCameraControls(cap);
                std::cout << "\nTo save these settings to config.json, use: --save-camera" << std::endl;
            } else {
                std::cerr << "Failed to open camera" << std::endl;
            }
            return 0;
        }
        
        // Check for --save-camera flag
        bool save_camera_settings = false;
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "--save-camera" || arg == "-s") {
                save_camera_settings = true;
            }
        }
        
        if (argc > 1 && !save_camera_settings) {
            std::string arg = argv[1];
            if (arg == "--test" || arg == "-t") {
                use_test_image = true;
            } else {
                // Try as device path or index
                if (arg.find("/dev/video") == 0) {
                    cap.open(arg, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
                } else {
                    int idx = std::stoi(arg);
                    cap.open(idx, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
                }
            }
        } else {
            // Use configured device
            if (config.camera.device.find("/dev/video") == 0) {
                cap.open(config.camera.device, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
            } else {
                int idx = std::stoi(config.camera.device);
                cap.open(idx, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
            }
            if (!cap.isOpened()) {
                cap.open(0, config.camera.use_v4l2 ? cv::CAP_V4L2 : 0);
            }
            if (!cap.isOpened()) {
                // Fallback to default backend
                cap.open(0);
            }
        }

        if (!cap.isOpened() && !use_test_image) {
            std::cerr << "Failed to open camera. Use --test to test with synthetic image.\n";
            return 1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, config.camera.fps);
        
        // Handle save camera settings mode
        if (save_camera_settings) {
            if (saveCameraControlsToConfig(cap, config_file)) {
                std::cout << "Camera settings saved to " << config_file << std::endl;
                std::cout << "You can now edit the config file to adjust values." << std::endl;
            }
            return 0;
        }
        
        // Read current camera settings
        std::cout << "\nReading current camera settings..." << std::endl;
        AppConfig::Camera::Controls current_controls = readCameraControls(cap);
        printCameraControls(cap);
        
        // Save current camera settings to config.json (so config always reflects actual camera state)
        config.camera.controls = current_controls;
        if (AppConfig::saveToJSON(config_file, config)) {
            std::cout << "Saved current camera settings to " << config_file << std::endl;
        }
        
        // Apply camera control settings from config (if any were specified)
        bool has_manual_settings = false;
        if (config.camera.controls.brightness >= 0 || 
            config.camera.controls.contrast >= 0 ||
            config.camera.controls.exposure >= 0 ||
            config.camera.controls.auto_exposure >= 0) {
            has_manual_settings = true;
        }
        
        if (has_manual_settings) {
            std::cout << "\nApplying camera controls from config..." << std::endl;
            applyCameraControls(cap, config.camera.controls);
            
            // Read and save settings after applying
            std::cout << "\nReading camera settings after applying config..." << std::endl;
            current_controls = readCameraControls(cap);
            printCameraControls(cap);
            
            // Update config with actual values after applying
            config.camera.controls = current_controls;
            if (AppConfig::saveToJSON(config_file, config)) {
                std::cout << "Updated " << config_file << " with actual camera settings" << std::endl;
            }
        } else {
            std::cout << "\nNo manual camera settings in config, using camera defaults." << std::endl;
            std::cout << "Edit " << config_file << " to set specific values." << std::endl;
        }

        GpuContext ctx(0);

        CameraIntrinsics intr;
        
        // Try to load from JSON file, otherwise use defaults
        if (!CameraIntrinsicsUtils::loadFromJSON(config.files.intrinsics, intr)) {
            // Use default intrinsics from config
            std::cout << "Using default intrinsics (load from " << config.files.intrinsics << " for calibrated values)" << std::endl;
            intr = config.default_intrinsics;
        } else {
            std::cout << "Loaded camera intrinsics from " << config.files.intrinsics << std::endl;
        }

        ImagePreprocessor pre(ctx, width, height, decimation, &intr);

        cv::Matx33f K(intr.fx, 0.f, intr.cx,
                      0.f, intr.fy, intr.cy,
                      0.f, 0.f, 1.f);

        AprilTagGpuDetector detector(ctx,
                                     pre.workingWidth(),
                                     pre.workingHeight(),
                                     config.tag.size_m,
                                     K);
        
        // Apply detection settings from config
        detector.setUseGpuQuadExtraction(config.detection.use_gpu_quad_extraction);
        detector.setMinQuality(config.detection.min_quality);
        detector.setMaxReprojectionError(config.detection.max_reprojection_error);
        detector.setMinDecisionMargin(config.detection.min_decision_margin);
        detector.setEnableSubpixelRefinement(config.detection.enable_subpixel_refinement);
        detector.setEnableTemporalFiltering(config.detection.enable_temporal_filtering);
        
        if (config.detection.use_gpu_quad_extraction) {
            std::cout << "Using GPU quad extraction..." << std::endl;
        } else {
            std::cout << "Using OpenCV CPU quad extraction..." << std::endl;
        }

        cv::Mat frame;

        bool use_gui = true;
        try {
            cv::namedWindow("cuda_apriltag", cv::WINDOW_NORMAL);
        } catch (const cv::Exception&) {
            use_gui = false;
        }

        int frame_count = 0;
        auto t_start = cv::getTickCount();
        auto t_last_log = t_start;
        
        // Statistics tracking for test
        std::map<int, int> tag_detection_counts;  // tag_id -> detection count
        int total_frames_processed = 0;
        int frames_with_detections = 0;
        const double test_duration_seconds = config.test.duration_seconds;
        bool test_complete = false;
        
        // Frame capture for full frame vs ROI visualization
        int full_frame_captures = 0;
        int roi_captures = 0;
        const int max_captures = config.test.max_captures;
        std::string capture_dir = config.test.capture_dir;
        system(("mkdir -p " + capture_dir).c_str());

        cv::Mat test_frame;
        if (use_test_image) {
            // Create a test frame with a simple pattern
            test_frame = cv::Mat::zeros(height, width, CV_8UC3);
            cv::rectangle(test_frame, cv::Point(100, 100), cv::Point(500, 500), cv::Scalar(255, 255, 255), -1);
            cv::rectangle(test_frame, cv::Point(150, 150), cv::Point(450, 450), cv::Scalar(0, 0, 0), -1);
        }
        
        while (true) {
            if (use_test_image) {
                frame = test_frame.clone();
                // Simulate frame delay
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
            } else {
                if (!cap.read(frame)) {
                    std::cerr << "Camera read failed\n";
                    break;
                }
            }

            PreprocessTimings pre_timings;
            DetectionTimings det_timings;

            auto t_cap_end = cv::getTickCount();

            unsigned char* d_gray = pre.preprocess(frame, &pre_timings);
            
            // Determine if this is a full frame or ROI detection
            bool is_full_frame = detector.isFullFrameDetection();
            auto detections = detector.detect(d_gray, true, &det_timings);
            
            // Get ROI regions for visualization
            auto rois = detector.getROIs();

            // Track statistics
            total_frames_processed++;
            if (!detections.empty()) {
                frames_with_detections++;
                for (const auto& det : detections) {
                    tag_detection_counts[det.id]++;
                }
            }
            
            // Check if 1 minute test is complete
            double t_now = cv::getTickCount();
            frame_count++;
            double elapsed = (t_now - t_start) / cv::getTickFrequency();
            double fps = frame_count / std::max(elapsed, 1e-3);
            
            if (elapsed >= test_duration_seconds && !test_complete) {
                test_complete = true;
                // Print final statistics (flush immediately)
                std::cout << std::endl;
                std::cout << "\n========== 1-MINUTE DETECTION TEST RESULTS ==========\n";
                std::cout << "Total frames processed: " << total_frames_processed << "\n";
                std::cout << "Frames with detections: " << frames_with_detections << "\n";
                double detection_probability = (frames_with_detections * 100.0) / std::max(total_frames_processed, 1);
                std::cout << "Detection probability: " << std::fixed << std::setprecision(2) << detection_probability << "%\n";
                std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << (total_frames_processed / elapsed) << "\n";
                std::cout << "Test duration: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n";
                std::cout << "\nTag Detection Counts:\n";
                if (tag_detection_counts.empty()) {
                    std::cout << "  No tags detected during test period.\n";
                } else {
                    for (const auto& pair : tag_detection_counts) {
                        double tag_prob = (pair.second * 100.0) / total_frames_processed;
                        std::cout << "  Tag #" << pair.first << ": detected " << pair.second 
                                  << " times (" << std::fixed << std::setprecision(2) << tag_prob << "% of frames)\n";
                    }
                }
                std::cout << "====================================================\n" << std::endl;
                std::cout.flush();
                
                // Continue running but mark test as complete
            }

            // Draw detection mode indicator
            std::string mode_text = is_full_frame ? "FULL FRAME" : "ROI";
            cv::Scalar mode_color = is_full_frame ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);  // Green for full, Orange for ROI
            
            if (use_gui) {
                cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)),
                            cv::Point(30, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            cv::Scalar(0, 255, 0), 2);
                
                // Draw mode indicator
                cv::putText(frame, "MODE: " + mode_text,
                            cv::Point(30, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            mode_color, 2);
                
                // Draw ROI boxes (if in ROI mode)
                // ROIs are in working resolution, need to scale to full frame
                float scale_x = static_cast<float>(width) / static_cast<float>(pre.workingWidth());
                float scale_y = static_cast<float>(height) / static_cast<float>(pre.workingHeight());
                
                if (!is_full_frame && !rois.empty()) {
                    for (const auto& roi : rois) {
                        if (roi.age < 20) {  // Only draw active ROIs
                            // Scale ROI coordinates to full frame
                            int scaled_x = static_cast<int>(roi.x * scale_x);
                            int scaled_y = static_cast<int>(roi.y * scale_y);
                            int scaled_w = static_cast<int>(roi.w * scale_x);
                            int scaled_h = static_cast<int>(roi.h * scale_y);
                            
                            cv::rectangle(frame, 
                                         cv::Point(scaled_x, scaled_y),
                                         cv::Point(scaled_x + scaled_w, scaled_y + scaled_h),
                                         cv::Scalar(255, 255, 0), 2);  // Yellow for ROI boxes
                            cv::putText(frame, "ROI",
                                       cv::Point(scaled_x, scaled_y - 5),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                       cv::Scalar(255, 255, 0), 1);
                        }
                    }
                }

                // Draw detected tags with 3D cube visualization
                // Corners are in working resolution (640x360), need to scale to full frame (1280x720)
                // (scale_x and scale_y already computed above for ROI scaling)
                
                // Camera intrinsics for 3D projection (scaled to full frame)
                cv::Matx33f K_full(intr.fx * scale_x, 0.f, intr.cx * scale_x,
                                   0.f, intr.fy * scale_y, intr.cy * scale_y,
                                   0.f, 0.f, 1.f);
                
                for (const auto& det : detections) {
                    cv::Point2f scaled_corners[4];
                    for (int i = 0; i < 4; ++i) {
                        scaled_corners[i] = cv::Point2f(det.corners[i].x * scale_x, det.corners[i].y * scale_y);
                    }
                    
                    // Draw quad outline
                    for (int i = 0; i < 4; ++i) {
                        cv::line(frame, scaled_corners[i],
                                scaled_corners[(i + 1) % 4],
                                cv::Scalar(0, 0, 255), 2);
                    }
                    
                    // Draw 3D cube on the tag
                    float cube_size = config.tag.size_m * 0.5f;  // Half tag size for cube
                    std::vector<cv::Point3f> cube_points_3d;
                    
                    // Define cube corners in tag coordinate system (tag is in XY plane, Z=0)
                    // Bottom face (on tag plane)
                    cube_points_3d.push_back(cv::Point3f(-cube_size, -cube_size, 0));  // 0: bottom-left-back
                    cube_points_3d.push_back(cv::Point3f( cube_size, -cube_size, 0));  // 1: bottom-right-back
                    cube_points_3d.push_back(cv::Point3f( cube_size,  cube_size, 0));  // 2: bottom-right-front
                    cube_points_3d.push_back(cv::Point3f(-cube_size,  cube_size, 0));  // 3: bottom-left-front
                    // Top face (above tag)
                    cube_points_3d.push_back(cv::Point3f(-cube_size, -cube_size, cube_size * 2));  // 4: top-left-back
                    cube_points_3d.push_back(cv::Point3f( cube_size, -cube_size, cube_size * 2));  // 5: top-right-back
                    cube_points_3d.push_back(cv::Point3f( cube_size,  cube_size, cube_size * 2));  // 6: top-right-front
                    cube_points_3d.push_back(cv::Point3f(-cube_size,  cube_size, cube_size * 2));  // 7: top-left-front
                    
                    // Transform 3D points from tag frame to camera frame
                    std::vector<cv::Point3f> cube_points_cam;
                    for (const auto& pt : cube_points_3d) {
                        cv::Vec4f pt_homogeneous(pt.x, pt.y, pt.z, 1.0f);
                        cv::Vec4f pt_cam = det.T_cam_tag * pt_homogeneous;
                        cube_points_cam.push_back(cv::Point3f(pt_cam[0], pt_cam[1], pt_cam[2]));
                    }
                    
                    // Project 3D points to 2D image coordinates
                    std::vector<cv::Point2f> cube_points_2d;
                    for (const auto& pt : cube_points_cam) {
                        if (pt.z > 0) {  // Only project points in front of camera
                            float x = (pt.x / pt.z) * K_full(0, 0) + K_full(0, 2);
                            float y = (pt.y / pt.z) * K_full(1, 1) + K_full(1, 2);
                            cube_points_2d.push_back(cv::Point2f(x, y));
                        } else {
                            cube_points_2d.push_back(cv::Point2f(-1, -1));  // Invalid point
                        }
                    }
                    
                    // Draw cube edges
                    if (cube_points_2d.size() == 8) {
                        // Bottom face edges
                        for (int i = 0; i < 4; ++i) {
                            int next = (i + 1) % 4;
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[next].x >= 0) {
                                cv::line(frame, cube_points_2d[i], cube_points_2d[next], cv::Scalar(0, 255, 0), 2);
                            }
                        }
                        // Top face edges
                        for (int i = 4; i < 8; ++i) {
                            int next = 4 + ((i - 4 + 1) % 4);
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[next].x >= 0) {
                                cv::line(frame, cube_points_2d[i], cube_points_2d[next], cv::Scalar(0, 255, 0), 2);
                            }
                        }
                        // Vertical edges connecting bottom to top
                        for (int i = 0; i < 4; ++i) {
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[i + 4].x >= 0) {
                                cv::line(frame, cube_points_2d[i], cube_points_2d[i + 4], cv::Scalar(0, 255, 0), 2);
                            }
                        }
                    }
                    
                    // Draw corner markers to verify order
                    cv::circle(frame, scaled_corners[0], 5, cv::Scalar(0, 255, 0), -1);  // Green: corner 0 (top-left)
                    cv::circle(frame, scaled_corners[1], 5, cv::Scalar(255, 0, 0), -1);  // Blue: corner 1 (top-right)
                    cv::circle(frame, scaled_corners[2], 5, cv::Scalar(0, 255, 255), -1); // Yellow: corner 2 (bottom-right)
                    cv::circle(frame, scaled_corners[3], 5, cv::Scalar(255, 0, 255), -1); // Magenta: corner 3 (bottom-left)
                    cv::putText(frame, std::to_string(det.id),
                                scaled_corners[0],
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 0, 0), 2);
                }

                cv::imshow("cuda_apriltag", frame);
            }
            
            // Capture frames for documentation
            if (is_full_frame && full_frame_captures < max_captures && !detections.empty()) {
                cv::Mat capture_frame = frame.clone();
                cv::putText(capture_frame, "FULL FRAME DETECTION",
                           cv::Point(30, 40),
                           cv::FONT_HERSHEY_SIMPLEX, 1.2,
                           cv::Scalar(0, 255, 0), 3);
                cv::putText(capture_frame, "FPS: " + std::to_string(static_cast<int>(fps)),
                           cv::Point(30, 80),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(0, 255, 0), 2);
                cv::putText(capture_frame, "Tags: " + std::to_string(detections.size()),
                           cv::Point(30, 120),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(0, 255, 0), 2);
                
                // Scale corners for capture
                float scale_x_cap = static_cast<float>(width) / static_cast<float>(pre.workingWidth());
                float scale_y_cap = static_cast<float>(height) / static_cast<float>(pre.workingHeight());
                
                // Camera intrinsics for 3D projection (scaled to full frame)
                cv::Matx33f K_full_cap(intr.fx * scale_x_cap, 0.f, intr.cx * scale_x_cap,
                                       0.f, intr.fy * scale_y_cap, intr.cy * scale_y_cap,
                                       0.f, 0.f, 1.f);
                
                // Draw detected tags with 3D cube (scaled)
                for (const auto& det : detections) {
                    cv::Point2f scaled_corners_cap[4];
                    for (int i = 0; i < 4; ++i) {
                        scaled_corners_cap[i] = cv::Point2f(det.corners[i].x * scale_x_cap, det.corners[i].y * scale_y_cap);
                    }
                    
                    // Draw quad outline
                    for (int i = 0; i < 4; ++i) {
                        cv::line(capture_frame, scaled_corners_cap[i],
                                scaled_corners_cap[(i + 1) % 4],
                                cv::Scalar(0, 0, 255), 3);
                    }
                    
                    // Draw 3D cube on the tag
                    float cube_size = config.tag.size_m * 0.5f;
                    std::vector<cv::Point3f> cube_points_3d;
                    cube_points_3d.push_back(cv::Point3f(-cube_size, -cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f( cube_size, -cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f( cube_size,  cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f(-cube_size,  cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f(-cube_size, -cube_size, cube_size * 2));
                    cube_points_3d.push_back(cv::Point3f( cube_size, -cube_size, cube_size * 2));
                    cube_points_3d.push_back(cv::Point3f( cube_size,  cube_size, cube_size * 2));
                    cube_points_3d.push_back(cv::Point3f(-cube_size,  cube_size, cube_size * 2));
                    
                    std::vector<cv::Point3f> cube_points_cam;
                    for (const auto& pt : cube_points_3d) {
                        cv::Vec4f pt_homogeneous(pt.x, pt.y, pt.z, 1.0f);
                        cv::Vec4f pt_cam = det.T_cam_tag * pt_homogeneous;
                        cube_points_cam.push_back(cv::Point3f(pt_cam[0], pt_cam[1], pt_cam[2]));
                    }
                    
                    std::vector<cv::Point2f> cube_points_2d;
                    for (const auto& pt : cube_points_cam) {
                        if (pt.z > 0) {
                            float x = (pt.x / pt.z) * K_full_cap(0, 0) + K_full_cap(0, 2);
                            float y = (pt.y / pt.z) * K_full_cap(1, 1) + K_full_cap(1, 2);
                            cube_points_2d.push_back(cv::Point2f(x, y));
                        } else {
                            cube_points_2d.push_back(cv::Point2f(-1, -1));
                        }
                    }
                    
                    if (cube_points_2d.size() == 8) {
                        for (int i = 0; i < 4; ++i) {
                            int next = (i + 1) % 4;
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[next].x >= 0) {
                                cv::line(capture_frame, cube_points_2d[i], cube_points_2d[next], cv::Scalar(0, 255, 0), 3);
                            }
                        }
                        for (int i = 4; i < 8; ++i) {
                            int next = 4 + ((i - 4 + 1) % 4);
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[next].x >= 0) {
                                cv::line(capture_frame, cube_points_2d[i], cube_points_2d[next], cv::Scalar(0, 255, 0), 3);
                            }
                        }
                        for (int i = 0; i < 4; ++i) {
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[i + 4].x >= 0) {
                                cv::line(capture_frame, cube_points_2d[i], cube_points_2d[i + 4], cv::Scalar(0, 255, 0), 3);
                            }
                        }
                    }
                    
                    cv::putText(capture_frame, "ID:" + std::to_string(det.id),
                               scaled_corners_cap[0],
                               cv::FONT_HERSHEY_SIMPLEX, 0.7,
                               cv::Scalar(255, 0, 0), 2);
                }
                
                std::string filename = capture_dir + "/full_frame_" + std::to_string(full_frame_captures) + ".jpg";
                cv::imwrite(filename, capture_frame);
                std::cout << "Captured full frame detection: " << filename << std::endl;
                full_frame_captures++;
            }
            
            if (!is_full_frame && roi_captures < max_captures && !rois.empty()) {
                cv::Mat capture_frame = frame.clone();
                cv::putText(capture_frame, "ROI DETECTION",
                           cv::Point(30, 40),
                           cv::FONT_HERSHEY_SIMPLEX, 1.2,
                           cv::Scalar(0, 165, 255), 3);
                cv::putText(capture_frame, "FPS: " + std::to_string(static_cast<int>(fps)),
                           cv::Point(30, 80),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(0, 165, 255), 2);
                cv::putText(capture_frame, "ROIs: " + std::to_string(rois.size()),
                           cv::Point(30, 120),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(0, 165, 255), 2);
                
                // Scale ROI and corner coordinates for capture
                float scale_x_roi = static_cast<float>(width) / static_cast<float>(pre.workingWidth());
                float scale_y_roi = static_cast<float>(height) / static_cast<float>(pre.workingHeight());
                
                // Draw ROI boxes (scaled)
                for (const auto& roi : rois) {
                    if (roi.age < 20) {
                        int scaled_x = static_cast<int>(roi.x * scale_x_roi);
                        int scaled_y = static_cast<int>(roi.y * scale_y_roi);
                        int scaled_w = static_cast<int>(roi.w * scale_x_roi);
                        int scaled_h = static_cast<int>(roi.h * scale_y_roi);
                        
                        cv::rectangle(capture_frame,
                                     cv::Point(scaled_x, scaled_y),
                                     cv::Point(scaled_x + scaled_w, scaled_y + scaled_h),
                                     cv::Scalar(255, 255, 0), 3);
                        cv::putText(capture_frame, "ROI",
                                   cv::Point(scaled_x, scaled_y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                   cv::Scalar(255, 255, 0), 2);
                    }
                }
                
                // Camera intrinsics for 3D projection (scaled to full frame)
                cv::Matx33f K_full_roi(intr.fx * scale_x_roi, 0.f, intr.cx * scale_x_roi,
                                       0.f, intr.fy * scale_y_roi, intr.cy * scale_y_roi,
                                       0.f, 0.f, 1.f);
                
                // Draw detected tags with 3D cube (scaled)
                for (const auto& det : detections) {
                    cv::Point2f scaled_corners_roi[4];
                    for (int i = 0; i < 4; ++i) {
                        scaled_corners_roi[i] = cv::Point2f(det.corners[i].x * scale_x_roi, det.corners[i].y * scale_y_roi);
                    }
                    
                    // Draw quad outline
                    for (int i = 0; i < 4; ++i) {
                        cv::line(capture_frame, scaled_corners_roi[i],
                                scaled_corners_roi[(i + 1) % 4],
                                cv::Scalar(0, 0, 255), 3);
                    }
                    
                    // Draw 3D cube on the tag
                    float cube_size = config.tag.size_m * 0.5f;
                    std::vector<cv::Point3f> cube_points_3d;
                    cube_points_3d.push_back(cv::Point3f(-cube_size, -cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f( cube_size, -cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f( cube_size,  cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f(-cube_size,  cube_size, 0));
                    cube_points_3d.push_back(cv::Point3f(-cube_size, -cube_size, cube_size * 2));
                    cube_points_3d.push_back(cv::Point3f( cube_size, -cube_size, cube_size * 2));
                    cube_points_3d.push_back(cv::Point3f( cube_size,  cube_size, cube_size * 2));
                    cube_points_3d.push_back(cv::Point3f(-cube_size,  cube_size, cube_size * 2));
                    
                    std::vector<cv::Point3f> cube_points_cam;
                    for (const auto& pt : cube_points_3d) {
                        cv::Vec4f pt_homogeneous(pt.x, pt.y, pt.z, 1.0f);
                        cv::Vec4f pt_cam = det.T_cam_tag * pt_homogeneous;
                        cube_points_cam.push_back(cv::Point3f(pt_cam[0], pt_cam[1], pt_cam[2]));
                    }
                    
                    std::vector<cv::Point2f> cube_points_2d;
                    for (const auto& pt : cube_points_cam) {
                        if (pt.z > 0) {
                            float x = (pt.x / pt.z) * K_full_roi(0, 0) + K_full_roi(0, 2);
                            float y = (pt.y / pt.z) * K_full_roi(1, 1) + K_full_roi(1, 2);
                            cube_points_2d.push_back(cv::Point2f(x, y));
                        } else {
                            cube_points_2d.push_back(cv::Point2f(-1, -1));
                        }
                    }
                    
                    if (cube_points_2d.size() == 8) {
                        for (int i = 0; i < 4; ++i) {
                            int next = (i + 1) % 4;
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[next].x >= 0) {
                                cv::line(capture_frame, cube_points_2d[i], cube_points_2d[next], cv::Scalar(0, 255, 0), 3);
                            }
                        }
                        for (int i = 4; i < 8; ++i) {
                            int next = 4 + ((i - 4 + 1) % 4);
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[next].x >= 0) {
                                cv::line(capture_frame, cube_points_2d[i], cube_points_2d[next], cv::Scalar(0, 255, 0), 3);
                            }
                        }
                        for (int i = 0; i < 4; ++i) {
                            if (cube_points_2d[i].x >= 0 && cube_points_2d[i + 4].x >= 0) {
                                cv::line(capture_frame, cube_points_2d[i], cube_points_2d[i + 4], cv::Scalar(0, 255, 0), 3);
                            }
                        }
                    }
                    
                    cv::putText(capture_frame, "ID:" + std::to_string(det.id),
                               scaled_corners_roi[0],
                               cv::FONT_HERSHEY_SIMPLEX, 0.7,
                               cv::Scalar(255, 0, 0), 2);
                }
                
                std::string filename = capture_dir + "/roi_" + std::to_string(roi_captures) + ".jpg";
                cv::imwrite(filename, capture_frame);
                std::cout << "Captured ROI detection: " << filename << std::endl;
                roi_captures++;
            }
            
            // Stop capturing after we have enough
            if (full_frame_captures >= max_captures && roi_captures >= max_captures) {
                std::cout << "\n=== Frame capture complete! ===" << std::endl;
                std::cout << "Captured " << full_frame_captures << " full frame images" << std::endl;
                std::cout << "Captured " << roi_captures << " ROI images" << std::endl;
                std::cout << "Images saved in: " << capture_dir << "/" << std::endl;
                // Continue running but stop capturing
            }

            // Periodic console log for monitoring.
            double log_elapsed = (t_now - t_last_log) / cv::getTickFrequency();
            if (log_elapsed >= 1.0) {
                double cap_ms = (t_cap_end - t_start) * 1000.0 / cv::getTickFrequency();
                std::cout << "FPS=" << fps
                          << " detections=" << detections.size();
                if (!detections.empty()) {
                    std::cout << " tag_ids=[";
                    for (size_t i = 0; i < detections.size(); ++i) {
                        std::cout << detections[i].id;
                        if (i < detections.size() - 1) std::cout << ",";
                    }
                    std::cout << "]";
                }
                std::cout << " | CAP(ms)=" << cap_ms
                          << " PRE(ms): memcpy=" << pre_timings.memcpy_host_ms
                          << " h2d=" << pre_timings.h2d_ms
                          << " bgr2gray=" << pre_timings.bgr2gray_ms
                          << " decim=" << pre_timings.decimate_ms
                          << " DET(ms): grad=" << det_timings.grad_ms
                          << " edge=" << det_timings.edge_ms
                          << " quad=" << det_timings.quad_ms
                          << " decode=" << det_timings.decode_ms
                          << " pnp=" << det_timings.pnp_ms
                          << " total=" << det_timings.total_ms;
                if (test_complete) {
                    std::cout << " [TEST COMPLETE]";
                }
                std::cout << std::endl;
                t_last_log = t_now;
            }
            
            // Exit after test completion in headless mode
            if (test_complete && !use_gui) {
                break;
            }
            if (use_gui) {
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    break;
                }
            } else {
                // In headless mode, continue until test is complete
                if (test_complete) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}


