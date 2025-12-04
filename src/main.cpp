#include "gpu_context.h"
#include "image_preprocessor.h"
#include "apriltag_gpu.h"

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
        int width = 1280;
        int height = 720;
        int decimation = 2;  // working 640x360 as in the requirements
        // Note: Can increase decimation to 3 (426x240) for even faster processing if needed

        cv::VideoCapture cap;
        bool use_test_image = false;
        
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "--test" || arg == "-t") {
                use_test_image = true;
            } else {
                // Try as device path or index
                if (arg.find("/dev/video") == 0) {
                    cap.open(arg, cv::CAP_V4L2);
                } else {
                    int idx = std::stoi(arg);
                    cap.open(idx, cv::CAP_V4L2);
                }
            }
        } else {
            // Try USB camera with V4L2 first (for Arducam and other USB cameras)
            cap.open("/dev/video0", cv::CAP_V4L2);
            if (!cap.isOpened()) {
                cap.open(0, cv::CAP_V4L2);
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
        cap.set(cv::CAP_PROP_FPS, 120);

        GpuContext ctx(0);

        CameraIntrinsics intr;
        // TODO: fill with real calibration values
        intr.fx = 1000.f;
        intr.fy = 1000.f;
        intr.cx = width / 2.f;
        intr.cy = height / 2.f;

        ImagePreprocessor pre(ctx, width, height, decimation, &intr);

        cv::Matx33f K(intr.fx, 0.f, intr.cx,
                      0.f, intr.fy, intr.cy,
                      0.f, 0.f, 1.f);

        float tag_size_m = 0.165f;  // example tag size in meters
        AprilTagGpuDetector detector(ctx,
                                     pre.workingWidth(),
                                     pre.workingHeight(),
                                     tag_size_m,
                                     K);
        
        // Temporarily use OpenCV CPU quad extraction for verification
        detector.setUseGpuQuadExtraction(false);
        std::cout << "Using OpenCV CPU quad extraction for tag verification..." << std::endl;

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
        
        // Statistics tracking for 1-minute test
        std::map<int, int> tag_detection_counts;  // tag_id -> detection count
        int total_frames_processed = 0;
        int frames_with_detections = 0;
        const double test_duration_seconds = 60.0;  // 1 minute test
        bool test_complete = false;
        
        // Frame capture for full frame vs ROI visualization
        int full_frame_captures = 0;
        int roi_captures = 0;
        const int max_captures = 5;  // Capture 5 of each type
        std::string capture_dir = "captures";
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

                // Draw detected tags
                // Corners are in working resolution (640x360), need to scale to full frame (1280x720)
                // (scale_x and scale_y already computed above for ROI scaling)
                
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
                
                // Draw detected tags (scaled)
                for (const auto& det : detections) {
                    cv::Point2f scaled_corners_cap[4];
                    for (int i = 0; i < 4; ++i) {
                        scaled_corners_cap[i] = cv::Point2f(det.corners[i].x * scale_x_cap, det.corners[i].y * scale_y_cap);
                    }
                    
                    for (int i = 0; i < 4; ++i) {
                        cv::line(capture_frame, scaled_corners_cap[i],
                                scaled_corners_cap[(i + 1) % 4],
                                cv::Scalar(0, 0, 255), 3);
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
                
                // Draw detected tags (scaled)
                for (const auto& det : detections) {
                    cv::Point2f scaled_corners_roi[4];
                    for (int i = 0; i < 4; ++i) {
                        scaled_corners_roi[i] = cv::Point2f(det.corners[i].x * scale_x_roi, det.corners[i].y * scale_y_roi);
                    }
                    
                    for (int i = 0; i < 4; ++i) {
                        cv::line(capture_frame, scaled_corners_roi[i],
                                scaled_corners_roi[(i + 1) % 4],
                                cv::Scalar(0, 0, 255), 3);
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


