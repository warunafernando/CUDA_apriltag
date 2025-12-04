#include "apriltag_gpu.h"
#include "image_preprocessor.h"
#include "gpu_context.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>
#include <thread>

struct BenchmarkStats {
    int total_frames = 0;
    int frames_with_detections = 0;
    double total_quad_time_ms = 0.0;
    double total_detection_time_ms = 0.0;
    double min_quad_time_ms = 1e6;
    double max_quad_time_ms = 0.0;
    std::map<int, int> tag_detection_counts;
};

void runBenchmark(bool use_gpu_quad, const std::string& mode_name, double duration_seconds = 60.0) {
    std::cout << "\n========================================\n";
    std::cout << "Running " << mode_name << " Quad Extraction Benchmark\n";
    std::cout << "Duration: " << duration_seconds << " seconds\n";
    std::cout << "========================================\n";
    
    // Initialize camera
    cv::VideoCapture cap;
    if (!cap.open(0, cv::CAP_V4L2)) {
        if (!cap.open(0)) {
            std::cerr << "Failed to open camera\n";
            return;
        }
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 120);
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Camera: " << width << "x" << height << "\n";
    
    // Initialize GPU context
    GpuContext gpu_ctx;
    
    // Initialize preprocessor
    int decimation = 2;
    ImagePreprocessor preprocessor(gpu_ctx, width, height, decimation, nullptr);
    
    // Initialize detector
    float tag_size = 0.1524f;  // 6 inches in meters
    cv::Matx33f K(800.0f, 0.0f, width / 2.0f,
                  0.0f, 800.0f, height / 2.0f,
                  0.0f, 0.0f, 1.0f);
    AprilTagGpuDetector detector(gpu_ctx, preprocessor.workingWidth(), preprocessor.workingHeight(), tag_size, K);
    detector.setUseGpuQuadExtraction(use_gpu_quad);
    
    std::cout << "Quad Extraction Mode: " << (use_gpu_quad ? "GPU" : "OpenCV CPU") << "\n\n";
    
    BenchmarkStats stats;
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::duration<double>(duration_seconds);
    
    cv::Mat frame;
    int frame_count = 0;
    
    while (std::chrono::steady_clock::now() < end_time) {
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Preprocess
        unsigned char* gray_dev = preprocessor.preprocess(frame, nullptr);
        
        // Detect
        DetectionTimings timings;
        auto detections = detector.detect(gray_dev, true, &timings);
        
        // Update statistics
        stats.total_frames++;
        stats.total_quad_time_ms += timings.quad_ms;
        stats.total_detection_time_ms += timings.total_ms;
        stats.min_quad_time_ms = std::min(stats.min_quad_time_ms, static_cast<double>(timings.quad_ms));
        stats.max_quad_time_ms = std::max(stats.max_quad_time_ms, static_cast<double>(timings.quad_ms));
        
        if (!detections.empty()) {
            stats.frames_with_detections++;
            for (const auto& det : detections) {
                stats.tag_detection_counts[det.id]++;
            }
        }
        
        frame_count++;
        
        // Print progress every 5 seconds
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        if (frame_count % 600 == 0) {  // ~10 FPS * 60 = 600 frames
            double fps = stats.total_frames / elapsed;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                      << (elapsed / duration_seconds * 100.0) << "% | "
                      << "FPS: " << std::setprecision(2) << fps << " | "
                      << "Avg Quad Time: " << std::setprecision(3) 
                      << (stats.total_quad_time_ms / stats.total_frames) << " ms\n";
        }
    }
    
    auto total_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    
    // Print final statistics
    std::cout << "\n========== " << mode_name << " RESULTS ==========\n";
    std::cout << "Total frames processed: " << stats.total_frames << "\n";
    std::cout << "Frames with detections: " << stats.frames_with_detections << "\n";
    std::cout << "Detection rate: " << std::fixed << std::setprecision(2)
              << (stats.frames_with_detections * 100.0 / stats.total_frames) << "%\n";
    std::cout << "Average FPS: " << std::setprecision(2) 
              << (stats.total_frames / total_elapsed) << "\n";
    std::cout << "\nQuad Extraction Timing:\n";
    std::cout << "  Average: " << std::setprecision(3) 
              << (stats.total_quad_time_ms / stats.total_frames) << " ms\n";
    std::cout << "  Min: " << std::setprecision(3) << stats.min_quad_time_ms << " ms\n";
    std::cout << "  Max: " << std::setprecision(3) << stats.max_quad_time_ms << " ms\n";
    std::cout << "  Total: " << std::setprecision(2) 
              << stats.total_quad_time_ms << " ms\n";
    std::cout << "\nTotal Detection Timing:\n";
    std::cout << "  Average: " << std::setprecision(3) 
              << (stats.total_detection_time_ms / stats.total_frames) << " ms\n";
    std::cout << "  Average FPS: " << std::setprecision(2)
              << (1000.0 / (stats.total_detection_time_ms / stats.total_frames)) << "\n";
    
    if (!stats.tag_detection_counts.empty()) {
        std::cout << "\nTag Detection Counts:\n";
        for (const auto& pair : stats.tag_detection_counts) {
            double tag_prob = (pair.second * 100.0) / stats.total_frames;
            std::cout << "  Tag #" << pair.first << ": " << pair.second 
                      << " detections (" << std::setprecision(2) << tag_prob << "%)\n";
        }
    }
    std::cout << "========================================\n";
}

int main(int argc, char* argv[]) {
    double duration = 60.0;
    if (argc > 1) {
        duration = std::stod(argv[1]);
    }
    
    std::cout << "Quad Extraction Efficiency Comparison\n";
    std::cout << "=====================================\n";
    
    // Run GPU quad extraction benchmark
    runBenchmark(true, "GPU", duration);
    
    // Wait a bit between tests
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Run OpenCV CPU quad extraction benchmark
    runBenchmark(false, "OpenCV CPU", duration);
    
    std::cout << "\nComparison Complete!\n";
    
    return 0;
}

