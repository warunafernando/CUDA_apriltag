#!/usr/bin/env python3
"""
Example script demonstrating camera calibration integration with CUDA AprilTag detector.

This script shows how to:
1. Calibrate a camera from chessboard images
2. Load calibration from a file
3. Use calibration with the AprilTag detector
"""

import cv2
import numpy as np
import cuda_apriltag_py as cuda_apriltag
import glob
import os

def calibrate_camera_example():
    """Example: Calibrate camera from chessboard images"""
    print("=== Camera Calibration Example ===")
    
    # Find all calibration images
    image_paths = sorted(glob.glob("calibration_images/*.jpg") + 
                        glob.glob("calibration_images/*.png"))
    
    if not image_paths:
        print("No calibration images found in 'calibration_images/' directory")
        print("Please capture chessboard images and place them in that directory")
        return False
    
    print(f"Found {len(image_paths)} calibration images")
    
    # Chessboard parameters
    board_width = 9   # Inner corners (not squares)
    board_height = 6
    square_size = 0.025  # 25mm squares (adjust to your chessboard)
    output_file = "camera_calibration.yaml"
    
    # Run calibration
    success = cuda_apriltag.calibrate_camera(
        image_paths, board_width, board_height, square_size, output_file
    )
    
    if success:
        print(f"Calibration saved to {output_file}")
        return True
    else:
        print("Calibration failed!")
        return False


def load_and_use_calibration_example():
    """Example: Load calibration and use with detector"""
    print("\n=== Using Calibration with Detector ===")
    
    calibration_file = "camera_calibration.yaml"
    
    if not os.path.exists(calibration_file):
        print(f"Calibration file {calibration_file} not found!")
        print("Run calibrate_camera_example() first")
        return
    
    # Method 1: Load calibration parameters manually
    fx = fy = cx = cy = k1 = k2 = p1 = p2 = k3 = 0.0
    width = height = 0
    
    success = cuda_apriltag.load_calibration(
        calibration_file, fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height
    )
    
    if not success:
        print("Failed to load calibration")
        return
    
    print(f"Loaded calibration:")
    print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Distortion: k1={k1:.4f}, k2={k2:.4f}, p1={p1:.4f}, p2={p2:.4f}, k3={k3:.4f}")
    print(f"  Image size: {width}x{height}")
    
    # Create detector with calibration
    tag_size_m = 0.165  # 16.5 cm tag
    detector = cuda_apriltag.CudaAprilTag(
        width, height, decimation=2,
        fx, fy, cx, cy, tag_size_m,
        k1, k2, p1, p2, k3
    )
    
    print("Detector created with calibration!")
    
    # Method 2: Use factory function (simpler)
    detector2 = cuda_apriltag.create_from_calibration_file(
        calibration_file, width=0, height=0, decimation=2, tag_size_m=0.165
    )
    
    print("Detector created from calibration file!")
    
    return detector


def detect_with_calibration_example(detector):
    """Example: Detect tags using calibrated detector"""
    print("\n=== Detection with Calibration ===")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        detections = detector.detect(gray)
        
        # Draw detections
        for det in detections:
            # Draw quad
            corners = det["corners"]
            for i in range(4):
                pt1 = tuple(map(int, corners[i]))
                pt2 = tuple(map(int, corners[(i + 1) % 4]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            
            # Draw tag ID
            pose = det["pose"]
            cv2.putText(frame, f"ID: {det['id']}", 
                       tuple(map(int, corners[0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Print pose information
            print(f"Tag {det['id']}: tx={pose['tx']:.3f}, ty={pose['ty']:.3f}, tz={pose['tz']:.3f}")
            print(f"  Quality: {det['quality']:.3f}, Reprojection error: {det['reprojection_error']:.3f}px")
        
        cv2.imshow("AprilTag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: Calibrate camera (run once)
    # calibrate_camera_example()
    
    # Step 2: Load and use calibration
    detector = load_and_use_calibration_example()
    
    # Step 3: Detect tags with calibrated detector
    if detector:
        detect_with_calibration_example(detector)

