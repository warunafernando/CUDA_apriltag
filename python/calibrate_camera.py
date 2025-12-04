#!/usr/bin/env python3
"""
Camera Calibration Script for CUDA AprilTag Detector

This script helps you calibrate your camera using a chessboard pattern.
It captures images, finds chessboard corners, and saves the calibration to a file.

Usage:
    python calibrate_camera.py [--board-width W] [--board-height H] [--square-size S] [--output FILE]

Requirements:
    - A printed chessboard pattern (download from OpenCV or print your own)
    - At least 10-20 images of the chessboard from different angles
    - Images should be saved in a directory (default: 'calibration_images/')
"""

import cv2
import numpy as np
import cuda_apriltag_py as cuda_apriltag
import glob
import os
import argparse
from pathlib import Path


def capture_calibration_images(output_dir="calibration_images", camera_id=0, 
                              board_width=6, board_height=8, auto_capture=True, delay_seconds=10):
    """
    Capture calibration images from live camera.
    
    If auto_capture is True:
    - Automatically detects when chessboard is visible
    - Captures image after delay_seconds when chessboard is found
    - Press 'q' to quit
    
    If auto_capture is False:
    - Press SPACE to capture an image manually
    - Press 'q' to quit
    """
    print(f"\n=== Capturing Calibration Images ===")
    print(f"Images will be saved to: {output_dir}/")
    print(f"Chessboard: {board_width}x{board_height} inner corners")
    
    if auto_capture:
        print(f"Auto-capture mode: Will capture when chessboard detected (delay: {delay_seconds}s)")
        print("  - Show chessboard to camera")
        print("  - Wait for countdown and automatic capture")
        print("  - Press 'q' to finish capturing")
    else:
        print("Manual mode:")
        print("  - Press SPACE to capture an image")
        print("  - Press 'q' to finish capturing")
    
    print("  - Move the chessboard to different positions and angles")
    print("  - Aim for 15-20 images with good coverage\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    count = 0
    chessboard_size = (board_width, board_height)
    capturing = False
    capture_start_time = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find chessboard
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, 
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_FAST_CHECK + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # Draw chessboard if found
        if found:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, found)
            status_text = "Chessboard detected!"
            status_color = (0, 255, 0)
            
            if auto_capture and not capturing:
                # Start capture countdown
                capturing = True
                capture_start_time = cv2.getTickCount()
                print(f"Chessboard detected! Starting {delay_seconds}s countdown...")
        else:
            status_text = "No chessboard detected"
            status_color = (0, 0, 255)
            if auto_capture:
                capturing = False
                capture_start_time = None
        
        # Handle auto-capture countdown
        if auto_capture and capturing and capture_start_time is not None:
            elapsed = (cv2.getTickCount() - capture_start_time) / cv2.getTickFrequency()
            remaining = delay_seconds - elapsed
            
            if remaining > 0:
                # Show countdown
                countdown_text = f"Capturing in {int(remaining) + 1}..."
                cv2.putText(frame, countdown_text, (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                # Draw progress bar
                bar_width = 400
                bar_height = 30
                bar_x = (frame.shape[1] - bar_width) // 2
                bar_y = frame.shape[0] - 60
                progress = elapsed / delay_seconds
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + int(bar_width * progress), bar_y + bar_height), 
                             (0, 255, 0), -1)
            else:
                # Time's up, capture the image
                filename = os.path.join(output_dir, f"calib_{count:03d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename} ({count + 1} images)")
                count += 1
                capturing = False
                capture_start_time = None
                # Wait a bit before allowing next capture
                cv2.waitKey(500)
        
        # Display status
        cv2.putText(frame, f"Captured: {count} images", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if not auto_capture:
            cv2.putText(frame, "SPACE: Capture | 'q': Quit", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Calibration Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif not auto_capture and key == ord(' '):  # Spacebar (manual mode only)
            filename = os.path.join(output_dir, f"calib_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Captured: {filename}")
            count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {count} images to {output_dir}/")
    return count > 0


def calibrate_from_images(image_dir, board_width, board_height, square_size, output_file):
    """
    Calibrate camera from existing images in a directory.
    """
    print(f"\n=== Calibrating Camera ===")
    print(f"Image directory: {image_dir}")
    print(f"Chessboard size: {board_width}x{board_height} inner corners")
    print(f"Square size: {square_size}m")
    print(f"Output file: {output_file}\n")
    
    # Find all images
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    
    if not image_paths:
        print(f"Error: No images found in {image_dir}")
        print("Please provide images or use --capture to capture from camera")
        return False
    
    print(f"Found {len(image_paths)} images")
    print("Processing images...")
    
    # Run calibration
    success = cuda_apriltag.calibrate_camera(
        image_paths, board_width, board_height, square_size, output_file
    )
    
    if success:
        print(f"\n✓ Calibration successful!")
        print(f"  Saved to: {output_file}")
        
        # Load and display calibration results
        fx = fy = cx = cy = k1 = k2 = p1 = p2 = k3 = 0.0
        width = height = 0
        
        if cuda_apriltag.load_calibration(output_file, fx, fy, cx, cy, 
                                          k1, k2, p1, p2, k3, width, height):
            print(f"\nCalibration Parameters:")
            print(f"  Image size: {width}x{height}")
            print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
            print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
            print(f"  Distortion: k1={k1:.6f}, k2={k2:.6f}")
            print(f"            p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}")
        
        return True
    else:
        print("\n✗ Calibration failed!")
        print("  - Check that chessboard is visible in images")
        print("  - Verify board_width and board_height are correct")
        print("  - Ensure square_size is accurate")
        return False


def verify_calibration(calibration_file, camera_id=0):
    """
    Visual verification of calibration by showing undistorted camera feed.
    """
    print(f"\n=== Verifying Calibration ===")
    print(f"Loading calibration from: {calibration_file}")
    
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file not found: {calibration_file}")
        return False
    
    # Load calibration
    fx = fy = cx = cy = k1 = k2 = p1 = p2 = k3 = 0.0
    width = height = 0
    
    if not cuda_apriltag.load_calibration(calibration_file, fx, fy, cx, cy,
                                         k1, k2, p1, p2, k3, width, height):
        print("Error: Failed to load calibration file")
        return False
    
    # Create camera matrix and distortion coefficients
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width if width > 0 else 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height if height > 0 else 720)
    
    print("Press 'q' to quit")
    print("Left: Original (distorted) | Right: Undistorted")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort frame
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, 
                                   None, new_camera_matrix)
        
        # Combine images side by side
        combined = np.hstack([frame, undistorted])
        
        # Add labels
        cv2.putText(combined, "Original (Distorted)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Undistorted", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibration Verification", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Camera Calibration Tool for CUDA AprilTag Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-capture images from camera (detects chessboard, 10s delay)
  python calibrate_camera.py --capture
  
  # Manual capture mode (press SPACE to capture)
  python calibrate_camera.py --capture --manual
  
  # Custom delay for auto-capture (5 seconds)
  python calibrate_camera.py --capture --delay 5.0
  
  # Calibrate from existing images
  python calibrate_camera.py --image-dir calibration_images/
  
  # Custom chessboard size (6x8 inner corners, 1 inch squares - default)
  python calibrate_camera.py --board-width 6 --board-height 8 --square-size 0.0254
  
  # Verify calibration
  python calibrate_camera.py --verify calibration.yaml
        """
    )
    
    parser.add_argument("--capture", action="store_true",
                       help="Capture calibration images from camera")
    parser.add_argument("--image-dir", type=str, default="calibration_images",
                       help="Directory containing calibration images (default: calibration_images)")
    parser.add_argument("--board-width", type=int, default=6,
                       help="Number of inner corners in width (default: 6)")
    parser.add_argument("--board-height", type=int, default=8,
                       help="Number of inner corners in height (default: 8)")
    parser.add_argument("--square-size", type=float, default=0.0254,
                       help="Size of chessboard square in meters (default: 0.0254 = 1 inch)")
    parser.add_argument("--manual", action="store_true",
                       help="Use manual capture mode (press SPACE to capture)")
    parser.add_argument("--delay", type=float, default=10.0,
                       help="Delay in seconds before auto-capture (default: 10.0)")
    parser.add_argument("--output", type=str, default="camera_calibration.yaml",
                       help="Output calibration file (default: camera_calibration.yaml)")
    parser.add_argument("--verify", type=str, metavar="FILE",
                       help="Verify existing calibration file")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID for capture/verify (default: 0)")
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        verify_calibration(args.verify, args.camera)
        return
    
    # Capture mode
    if args.capture:
        auto_mode = not args.manual
        if not capture_calibration_images(args.image_dir, args.camera, 
                                         args.board_width, args.board_height,
                                         auto_mode, args.delay):
            print("Failed to capture images")
            return
        print("\nNow run calibration:")
        print(f"  python calibrate_camera.py --image-dir {args.image_dir}")
        return
    
    # Calibration mode
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        print("\nOptions:")
        print("  1. Use --capture to capture images from camera")
        print("  2. Place images in the directory and run again")
        return
    
    calibrate_from_images(
        args.image_dir, args.board_width, args.board_height,
        args.square_size, args.output
    )


if __name__ == "__main__":
    main()

