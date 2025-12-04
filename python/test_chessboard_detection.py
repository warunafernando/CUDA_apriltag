#!/usr/bin/env python3
"""
Simple test script to verify chessboard detection is working.
Shows live camera feed with chessboard detection overlay.
"""

import cv2
import sys

def test_chessboard_detection(camera_id=0, board_width=6, board_height=8, headless=False, test_duration=10):
    """
    Test chessboard detection by showing live camera feed with detection overlay.
    
    Args:
        camera_id: Camera device ID
        board_width: Inner corners width
        board_height: Inner corners height
        headless: If True, run without GUI (save images instead)
        test_duration: Duration in seconds for headless mode
    """
    print("=== Chessboard Detection Test ===")
    print(f"Chessboard size: {board_width}x{board_height} inner corners")
    
    if headless:
        print(f"Headless mode: Testing for {test_duration} seconds")
        print("Images with detections will be saved to 'detection_test/'\n")
    else:
        print("Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Trying alternative camera IDs...")
        for alt_id in [1, 2, "/dev/video0"]:
            try:
                cap = cv2.VideoCapture(alt_id)
                if cap.isOpened():
                    print(f"Opened camera: {alt_id}")
                    break
            except:
                pass
        
        if not cap.isOpened():
            print("Failed to open any camera")
            return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    chessboard_size = (board_width, board_height)
    detection_count = 0
    frame_count = 0
    saved_count = 0
    
    # Check if GUI is available
    try:
        cv2.namedWindow("Chessboard Detection Test", cv2.WINDOW_NORMAL)
        gui_available = True
    except:
        gui_available = False
        headless = True
        print("GUI not available, running in headless mode")
    
    if headless:
        import os
        import time
        os.makedirs("detection_test", exist_ok=True)
        start_time = time.time()
    
    print("Show the chessboard to the camera...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find chessboard
        found, corners = cv2.findChessboardCorners(
            gray, 
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Refine corners if found
        if found:
            detection_count += 1
            # Refine corners for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw chessboard corners
            cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, found)
            
            # Status text
            status_text = "✓ CHESSBOARD DETECTED!"
            status_color = (0, 255, 0)
            bg_color = (0, 200, 0)
            
            # Draw background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )
            cv2.rectangle(frame, (10, 10), (20 + text_width, 50), bg_color, -1)
            cv2.putText(frame, status_text, (15, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Draw corner count
            corner_text = f"Corners: {len(corners_refined)}"
            cv2.putText(frame, corner_text, (15, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save detection in headless mode
            if headless:
                filename = f"detection_test/detection_{saved_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                saved_count += 1
                print(f"  Detection #{detection_count}: Saved {filename}")
        else:
            status_text = "No chessboard detected"
            status_color = (0, 0, 255)
            cv2.putText(frame, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Display statistics
        detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
        stats_text = f"Frames: {frame_count} | Detections: {detection_count} ({detection_rate:.1f}%)"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if gui_available and not headless:
            # Instructions
            cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 200, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            cv2.imshow("Chessboard Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Headless mode: check duration
        if headless:
            elapsed = time.time() - start_time
            if elapsed >= test_duration:
                print(f"\nTest duration ({test_duration}s) reached")
                break
            # Print progress every second
            if int(elapsed) != int(elapsed - 0.1):
                remaining = test_duration - elapsed
                print(f"  [{int(elapsed)}s/{test_duration}s] Frames: {frame_count}, Detections: {detection_count}", end='\r')
    
    cap.release()
    if gui_available:
        cv2.destroyAllWindows()
    
    print(f"\n=== Test Results ===")
    print(f"Total frames: {frame_count}")
    print(f"Detections: {detection_count}")
    if frame_count > 0:
        print(f"Detection rate: {detection_count / frame_count * 100:.1f}%")
    
    if headless and saved_count > 0:
        print(f"Saved {saved_count} detection images to 'detection_test/'")
    
    if detection_count > 0:
        print("\n✓ Chessboard detection is working!")
        return True
    else:
        print("\n✗ No chessboard detections. Check:")
        print("  - Chessboard is visible in frame")
        print("  - Chessboard size matches (6x8 inner corners)")
        print("  - Good lighting and contrast")
        print("  - Chessboard is flat and in focus")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test chessboard detection")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID (default: 0)")
    parser.add_argument("--board-width", type=int, default=6,
                       help="Inner corners width (default: 6)")
    parser.add_argument("--board-height", type=int, default=8,
                       help="Inner corners height (default: 8)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI (save images instead)")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Test duration in seconds for headless mode (default: 10.0)")
    
    args = parser.parse_args()
    
    success = test_chessboard_detection(args.camera, args.board_width, args.board_height,
                                       args.headless, args.duration)
    sys.exit(0 if success else 1)

