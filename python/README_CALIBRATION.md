# Camera Calibration Guide

This guide explains how to calibrate your camera for accurate AprilTag pose estimation.

## Quick Start

### 1. Print a Chessboard Pattern

Download and print a chessboard pattern:
- **OpenCV standard**: 9x6 inner corners (10x7 squares)
- **Size**: Print at actual size (typically 25mm squares work well)
- **Material**: Print on a flat, rigid surface (cardboard or paper on a board)

You can generate a pattern using OpenCV:
```python
import cv2
import numpy as np

# Create 9x6 chessboard (10x7 squares)
pattern_size = (9, 6)  # Inner corners
square_size = 25  # mm
pattern = np.zeros((6 * square_size, 9 * square_size), dtype=np.uint8)

for i in range(6):
    for j in range(9):
        if (i + j) % 2 == 0:
            pattern[i*square_size:(i+1)*square_size, 
                   j*square_size:(j+1)*square_size] = 255

cv2.imwrite("chessboard.png", pattern)
```

### 2. Capture Calibration Images

**Option A: Use the calibration script**
```bash
python python/calibrate_camera.py --capture
```

- Press **SPACE** to capture an image
- Press **'q'** to finish
- Capture 15-20 images with:
  - Different angles (tilted, rotated)
  - Different distances
  - Different positions in frame
  - Good lighting

**Option B: Use existing images**
Place your calibration images in a directory (e.g., `calibration_images/`)

### 3. Run Calibration

```bash
python python/calibrate_camera.py --image-dir calibration_images/
```

**Parameters:**
- `--board-width`: Number of inner corners horizontally (default: 9)
- `--board-height`: Number of inner corners vertically (default: 6)
- `--square-size`: Size of each square in meters (default: 0.025 = 25mm)
- `--output`: Output file name (default: camera_calibration.yaml)

**Example with custom parameters:**
```bash
python python/calibrate_camera.py \
    --image-dir calibration_images/ \
    --board-width 9 \
    --board-height 6 \
    --square-size 0.025 \
    --output my_camera_calibration.yaml
```

### 4. Verify Calibration

View the undistorted camera feed:
```bash
python python/calibrate_camera.py --verify camera_calibration.yaml
```

You should see:
- **Left**: Original (distorted) image
- **Right**: Undistorted image
- Straight lines should appear straight in the undistorted view

### 5. Use Calibration with Detector

**Python:**
```python
import cuda_apriltag_py as cuda_apriltag

# Create detector from calibration file
detector = cuda_apriltag.create_from_calibration_file(
    "camera_calibration.yaml",
    width=1280, height=720,
    decimation=2,
    tag_size_m=0.165  # Your tag size in meters
)

# Or load manually
fx = fy = cx = cy = k1 = k2 = p1 = p2 = k3 = 0.0
width = height = 0
cuda_apriltag.load_calibration("camera_calibration.yaml", 
                               fx, fy, cx, cy, k1, k2, p1, p2, k3,
                               width, height)

detector = cuda_apriltag.CudaAprilTag(
    1280, 720, 2, fx, fy, cx, cy, 0.165,
    k1, k2, p1, p2, k3
)
```

## Calibration File Format

The calibration is saved as an OpenCV YAML file:
```yaml
%YAML:1.0
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0, cx, 0, fy, cy, 0, 0, 1 ]
distortion_coefficients: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ k1, k2, p1, p2, k3 ]
image_width: 1280
image_height: 720
rms_error: 0.5
```

## Tips for Good Calibration

1. **Image Quality**
   - Use good lighting (avoid shadows on chessboard)
   - Ensure chessboard is in focus
   - Avoid motion blur

2. **Coverage**
   - Capture images from different angles
   - Include corners and edges of the image
   - Vary the distance to camera

3. **Chessboard**
   - Keep it flat (no wrinkles or bends)
   - Ensure good contrast (black squares are black, white are white)
   - Use a rigid surface

4. **Number of Images**
   - Minimum: 10 images
   - Recommended: 15-20 images
   - More images = better calibration (up to ~30)

5. **RMS Error**
   - Good: < 0.5 pixels
   - Acceptable: < 1.0 pixels
   - Poor: > 1.0 pixels (recalibrate)

## Troubleshooting

**"No valid calibration images found"**
- Check that chessboard is visible in images
- Verify `board_width` and `board_height` match your pattern
- Ensure images are in the correct directory

**High RMS error (> 1.0 pixels)**
- Capture more images
- Ensure chessboard is flat and well-lit
- Check that square size is accurate
- Verify pattern dimensions

**Poor undistortion results**
- Recalibrate with more images
- Check that camera hasn't moved/focused since calibration
- Ensure calibration images match current camera settings

**AprilTag poses are inaccurate**
- Verify tag size is correct
- Check that calibration matches camera resolution
- Ensure calibration is recent (camera may have changed)

## Advanced Usage

### Custom Chessboard Sizes

For different chessboard patterns:
```bash
python python/calibrate_camera.py \
    --board-width 7 \
    --board-height 5 \
    --square-size 0.030 \
    --image-dir my_images/
```

### Batch Processing

Calibrate multiple cameras:
```bash
for camera in camera1 camera2 camera3; do
    python python/calibrate_camera.py \
        --image-dir ${camera}_images/ \
        --output ${camera}_calibration.yaml
done
```

## Integration with AprilTag Detection

Once calibrated, the detector will:
- Use accurate camera intrinsics for pose estimation
- Apply distortion correction (if implemented in preprocessor)
- Provide more accurate 3D pose estimates

The calibration parameters are used in:
- `ImagePreprocessor`: For undistortion (if enabled)
- `AprilTagGpuDetector`: For PnP pose estimation

