# GPU-Accelerated AprilTag Detection Engine for Jetson Orin NX

A high-performance, GPU-accelerated AprilTag detection system designed for FRC-style localization on NVIDIA Jetson Orin NX. This implementation achieves **~104 FPS** with **99.90% detection probability** at 720p input resolution.

## Features

- **GPU-Accelerated Pipeline**: Full CUDA implementation for preprocessing, edge detection, quad extraction, tag decoding, and pose estimation
- **High Performance**: ~104 FPS effective update rate using multi-rate detection (full-frame + ROI tracking)
- **Robust Detection**: 99.90% detection probability with low reprojection error (~0.535 pixels)
- **Quality Filtering**: Configurable thresholds for decision margin, reprojection error, and overall quality
- **ROI-Based Tracking**: Intelligent region-of-interest tracking for efficient multi-rate detection
- **Sub-pixel Refinement**: GPU-based corner refinement for improved pose accuracy
- **Adaptive Thresholding**: Local adaptive thresholding for robust edge detection in varying lighting
- **GPU NMS**: Non-maximum suppression on GPU to reduce false positives

## Performance Summary

### Test Results (1-Minute Test)

- **Total Frames Processed**: 6,227
- **Frames with Detections**: 6,221
- **Detection Probability**: **99.90%**
- **Average FPS**: **103.77 FPS**
- **Reprojection Error**: **0.535 pixels** (excellent accuracy)
- **Tag Detection**: Tag #7 detected consistently

### Timing Breakdown (per frame)

- **Preprocessing**: ~1.0 ms
  - Memory copy: ~0.35 ms
  - Host-to-Device: ~0.36 ms
  - BGR to Gray: ~0.24 ms
  - Decimation: ~0.08 ms

- **Detection Pipeline**: ~4.3-6.5 ms (full frame) / ~1.7-3.2 ms (ROI)
  - Gradient: ~0.12 ms
  - Edge Detection: ~0.70 ms
  - Quad Extraction: ~3.4-5.6 ms (OpenCV CPU) / ~0.8-1.2 ms (GPU)
  - Tag Decode: ~0.009 ms
  - Pose Estimation (PnP): ~0.04 ms

- **Total Pipeline**: ~5.3-7.5 ms per frame (full frame) / ~2.7-4.2 ms (ROI)

### Architecture

The system uses a multi-rate detection strategy:
- **Full-Frame Detection**: Runs every 30 frames (~4 FPS) to discover new tags
- **ROI Detection**: Runs at full frame rate (~104 FPS) to track previously detected tags
- **Effective Update Rate**: ~104 FPS for pose updates

## Requirements

- NVIDIA Jetson Orin NX (or compatible CUDA-capable device)
- CUDA Toolkit (tested with CUDA 11.4+)
- OpenCV 4.x (with CUDA support)
- CMake 3.10+
- C++17 compiler
- Python 3.x (optional, for Python bindings)

## Building

```bash
mkdir build && cd build
cmake ..
make -j4
```

This will build:
- `apriltag_demo`: Main demo application
- `libcuda_apriltag.so`: Shared library
- `cuda_apriltag_py`: Python bindings (optional)

## Usage

### C++ Demo

```bash
./build/apriltag_demo
```

The demo will:
- Open the camera at 1280x720 @ 120 FPS
- Run GPU-accelerated AprilTag detection
- Display detections with visualization
- Print FPS and timing statistics
- Save sample frames to `captures/` directory

### Python API

```python
import cuda_apriltag

detector = cuda_apriltag.PyCudaAprilTag(
    width=1280,
    height=720,
    decimation=2,
    fx=1000.0, fy=1000.0,  # Camera intrinsics
    cx=640.0, cy=360.0,
    tag_size_m=0.165  # Tag size in meters
)

detections = detector.detect(frame)
for det in detections:
    print(f"Tag ID: {det.id}, Pose: {det.tx}, {det.ty}, {det.tz}")
```

## Configuration

### Camera Intrinsics

Update camera intrinsics in `src/main.cpp`:

```cpp
CameraIntrinsics intr;
intr.fx = 1000.f;  // Focal length X
intr.fy = 1000.f;  // Focal length Y
intr.cx = width / 2.f;  // Principal point X
intr.cy = height / 2.f;  // Principal point Y
```

### Tag Size

Set the physical tag size (in meters):

```cpp
float tag_size_m = 0.165f;  // 16.5 cm tag
```

### Quality Filtering

Configure detection quality thresholds:

```cpp
detector.setMinQuality(0.01f);              // Minimum quality score (0-1)
detector.setMaxReprojectionError(500.0f);   // Max reprojection error (pixels)
detector.setMinDecisionMargin(0.05f);        // Minimum decode confidence
detector.setEnableSubpixelRefinement(true);  // Enable sub-pixel corner refinement
```

### Detection Mode

Choose between GPU and CPU quad extraction:

```cpp
detector.setUseGpuQuadExtraction(false);  // Use OpenCV CPU (more robust)
// or
detector.setUseGpuQuadExtraction(true);   // Use GPU (faster, ~2.85x speedup)
```

## Project Structure

```
CUDA_Apriltag/
├── include/              # Header files
│   ├── gpu_context.h    # CUDA context management
│   ├── image_preprocessor.h  # GPU image preprocessing
│   └── apriltag_gpu.h   # Main detector API
├── src/                 # Source files
│   ├── gpu_context.cpp
│   ├── image_preprocessor.cu
│   ├── apriltag_gpu.cu  # Main detection pipeline
│   └── main.cpp         # Demo application
├── python/              # Python bindings
│   └── binding.cpp
├── captures/            # Sample detection images
├── CMakeLists.txt
├── README.md
├── PERFORMANCE_REPORT.md
├── IMPLEMENTATION_SUMMARY.md
├── TIMING_BREAKDOWN.md
└── QUICK_START.md
```

## Implementation Phases

1. **Phase 0**: CPU baseline (apriltag3) for reference
2. **Phase 1**: GPU preprocess + CPU detection
3. **Phase 2**: Simple GPU detection
4. **Phase 3**: Optimized GPU detection (current)
5. **Phase 4**: Persistent kernel + ROI / multi-rate detection (current)

## Key Optimizations

- **Multi-rate Detection**: Full-frame detection every 30 frames, ROI tracking at full FPS
- **ROI Tracking**: Intelligent region-of-interest tracking with decay mechanism
- **Adaptive Thresholding**: Local adaptive thresholds for robust edge detection
- **GPU NMS**: Non-maximum suppression on GPU to reduce false positives
- **Sub-pixel Refinement**: GPU-based corner refinement for improved accuracy
- **Asynchronous Memory Transfers**: Overlapped H2D transfers with computation
- **Pinned Host Memory**: Fast host-to-device transfers

## Test Results

### 1-Minute Detection Test

**Test Configuration:**
- Camera: 1280x720 @ 120 FPS
- Working Resolution: 640x360 (decimation = 2)
- Tag: AprilTag 36h11, ID #7
- Tag Size: 0.165 meters
- Test Duration: 60 seconds

**Results:**
```
Total frames processed: 6,227
Frames with detections: 6,221
Detection probability: 99.90%
Average FPS: 103.77 FPS
Reprojection error: 0.535 pixels
Tag #7: detected 6,221 times (99.90% of frames)
```

**Performance Metrics:**
- **Preprocessing Time**: ~1.0 ms/frame
- **Full-Frame Detection**: ~4.3-6.5 ms/frame
- **ROI Detection**: ~1.7-3.2 ms/frame
- **Effective Update Rate**: ~104 FPS
- **Detection Accuracy**: 99.90%
- **Pose Accuracy**: 0.535 pixels reprojection error

**Quality Metrics:**
- **Decision Margin**: 0.8 (high confidence)
- **Reprojection Error**: 0.535 pixels (excellent)
- **Hamming Distance**: 0 (perfect decode)
- **Edge Strength**: High (strong edge detection)

### Comparison: GPU vs CPU Quad Extraction

**GPU Quad Extraction:**
- Average time: ~0.8-1.2 ms
- Detection rate: Comparable to CPU
- Speedup: ~2.85x faster than OpenCV CPU

**OpenCV CPU Quad Extraction:**
- Average time: ~3.4-5.6 ms
- Detection rate: 99.90%
- More robust for difficult lighting conditions

### Frame Capture Analysis

Sample frames captured during testing show:
- Accurate tag detection with proper corner alignment
- Correct coordinate scaling from working resolution to full frame
- Proper ROI tracking in multi-rate detection mode
- Low reprojection error (visual verification)

## Known Limitations

- Camera intrinsics are currently hardcoded (should be calibrated)
- Undistortion is not yet implemented (placeholder in code)
- Temporal filtering is implemented but not yet enabled
- GPU quad extraction is faster but slightly less robust than OpenCV CPU

## Future Enhancements

- [ ] Camera calibration integration
- [ ] Undistortion kernel implementation
- [ ] Temporal filtering for pose smoothing
- [ ] Multi-camera support (up to 4 cameras)
- [ ] NetworkTables integration for FRC
- [ ] Pose estimator integration
- [ ] Additional tag families (beyond 36h11)

## License

[Add your license here]

## Acknowledgments

- AprilTag library (original CPU implementation)
- OpenCV for computer vision primitives
- NVIDIA CUDA for GPU acceleration

## Contact

[Add contact information]
