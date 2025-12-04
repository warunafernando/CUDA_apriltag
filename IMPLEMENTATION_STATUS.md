# GPU AprilTag Engine Implementation Status

## ✅ Completed Features

### Phase 1: GPU Preprocess + CPU Detection
- ✅ Pinned host memory buffers
- ✅ Async GPU upload (H2D)
- ✅ GPU BGR→Gray conversion
- ✅ GPU decimation (1280×720 → 640×360)
- ✅ CPU AprilTag detection (OpenCV aruco)
- ✅ CPU PnP pose estimation
- ✅ Quaternion output (w, x, y, z)

### Phase 2: GPU Edge Detection
- ✅ GPU gradient magnitude kernel (optimized with shared memory)
- ✅ GPU edge thresholding
- ✅ Detailed timing breakdown per stage

### Phase 3: ROI Tracking & Multi-Rate Detection
- ✅ ROI tracking based on detected tags
- ✅ Multi-rate detection (full frame every 5 frames, ROI otherwise)
- ✅ Automatic ROI decay after 10 frames without detection
- ✅ Up to 16 concurrent ROIs

### Architecture
- ✅ C++ API (`GpuContext`, `ImagePreprocessor`, `AprilTagGpuDetector`)
- ✅ Python wrapper (`cuda_apriltag_py.CudaAprilTag`)
- ✅ Demo program (`apriltag_demo`)
- ✅ Comprehensive timing instrumentation

## 📊 Performance Results

**Test Configuration:**
- Input: 1280×720 @ 120 FPS (camera request)
- Working resolution: 640×360 (decimation=2)
- Platform: Jetson Orin NX

**Measured Performance:**
- **Throughput: ~59-60 FPS** (steady with 1 tag in view)
- **GPU stages:** <0.5 ms total (gradient + edge)
- **CPU decode:** 1.7-11 ms (ROI vs full frame)
- **Total detection:** 3.8-15 ms per frame

**Timing Breakdown (typical frame):**
```
PRE(ms): memcpy=0.34  h2d=0.33  bgr2gray=0.21  decim=0.08
DET(ms): grad=0.12  edge=0.05  quad=1.9  decode=2.1  pnp=0  total=4.2
```

**ROI Tracking Impact:**
- Full frame decode: ~10-11 ms
- ROI-only decode: ~1.7-3.2 ms (3-6x speedup)

## 🎯 Remaining Work for 120 FPS Target

To reach 120 FPS, the main bottleneck is **CPU-based tag decode** (OpenCV `detectMarkers`). Options:

1. **Full GPU AprilTag Decoder** (Phase 2 completion)
   - GPU-based quad extraction (contour following)
   - GPU-based tag bit sampling + Hamming decode
   - Estimated: 0.5-1 ms on GPU vs 1.7-11 ms on CPU

2. **Additional Optimizations**
   - GPU-based PnP (currently CPU, ~0.5 ms)
   - Persistent kernel for continuous processing
   - Multi-camera support (4 cameras as per requirements)

3. **Camera HAL Integration**
   - Abstract camera interface for IMX477/AR0234
   - Direct zero-copy from camera to GPU memory

## 📁 Project Structure

```
CUDA_Apriltag/
├── include/
│   ├── gpu_context.h          # CUDA context & memory management
│   ├── image_preprocessor.h    # GPU image preprocessing
│   └── apriltag_gpu.h         # GPU AprilTag detector
├── src/
│   ├── gpu_context.cpp
│   ├── image_preprocessor.cu   # CUDA kernels for preprocessing
│   ├── apriltag_gpu.cu        # CUDA kernels + detection logic
│   └── main.cpp               # Demo program
├── python/
│   └── binding.cpp            # Python wrapper (pybind11)
└── build/                      # Build directory
```

## 🚀 Usage

**Build:**
```bash
cd build && cmake .. && make -j$(nproc)
```

**Run Demo:**
```bash
./apriltag_demo [camera_index_or_path]
```

**Python API:**
```python
import cuda_apriltag_py
detector = cuda_apriltag_py.CudaAprilTag(1280, 720, 2, fx, fy, cx, cy, 0.165)
detections = detector.detect(frame)  # numpy array (H, W) or (H, W, 3)
```

## 📝 Notes

- Current implementation uses hybrid GPU/CPU approach
- GPU handles: preprocessing, gradients, edge detection
- CPU handles: quad extraction, tag decode, PnP (via OpenCV)
- ROI tracking significantly reduces decode time when tags are tracked
- Quaternion output is computed from rotation matrix
- All timing data is available via `DetectionTimings` struct

