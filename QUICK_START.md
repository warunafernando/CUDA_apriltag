# Quick Start Guide

## Performance Summary
- **FPS:** 118.8 (99% of 120 FPS target)
- **Detection Rate:** 99.92% (7,122/7,128 frames)
- **Tag Detected:** #7 (36h11 family)

## Build & Run
```bash
mkdir build && cd build
cmake ..
make -j4
./apriltag_demo
```

## Key Files
- `README.md` - Project overview and usage
- `PERFORMANCE_REPORT.md` - Detailed performance analysis
- `TIMING_BREAKDOWN.md` - Per-stage timing breakdown
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

## Architecture
- **Preprocessing:** GPU (1.0 ms)
- **Detection:** Hybrid GPU/CPU (4.5 ms)
- **Strategy:** Multi-rate with ROI tracking

## Timing (Per Frame)
- Camera: 8.4 ms
- Preprocess: 1.0 ms (GPU)
- Detect: 4.5 ms (Hybrid)
- **Total: 8.5 ms → 118.8 FPS**
