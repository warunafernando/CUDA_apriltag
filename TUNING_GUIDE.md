# Camera Tuning with AprilTag

This guide explains how to tune your camera intrinsics using an AprilTag instead of a chessboard pattern.

## Quick Start

### 1. Interactive Tuning (Recommended)

```bash
./build/tune_camera_with_tag
```

**Controls:**
- `w`/`s`: Adjust fx (+/-10)
- `e`/`d`: Adjust fy (+/-10)
- `r`/`f`: Adjust cx (+/-5)
- `t`/`g`: Adjust cy (+/-5)
- `a`: Auto-estimate from current tag detection
- `SPACE`: Save intrinsics to JSON
- `q`: Quit

**How it works:**
1. Show an AprilTag to the camera
2. The program detects the tag and shows reprojection error
3. Adjust intrinsics using keyboard controls to minimize reprojection error
4. Press `a` to auto-estimate from the current tag
5. Press `SPACE` to save when satisfied

### 2. Auto-Tune Mode

```bash
./build/tune_camera_with_tag --auto --samples 20
```

**How it works:**
1. Show the tag to the camera from different angles and distances
2. Press `SPACE` to capture a sample when tag is detected
3. The program collects multiple samples and averages the intrinsics
4. Automatically saves to JSON when done

## Command Line Options

```bash
./build/tune_camera_with_tag [OPTIONS]

Options:
  --tag-size SIZE     Tag size in meters (default: 0.165)
  --file FILE         JSON file to save/load (default: camera_intrinsics.json)
  --auto              Use auto-tune mode
  --samples N         Number of samples for auto-tune (default: 20)
```

## JSON Format

The intrinsics are saved in a simple JSON format:

```json
{
  "fx": 1000.0,
  "fy": 1000.0,
  "cx": 640.0,
  "cy": 360.0,
  "k1": 0.0,
  "k2": 0.0,
  "p1": 0.0,
  "p2": 0.0,
  "k3": 0.0
}
```

## Integration

### C++

The main demo (`apriltag_demo`) automatically loads `camera_intrinsics.json` if it exists:

```cpp
CameraIntrinsics intr;
if (!CameraIntrinsicsUtils::loadFromJSON("camera_intrinsics.json", intr)) {
    // Use defaults
}
```

### Python

```python
import cuda_apriltag_py as cuda_apriltag

# Load from JSON
intrinsics = cuda_apriltag.load_intrinsics_from_json("camera_intrinsics.json")
detector = cuda_apriltag.CudaAprilTag(
    1280, 720, 2,
    intrinsics["fx"], intrinsics["fy"],
    intrinsics["cx"], intrinsics["cy"],
    0.165,
    intrinsics["k1"], intrinsics["k2"],
    intrinsics["p1"], intrinsics["p2"], intrinsics["k3"]
)

# Save to JSON
cuda_apriltag.save_intrinsics_to_json(
    "my_intrinsics.json",
    fx=1000.0, fy=1000.0, cx=640.0, cy=360.0
)
```

## Tips for Good Tuning

1. **Tag Placement:**
   - Place tag at different distances (close, medium, far)
   - Use different angles (straight-on, tilted, rotated)
   - Ensure tag is fully visible and in focus

2. **Reprojection Error:**
   - Lower is better (aim for < 1.0 pixels)
   - Adjust intrinsics to minimize this value
   - Use auto-estimate (`a` key) as a starting point

3. **Focal Length (fx, fy):**
   - Typically close to image width/height
   - Higher values = narrower field of view
   - Adjust to match tag size at different distances

4. **Principal Point (cx, cy):**
   - Usually near image center
   - Adjust if tag appears offset when it should be centered

5. **Distortion Coefficients:**
   - Start with zeros (no distortion)
   - Use chessboard calibration for accurate distortion values
   - Or manually adjust if you see barrel/pincushion distortion

## Comparison: Tag Tuning vs Chessboard Calibration

**Tag Tuning (This Tool):**
- ✅ Quick and convenient (uses existing AprilTag)
- ✅ Real-time feedback with reprojection error
- ✅ Interactive adjustment
- ⚠️ Less accurate than chessboard calibration
- ⚠️ Only estimates intrinsics (not distortion)

**Chessboard Calibration:**
- ✅ More accurate
- ✅ Provides distortion coefficients
- ✅ Industry standard method
- ⚠️ Requires printing chessboard pattern
- ⚠️ More setup time

**Recommendation:** Use tag tuning for quick setup, then refine with chessboard calibration for production use.

## Workflow

1. **Quick Setup:** Use tag tuning to get approximate intrinsics
2. **Save to JSON:** Press SPACE to save
3. **Test:** Run `apriltag_demo` - it will automatically load the JSON
4. **Refine:** Use chessboard calibration for final accurate values
5. **Update JSON:** Replace with calibrated values

## Troubleshooting

**"No tag detected"**
- Ensure tag is fully visible
- Check lighting (good contrast)
- Verify tag size parameter matches your tag

**High reprojection error**
- Try auto-estimate (`a` key)
- Adjust intrinsics manually
- Ensure tag is flat and in focus

**Intrinsics seem wrong**
- Start with auto-estimate
- Use chessboard calibration for accurate values
- Check that tag size is correct

