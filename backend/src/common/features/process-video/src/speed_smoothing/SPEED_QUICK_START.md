# Vehicle Speed Annotation - Quick Start Guide

## TL;DR

Add vehicle speed (in mph) to your video annotations in 3 ways:

### Method 1: Using main.py (Recommended for new processing)

```bash
python main.py happy1.mp4 --homography homography-points.json
```

### Method 2: Using example script (Recommended for existing detections)

```bash
python example_speed_annotation.py
```

### Method 3: Programmatic

```python
from src.annotate_video import annotate_video

annotate_video(
    video_path="video.mp4",
    detections_source="detections.jsonl",
    output_path="output.mp4",
    homography_file="homography-points.json"  # This enables speed!
)
```

## What You'll See

Video labels will display:

```
ID:1 | 15.2 mph    # Vehicle 1 at 15.2 mph
ID:3 | 28.7 mph    # Vehicle 3 at 28.7 mph
ID:5                # Vehicle 5 (speed calculating...)
```

## Requirements

1. **Homography file** (`homography-points.json`) - Already configured for your setup
2. **Video detections** - From YOLO tracking (JSONL format)
3. **Tracked vehicles** - Each vehicle needs a tracker ID

## Command-Line Options

### Full Processing with Speed

```bash
python main.py happy1.mp4 \
  --homography homography-points.json \
  --model yolov8s.pt \
  --conf 0.3 \
  --trail 10
```

### Annotate Existing Detections

Edit `example_speed_annotation.py` to set your paths:

```python
video_path = Path("your_video.mp4")
detections_path = Path("out/test_XX/detections.jsonl")
output_path = Path("out/test_XX/with-speed.mp4")
homography_file = "homography-points.json"
```

Then run:

```bash
python example_speed_annotation.py
```

## How It Works (Simple Version)

1. 🎯 Track vehicle position in each frame
2. 📍 Convert pixel position → geographic coordinates (lat/lng)
3. 📏 Calculate real-world distance between positions
4. ⏱️ Divide distance by time = speed!
5. 🏷️ Display speed next to vehicle ID

## Troubleshooting

### No speed showing?

- Vehicle needs to be tracked for at least 5 frames
- Check that homography file path is correct
- Verify tracker IDs are present in detections

### Speeds seem wrong?

- Check homography calibration in `homography-points.json`
- Ensure calibration points match the camera view
- Works best for vehicles on flat road surface

### Import errors?

- Make sure you're in the project root directory
- Check that `src/` directory is in your Python path

## What's Different Without Speed?

Omit `--homography` or `homography_file` parameter:

```bash
python main.py happy1.mp4  # No speed, just vehicle IDs
```

Labels will show only:

```
ID:1    # No speed info
ID:3
ID:5
```

## Files Involved

- **Input**: Your video + detections JSONL
- **Config**: `homography-points.json` (calibration data)
- **Output**: Annotated video with speeds
- **Code**: `src/annotate_video.py` + `src/estimate_distance.py`

## Example Workflow

```bash
# Step 1: Process video (creates detections + annotated video with speed)
python main.py happy1.mp4 --homography homography-points.json

# Output will be in: out/test_XX/overlay_supervision.mp4
# With labels showing: "ID:X | Y.Y mph"

# Step 2: Check results
open out/test_XX/overlay_supervision.mp4
```

Or for existing detections:

```bash
# Re-annotate with speed
python example_speed_annotation.py

# Output: out/test_37/detections-with-speed_annotated.mp4
```

## Tips

💡 **Speed accuracy**: Depends on homography calibration quality
💡 **Smoothing**: Speed is averaged over 5 frames for stability  
💡 **Units**: Currently displays mph (1 m/s = 2.237 mph)
💡 **Performance**: Minimal overhead (~same speed as without)

## Need More Details?

See full documentation:

- `SPEED_CALCULATION.md` - Technical details
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation info
- `src/estimate_distance.py` - Distance calculation code
- `src/annotate_video.py` - Annotation with speed

---

**Quick test**: Run `python example_speed_annotation.py` to see it in action! 🚗💨
