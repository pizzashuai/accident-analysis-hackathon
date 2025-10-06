# Backfill Detections - Usage Guide

## Overview

The `backfill_track_sequential` function provides a unified workflow for filling missing detections in a video track. It starts from an anchor segment (the longest, most confident detection segment) and walks backward and forward frame-by-frame to fill gaps.

## Key Features

1. **Anchor-based approach**: Finds the most reliable segment of detections and uses it as a reference
2. **Sequential processing**: Walks backward and forward from anchor, one frame at a time
3. **Two-tier filling strategy**:
   - **Detection-first**: Uses ROI-based YOLO detection with motion model prediction
   - **Optical flow fallback**: For short gaps (≤5 frames), propagates bbox using Lucas-Kanade optical flow
4. **Smart stopping**: Stops after 10 consecutive misses to avoid wasting computation

## Function Signature

```python
def backfill_track_sequential(
    track_id: int,
    detections: list[dict[str, Any]],
    video_path: str,
    max_frame: int,
    detector_handle: Any = None,
    img_size: dict[str, int] = {"width": 1280, "height": 720},
    detect_params: dict[str, Any] | None = None,
    class_id: int = 2,
    aoi_polygon: list[list[float]] | None = None,
    optical_flow_gap_threshold: int = 5,
    roi_scale: float = 2.0,
) -> dict[str, Any]
```

## Parameters

- `track_id`: Track ID to backfill
- `detections`: List of all detections (from JSONL file)
- `video_path`: Path to video file
- `max_frame`: Maximum frame number in video
- `detector_handle`: YOLO detector instance (if None, loads default yolov8s.pt)
- `img_size`: Video dimensions (width, height)
- `detect_params`: Detection parameters:
  - `conf`: Confidence threshold (default: 0.05)
  - `iou`: IoU threshold (default: 0.25)
  - `score_min`: Minimum score for candidate (default: 0.01)
- `class_id`: Class ID for detections (2 = car)
- `aoi_polygon`: Optional area of interest polygon to filter detections
- `optical_flow_gap_threshold`: Max gap for optical flow fallback (default: 5)
- `roi_scale`: ROI expansion factor (default: 2.0)

## Return Value

Returns a dictionary with:

- `new_detections`: List of newly created detections in JSONL format
- `summary`: Statistics dictionary with:
  - `attempted`: Number of frames attempted
  - `filled`: Number of frames successfully filled
  - `missed`: Number of frames that couldn't be filled
  - `detection_fills`: Number filled by detection
  - `optical_flow_fills`: Number filled by optical flow
- `fills`: Detailed fill information per frame

## Example Usage

```python
from pathlib import Path
from persist_detections import read_detections_from_jsonl
from backfill_detections import backfill_track_sequential

# Load detections
detections_path = Path("out/test_37/detections.jsonl")
detections = read_detections_from_jsonl(detections_path)

# Backfill track 17
result = backfill_track_sequential(
    track_id=17,
    detections=detections,
    video_path="happy1.mp4",
    max_frame=96,
    detector_handle=None,  # Will load default YOLO
    img_size={"width": 1280, "height": 720},
    detect_params={"conf": 0.05, "iou": 0.25, "score_min": 0.01},
    class_id=2,
    aoi_polygon=None,
    optical_flow_gap_threshold=5,
    roi_scale=2.0,
)

# Get results
new_detections = result["new_detections"]
summary = result["summary"]

print(f"Filled {summary['filled']} frames:")
print(f"  - Detection fills: {summary['detection_fills']}")
print(f"  - Optical flow fills: {summary['optical_flow_fills']}")

# Merge with original and save
all_detections = detections + new_detections
all_detections.sort(key=lambda x: x["frame"])

output_path = Path("out/test_37/detections-filled.jsonl")
with open(output_path, "w") as f:
    for det in all_detections:
        f.write(json.dumps(det) + "\n")
```

## How It Works

### 1. Find Anchor Segment

- Builds contiguous segments from existing detections
- Picks the longest segment with highest confidence
- Fits a linear motion model (constant velocity + size drift) on anchor

### 2. Walk Backward from Anchor

Starting from the frame before the anchor:

- Skip if frame already has detection
- Try ROI-based detection using motion model prediction
- If detection fails AND gap ≤ threshold: try optical flow propagation
- Stop after 10 consecutive misses

### 3. Walk Forward from Anchor

Same process as backward walk, but from anchor end going forward

### 4. Create Detection Records

Converts successful fills to standard JSONL format with:

- Frame number and timestamp
- Bounding box coordinates
- Track ID and class info
- Source indicator (detection or optical_flow)

## Tips

- **Lower confidence thresholds** for difficult tracks (e.g., `conf=0.03`)
- **Increase ROI scale** if objects move quickly (e.g., `roi_scale=2.5`)
- **Adjust optical flow threshold** based on video frame rate and motion speed
- **Use AOI polygon** to filter out false positives in irrelevant regions
- **Check summary statistics** to tune parameters for better fill rates

## Running the Script

```bash
cd /Users/shuaima/code/accident_analysis/accident-analysis-hackathon
python src/backfill_detections.py
```

The script will:

1. Load detections from `out/test_37/detections.jsonl`
2. Backfill track 17
3. Save merged detections to `out/test_37/detections-redone.jsonl`
4. Print summary with fill statistics
