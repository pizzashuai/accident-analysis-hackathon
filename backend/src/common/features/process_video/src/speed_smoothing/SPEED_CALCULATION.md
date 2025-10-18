# Vehicle Speed Calculation

This document explains how vehicle speed is calculated and annotated in the video analysis system.

## Overview

Vehicle speeds are calculated using:

1. **Homography transformation** - converts image coordinates to real-world geographic coordinates (lat/lng)
2. **Haversine distance** - calculates the actual distance between two geographic points
3. **Frame-based timing** - determines elapsed time between positions using the video FPS

## How It Works

### 1. Position Tracking

- Each vehicle is tracked frame-by-frame using its tracker ID
- The center point of the bounding box is used as the vehicle's position
- Positions are converted to normalized coordinates (0-1 range)

### 2. Distance Calculation

For each vehicle:

- Historical positions are stored (last 30 frames)
- Speed is calculated using positions from 5 frames apart (for smoothing)
- Image coordinates are transformed to lat/lng using homography
- Real-world distance is calculated using the Haversine formula

### 3. Speed Calculation Formula

```
distance_meters = haversine_distance(geo_point_old, geo_point_new)
time_seconds = (current_frame - old_frame) / video_fps
speed_mps = distance_meters / time_seconds
speed_mph = speed_mps * 2.23694  # Convert m/s to mph
```

### 4. Display

- Speed is displayed next to the vehicle ID in the format: `ID:123 | 25.3 mph`
- Speed updates continuously as the vehicle moves
- Speed calculation requires at least 5 frames of tracking history

## Usage

### Basic Usage

```python
from src.annotate_video import annotate_video

annotate_video(
    video_path="input.mp4",
    detections_source="detections.jsonl",
    output_path="output.mp4",
    homography_file="homography-points.json"  # Enable speed calculation
)
```

### Without Speed Calculation

```python
# Omit homography_file to disable speed calculation
annotate_video(
    video_path="input.mp4",
    detections_source="detections.jsonl",
    output_path="output.mp4"
)
```

### Advanced Usage

```python
from src.annotate_video import VideoAnnotator

# Create annotator with custom settings
annotator = VideoAnnotator(
    trail_length=15,
    text_scale=0.6,
    homography_file="homography-points.json"
)

# Annotate from JSONL file
annotator.annotate_video_from_jsonl(
    original_video_path="input.mp4",
    jsonl_path="detections.jsonl",
    output_path="output.mp4"
)
```

## Homography Setup

The homography file maps image coordinates to geographic coordinates. It requires at least 4 point pairs:

```json
{
  "pairs": [
    {
      "id": 1,
      "a": { "xNorm": 0.355, "yNorm": 0.285 }, // Image coords (normalized)
      "b": { "lat": 47.6169, "lng": -122.1433 } // Geographic coords
    }
    // ... at least 3 more pairs
  ]
}
```

## Technical Details

### Smoothing

- Speed is calculated over 5 frames (not 1) to reduce noise
- Only the last 30 frames of position history are kept
- Speed updates every frame once enough history is available

### Coordinate Systems

- **Image coordinates**: Pixel positions normalized to 0-1 range
- **Geographic coordinates**: WGS84 latitude/longitude
- **Distance**: Calculated using Earth's curvature (Haversine formula)

### Limitations

- Accuracy depends on homography calibration quality
- Works best for vehicles on the ground plane
- Speed calculation starts after 5 frames of tracking
- Vertical movement (if any) is not accounted for

## Example Output

Video annotations will show:

```
ID:1 | 15.2 mph    # Vehicle 1 traveling at 15.2 mph
ID:3 | 28.7 mph    # Vehicle 3 traveling at 28.7 mph
ID:5                # Vehicle 5 (speed not yet calculated)
```

## Files Modified

- `src/annotate_video.py` - Main annotation module with speed calculation
- `src/estimate_distance.py` - Homography and distance calculation utilities

## See Also

- `example_speed_annotation.py` - Example usage script
- `homography-points.json` - Sample homography calibration data
- `estimate_distance.py` - Distance estimation implementation
