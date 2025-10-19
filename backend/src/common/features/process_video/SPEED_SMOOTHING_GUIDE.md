# Speed Calculation and Smoothing Guide

## Overview

This guide explains how to calculate vehicle speeds from detection data with various smoothing techniques to eliminate impossible speeds (like 2000 mph) and produce accurate, realistic results.

## Quick Start

### 1. Run Speed Calculation with Multiple Smoothing Methods

```bash
cd /Users/shuaima/code/accident_analysis/accident-analysis-hackathon/backend
python3 src/common/features/process_video/test_speed_smoothing.py
```

This will:

- Process your detections JSONL file
- Apply 5 different smoothing methods
- Compare results and recommend the best method
- Save output files in `speed_test_output/`

### 2. Analyze Speed Distributions

```bash
python3 src/common/features/process_video/analyze_speeds.py
```

This will show:

- Speed statistics (avg, median, max, min)
- Speed distribution histograms
- Fastest tracks
- Potential outlier detection

## Smoothing Methods Comparison

Based on our testing with real data, here are the results:

| Method             | Max Speed | Avg Speed | Median   | Smoothness | Outlier Rejection |
| ------------------ | --------- | --------- | -------- | ---------- | ----------------- |
| **None**           | 63.95 mph | 5.95 mph  | 0.00 mph | ⭐         | ⭐                |
| **Moving Average** | 51.21 mph | 5.99 mph  | 1.42 mph | ⭐⭐⭐     | ⭐⭐⭐            |
| **Exponential MA** | 50.67 mph | 5.98 mph  | 1.47 mph | ⭐⭐⭐⭐   | ⭐⭐⭐⭐          |
| **Kalman Filter**  | 49.84 mph | 5.97 mph  | 1.71 mph | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐        |
| **Median + MA**    | 51.21 mph | 5.99 mph  | 1.42 mph | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐        |

### Recommendation

**Use `kalman_with_outlier_rejection` for production** - provides the smoothest results with best outlier rejection (max speed: 49.84 mph vs 63.95 mph without smoothing).

**Use `median_moving_average` for testing** - good balance of smoothness and responsiveness with excellent outlier rejection.

## How It Works

### 1. Speed Calculation

```python
# Calculate distance using homography
distance_meters = distance_estimator.estimate_distance(
    (old_x_norm, old_y_norm),
    (new_x_norm, new_y_norm)
)

# Calculate speed
time_diff = new_time - old_time
speed_mps = distance_meters / time_diff
speed_mph = speed_mps * 2.23694  # Convert to mph
```

### 2. Outlier Rejection

**Median Filter Method** (recommended):

```python
# Store recent speeds in a window
speed_history[track_id].append(raw_speed_mph)

# Calculate median to detect outliers
median_speed = np.median(speed_history[track_id])

# Reject if too far from median
if abs(raw_speed_mph - median_speed) > max_reasonable_speed * 0.5:
    speed_mph = median_speed  # Use median instead
else:
    speed_mph = np.mean(speed_history[track_id])  # Use moving average
```

**Bounds Checking**:

```python
# Clip speeds to reasonable range
if speed_mph < min_reasonable_speed or speed_mph > max_reasonable_speed:
    speed_mph = np.clip(speed_mph, min_reasonable_speed, max_reasonable_speed)
```

### 3. Smoothing Methods

**Kalman Filter** (best):

- Uses prediction and measurement update steps
- Automatically adapts to measurement uncertainty
- Best for noisy data

**Median + Moving Average**:

- Median filter removes outlier spikes
- Moving average smooths remaining data
- Good balance of robustness and responsiveness

**Exponential Moving Average**:

- Weighted average favoring recent values
- Smooth but responsive to changes
- Good for gradually changing speeds

## Configuration

### In `processor.py`:

```python
from processor import calculate_speeds_with_smoothing

output_path, stats = calculate_speeds_with_smoothing(
    jsonl_path=Path("detections.jsonl"),
    homography_file="homography-points.json",
    output_path=Path("detections_with_speeds.jsonl"),
    video_width=1280,
    video_height=720,
    smoothing_method="kalman_with_outlier_rejection",  # Recommended
    smoothing_window=5,
    max_reasonable_speed=100.0,  # Maximum reasonable speed in mph
    min_reasonable_speed=0.0,
    lookback_frames=5,  # Number of frames to look back for speed calculation
)
```

### Parameters:

- **`smoothing_method`**:

  - `"none"` - No smoothing (for debugging)
  - `"moving_average"` - Simple moving average
  - `"exponential_with_outlier_rejection"` - EMA with bounds checking
  - `"kalman_with_outlier_rejection"` - Kalman filter with bounds checking (recommended)
  - `"median_moving_average"` - Median filter + MA with aggressive outlier rejection

- **`smoothing_window`**: Window size for smoothing (default: 5)

  - Larger = smoother but less responsive
  - Smaller = more responsive but noisier

- **`max_reasonable_speed`**: Maximum speed to accept (default: 100 mph)

  - Speeds above this are clipped or rejected
  - Adjust based on your scenario (highway vs city)

- **`lookback_frames`**: Frames to look back for speed calculation (default: 5)
  - Larger = more stable but less responsive
  - Smaller = more responsive but noisier
  - At 30 fps: 5 frames = 0.167 seconds

## Fixing Impossible Speeds (2000 mph)

If you're getting impossible speeds, the issue is usually:

### 1. Track ID Switches

**Problem**: Tracker switches IDs between frames, causing huge position jumps
**Solution**: Use longer lookback frames (e.g., 5-10) to average over more data

### 2. Poor Homography Calibration

**Problem**: Homography transformation produces incorrect world coordinates
**Solution**: Recalibrate homography with more accurate ground control points

### 3. Incorrect Video Dimensions

**Problem**: Using wrong video width/height for normalization
**Solution**: Extract actual video dimensions from the video file

### 4. Insufficient Outlier Rejection

**Problem**: Spikes not being filtered out
**Solution**: Use `median_moving_average` or lower `max_reasonable_speed`

## Speed Distribution Analysis

From our test data:

```
0-5 mph       : 62.7% (stopped/slow moving)
5-15 mph      : 27.2% (city driving)
15-25 mph     : 3.4%  (residential)
25-35 mph     : 3.1%  (arterial roads)
35-50 mph     : 3.5%  (faster roads)
50-100 mph    : 0.1%  (outliers)
>100 mph      : 0.0%  (no outliers with smoothing!)
```

## Integration with Video Processing

The `processor.py` module already integrates speed calculation into the video processing pipeline:

```python
from processor import VideoProcessor

processor = VideoProcessor(
    model_path="yolov8s.pt",
    speed_smoothing_method="kalman",  # Use Kalman filter
    speed_smoothing_window=5,
)

# Process video with speed calculation
result = processor.process_video_detections(
    video_path=Path("video.mp4"),
    output_dir=Path("output/"),
    homography_file="homography-points.json",
)

# Speeds are automatically calculated and saved to detections.jsonl
```

## Output Format

The output JSONL file contains speed_mph for each detection:

```json
{
  "video_id": "video.mp4",
  "frame": 10,
  "time": 0.333,
  "track_id": 2,
  "class_name": "car",
  "bbox_xyxy": [268.5, 280.5, 418.1, 365.9],
  "center": [343.3, 323.2],
  "speed_mph": 23.14,
  "world_coords": [-122.1426, 47.6167]
}
```

## Troubleshooting

### Issue: All speeds are 0

**Cause**: No homography file provided or invalid file
**Fix**: Provide valid homography file with calibrated ground control points

### Issue: Speeds fluctuate wildly

**Cause**: Insufficient smoothing
**Fix**: Increase `smoothing_window` or use Kalman filter

### Issue: Speeds too smooth (not responsive)

**Cause**: Too much smoothing
**Fix**: Decrease `smoothing_window` or use `exponential` method

### Issue: Getting 2000+ mph speeds

**Cause**: Track ID switches or poor homography
**Fix**:

1. Lower `max_reasonable_speed` to 60-80 mph
2. Use `median_moving_average` method
3. Increase `lookback_frames` to 8-10
4. Recalibrate homography

## Testing Your Changes

After modifying speed calculation code:

```bash
# 1. Run tests
python3 src/common/features/process_video/test_speed_smoothing.py

# 2. Analyze results
python3 src/common/features/process_video/analyze_speeds.py

# 3. Check for outliers
# Look for "Potential Outlier Tracks" section
# Should show "None found! (Good)"
```

## References

- [Kalman Filter Tutorial](https://www.kalmanfilter.net/)
- [Moving Average Smoothing](https://en.wikipedia.org/wiki/Moving_average)
- [Median Filter](https://en.wikipedia.org/wiki/Median_filter)
