# Speed Estimation Debug and Smoothing Guide

## Problem Statement

Speed estimates can jump erratically between frames (e.g., 25 mph → 11 mph) due to:

1. **Bounding box size variations** - Detection boxes expand/contract, causing the tracking point to jump
2. **Detection noise** - Small pixel-level jitter in detections
3. **No temporal smoothing** - Each frame's speed is calculated independently

## Solution Overview

This guide shows how to:

1. **Debug speed calculations** - Generate detailed debug JSONL files with all intermediate data
2. **Apply smoothing** - Use various algorithms to smooth speed estimates
3. **Try different tracking points** - Use bottom-center instead of bbox center
4. **Analyze results** - Compare different approaches and identify issues

## Quick Start

### 1. Generate Debug Data with Different Smoothing Methods

```bash
# Compare all smoothing methods
python compare_smoothing_methods.py \
  --video happy1.mp4 \
  --detections out/test_37/detections.jsonl \
  --output-dir out/smoothing_comparison \
  --homography homography-points.json \
  --tracking-point center
```

This generates 4 annotated videos and debug files:

- `annotated_none.mp4` - No smoothing (raw speeds)
- `annotated_moving_average.mp4` - Moving average (default)
- `annotated_exponential.mp4` - Exponential moving average
- `annotated_kalman.mp4` - Kalman filter

Each has a corresponding `debug_*.jsonl` file with detailed calculation data.

### 2. Analyze Debug Data

```bash
# View summary of all tracks
python analyze_speed_debug.py out/smoothing_comparison/debug_moving_average.jsonl

# Analyze specific track in detail
python analyze_speed_debug.py out/smoothing_comparison/debug_moving_average.jsonl --track-id 1 --detailed

# Show first 20 frames of detailed data
python analyze_speed_debug.py out/smoothing_comparison/debug_moving_average.jsonl --track-id 1 --detailed --num-frames 20
```

### 3. Try Different Tracking Points

```bash
# Use bottom center instead of bbox center
python compare_smoothing_methods.py \
  --video happy1.mp4 \
  --detections out/test_37/detections.jsonl \
  --output-dir out/tracking_bottom_center \
  --homography homography-points.json \
  --tracking-point bottom_center
```

Bottom center is more stable because:

- It tracks the contact point with the ground
- Less affected by bbox height variations
- More physically meaningful for vehicles

## Smoothing Methods

### 1. None (Raw Speeds)

- **Use case**: Debugging, understanding raw data
- **Pros**: No lag, see actual calculations
- **Cons**: Very noisy, large jumps between frames

### 2. Moving Average (Default)

- **Algorithm**: Average of last N speed values
- **Parameters**: `smoothing_window` (default: 5)
- **Use case**: General purpose smoothing
- **Pros**: Simple, effective, easy to understand
- **Cons**: Introduces lag equal to window size

```python
from src.annotate_video import annotate_video

annotate_video(
    video_path="happy1.mp4",
    detections_source="out/test_37/detections.jsonl",
    output_path="out/test_smoothed.mp4",
    homography_file="homography-points.json",
    speed_smoothing="moving_average",
    smoothing_window=5,  # Average over 5 measurements
    debug_speed=True,
    debug_jsonl_path="out/debug.jsonl"
)
```

### 3. Exponential Moving Average

- **Algorithm**: Weighted average giving more weight to recent values
- **Parameters**: `ema_alpha` (default: 0.3, fixed in code)
- **Use case**: When you want smoothing with less lag
- **Pros**: Responsive to changes, smooth output
- **Cons**: Requires tuning alpha parameter

Formula: `new_ema = alpha * new_value + (1 - alpha) * old_ema`

```python
annotate_video(
    video_path="happy1.mp4",
    detections_source="out/test_37/detections.jsonl",
    output_path="out/test_ema.mp4",
    homography_file="homography-points.json",
    speed_smoothing="exponential",
    debug_speed=True,
    debug_jsonl_path="out/debug_ema.jsonl"
)
```

### 4. Kalman Filter

- **Algorithm**: Optimal estimator for linear systems
- **Parameters**: Q=0.1 (process noise), R=2.0 (measurement noise)
- **Use case**: When you want optimal smoothing with uncertainty modeling
- **Pros**: Theoretically optimal, handles noise well
- **Cons**: More complex, requires parameter tuning

```python
annotate_video(
    video_path="happy1.mp4",
    detections_source="out/test_37/detections.jsonl",
    output_path="out/test_kalman.mp4",
    homography_file="homography-points.json",
    speed_smoothing="kalman",
    debug_speed=True,
    debug_jsonl_path="out/debug_kalman.jsonl"
)
```

## Debug JSONL Format

Each line in the debug JSONL contains detailed information about one detection:

```json
{
  "frame": 10,
  "time": 0.345,
  "track_id": 1,
  "bbox_xyxy": [265.67, 279.49, 420.64, 366.4],
  "bbox_size": {
    "width": 154.97,
    "height": 86.91
  },
  "tracking_point": "center",
  "track_point_pixel": {
    "x": 343.15,
    "y": 322.95
  },
  "track_point_norm": {
    "x": 0.2679,
    "y": 0.4485
  },
  "speed_calc": {
    "frames_used": 5,
    "old_frame": 5,
    "new_frame": 10,
    "old_point_norm": {
      "x": 0.265,
      "y": 0.445
    },
    "new_point_norm": {
      "x": 0.2679,
      "y": 0.4485
    },
    "old_geo": {
      "lat": 37.7745,
      "lng": -122.4194
    },
    "new_geo": {
      "lat": 37.7746,
      "lng": -122.4193
    },
    "distance_meters": 5.23,
    "time_diff": 0.172,
    "speed_mps": 30.41,
    "raw_speed_mph": 68.03,
    "smoothed_speed_mph": 45.67,
    "smoothing_method": "moving_average"
  }
}
```

### Key Fields for Debugging

1. **bbox_size** - Monitor width/height changes that cause tracking point jumps
2. **track_point_pixel** - The actual pixel being tracked
3. **track_point_norm** - Normalized coordinates (0-1 range)
4. **old_point_norm / new_point_norm** - Points used for distance calculation
5. **distance_meters** - Calculated distance between frames
6. **raw_speed_mph** - Before smoothing
7. **smoothed_speed_mph** - After smoothing

## Common Issues and Solutions

### Issue 1: Large Speed Jumps

**Symptoms**: Speed jumps from 25 mph to 11 mph between frames

**Diagnosis**:

```bash
python analyze_speed_debug.py out/debug.jsonl --track-id 1
```

Look for:

- Large `bbox_size` variations (width/height changes)
- Large differences between consecutive `track_point_pixel` values

**Solutions**:

1. Use `tracking_point="bottom_center"` - more stable for vehicles
2. Increase `smoothing_window` to 7 or 10
3. Try Kalman filter: `speed_smoothing="kalman"`

### Issue 2: Speed Lags Behind Actual Speed

**Symptoms**: Vehicle appears to accelerate but speed shown lags

**Diagnosis**: Compare raw vs smoothed speeds in debug file

**Solutions**:

1. Reduce `smoothing_window` from 5 to 3
2. Use exponential smoothing: `speed_smoothing="exponential"`
3. Adjust `ema_alpha` in code (higher = more responsive)

### Issue 3: Still Noisy After Smoothing

**Symptoms**: Speed still varies too much frame-to-frame

**Diagnosis**: Check `max_jump` values in analysis output

**Solutions**:

1. Increase `smoothing_window` to 10 or 15
2. Use Kalman filter with higher R (measurement noise)
3. Calculate speed over more frames (edit code: change history[-5] to history[-10])

## Example Workflow

### Step 1: Generate Videos with All Methods

```bash
python compare_smoothing_methods.py \
  --video happy1.mp4 \
  --detections out/test_37/detections.jsonl \
  --output-dir out/comparison
```

### Step 2: Watch Videos Side-by-Side

Compare the 4 output videos to see which looks best visually.

### Step 3: Analyze the Best Method

```bash
# If moving_average looked best
python analyze_speed_debug.py out/comparison/debug_moving_average.jsonl
```

### Step 4: Deep Dive on Problem Track

```bash
# If track 1 still has issues
python analyze_speed_debug.py out/comparison/debug_moving_average.jsonl \
  --track-id 1 --detailed --num-frames 30
```

Look at the detailed output to see:

- How bbox size changes frame to frame
- How tracking point moves
- How distance calculations work
- Raw vs smoothed speed progression

### Step 5: Try Alternative Tracking Point

```bash
python compare_smoothing_methods.py \
  --video happy1.mp4 \
  --detections out/test_37/detections.jsonl \
  --output-dir out/bottom_center \
  --tracking-point bottom_center
```

### Step 6: Compare Results

Watch both sets of videos and analyze the debug files to see which combination works best.

## Tuning Parameters

### Smoothing Window (Moving Average)

- **3**: Very responsive, some noise
- **5**: Default, good balance
- **7**: Smoother, slight lag
- **10**: Very smooth, noticeable lag
- **15**: Extremely smooth, significant lag

### EMA Alpha (Exponential)

Edit in `src/annotate_video.py`, line ~94:

```python
self.ema_alpha = 0.3  # Change this value
```

- **0.1**: Very smooth, slow response
- **0.3**: Default, good balance
- **0.5**: More responsive
- **0.7**: Very responsive, less smoothing

### Kalman Filter Q and R

Edit in `src/annotate_video.py`, line ~482-483:

```python
Q = 0.1  # Process noise - how much you expect speed to change
R = 2.0  # Measurement noise - how much you trust measurements
```

- Higher Q: More responsive to changes
- Higher R: More smoothing (less trust in measurements)

## Advanced: Custom Smoothing

To implement your own smoothing algorithm:

1. Add a new method to `_apply_smoothing()` in `src/annotate_video.py`
2. Add storage for your algorithm's state in `__init__()`
3. Add your method name to the `speed_smoothing` choices

Example:

```python
elif self.speed_smoothing == "median":
    # Median filter
    if tracker_id not in self.raw_speeds:
        self.raw_speeds[tracker_id] = deque(maxlen=self.smoothing_window)

    self.raw_speeds[tracker_id].append(raw_speed)
    return statistics.median(self.raw_speeds[tracker_id])
```

## Recommendations

Based on common scenarios:

### For Highway/Freeway (consistent speed)

- **Method**: Moving average
- **Window**: 7-10
- **Tracking**: bottom_center

### For City Streets (variable speed)

- **Method**: Exponential with alpha=0.4
- **Tracking**: bottom_center

### For Intersections (start/stop)

- **Method**: Kalman filter
- **Tracking**: bottom_center

### For Research/Analysis (need accurate data)

- **Method**: None (raw) + post-processing
- Generate debug JSONL
- Apply offline smoothing in post-processing

## Troubleshooting

### Speed Shows as NaN or 0

- Check homography file exists and is valid
- Ensure video has been tracked (track_id present)
- Check if vehicle has enough motion history (needs 5+ frames)

### Debug File Not Generated

- Ensure `debug_speed=True`
- Ensure `debug_jsonl_path` is set
- Check file permissions in output directory

### Different Results Each Run

- Ensure you're using the same detection file
- Kalman filter has slight randomness in initialization

## Summary

The best approach for smooth speed estimates:

1. **Use bottom_center tracking point** - more stable than center
2. **Start with moving average (window=5)** - simple and effective
3. **Generate debug JSONL** - always enable for troubleshooting
4. **Analyze results** - use the analysis script to verify
5. **Tune if needed** - adjust window size or try different methods
6. **Iterate** - compare videos and debug data until satisfied
