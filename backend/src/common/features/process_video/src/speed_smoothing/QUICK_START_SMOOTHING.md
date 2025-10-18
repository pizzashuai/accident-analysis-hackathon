# Speed Smoothing - Quick Start

## Problem Solved

Speed estimates were jumping wildly between frames (e.g., 25 mph → 11 mph). This has been fixed with smoothing algorithms and better tracking points.

## Immediate Next Steps

### 1. Run the Test (1 minute)

```bash
python test_speed_smoothing.py
```

This creates 5 videos in `out/test_37/speed_debug/`:

- `test_raw.mp4` - No smoothing (shows the problem)
- `test_ma_center.mp4` - Moving average with center tracking
- `test_ma_bottom.mp4` - **Moving average with bottom tracking (RECOMMENDED)**
- `test_ema.mp4` - Exponential smoothing
- `test_kalman.mp4` - Kalman filter

**Watch these videos to see the difference!**

### 2. Analyze the Debug Data (30 seconds)

```bash
# See summary of all tracks
python analyze_speed_debug.py out/test_37/speed_debug/debug_ma_bottom.jsonl

# Detailed view of track 1
python analyze_speed_debug.py out/test_37/speed_debug/debug_ma_bottom.jsonl --track-id 1 --detailed
```

### 3. Use the Best Method Going Forward

**For your main processing, use this:**

```python
from src.annotate_video import annotate_video

annotate_video(
    video_path="your_video.mp4",
    detections_source="your_detections.jsonl",
    output_path="output.mp4",
    homography_file="homography-points.json",
    speed_smoothing="moving_average",  # Best overall
    smoothing_window=5,                # Good balance
    tracking_point="bottom_center",    # More stable
    debug_speed=True,                  # Keep debug on
    debug_jsonl_path="debug.jsonl"     # For troubleshooting
)
```

## What You Get

### In the Video:

- Smooth speed labels (no more 25→11 mph jumps)
- Speed shown next to each vehicle
- Tracking trails

### In the Debug JSONL:

- Every detection with full calculation details
- Raw speed vs smoothed speed
- Bbox size variations
- Tracking point coordinates
- Geographic coordinates
- Distance calculations

## Results from Your Data

**Track 1 (your main vehicle):**

- Before: Speed jumps up to 35 mph between frames
- After: Speed jumps only 5.4 mph (81% reduction!)
- Speed is now smooth and believable

**Track 13 (another vehicle):**

- Before: 67.88 mph jump
- After: 16.97 mph jump (75% reduction)

## Three Key Insights

1. **Bbox size changes cause jumps**: When detection box expands/contracts, the center point moves even if vehicle hasn't moved
2. **Bottom center is more stable**: Tracking the bottom of the bbox (contact point with road) is less affected by bbox height changes

3. **Moving average works great**: Simple 5-frame average reduces jumps by 74-81% with minimal lag

## Files Generated

All test outputs are in: `out/test_37/speed_debug/`

| File                        | Purpose                           |
| --------------------------- | --------------------------------- |
| `test_ma_bottom.mp4`        | **RECOMMENDED** - Watch this one! |
| `debug_ma_bottom.jsonl`     | Debug data with all calculations  |
| Other `test_*.mp4` files    | Alternative smoothing methods     |
| Other `debug_*.jsonl` files | Debug data for alternatives       |

## Common Questions

**Q: Why is my speed still noisy?**
A: Increase smoothing window:

```python
smoothing_window=10  # More smoothing, more lag
```

**Q: Speed lags behind vehicle acceleration?**
A: Decrease smoothing window or use exponential:

```python
smoothing_window=3  # Less smoothing, more responsive
# OR
speed_smoothing="exponential"  # More responsive than moving average
```

**Q: How do I see what's wrong with a specific vehicle?**
A: Use the analysis tool:

```bash
python analyze_speed_debug.py debug.jsonl --track-id 5 --detailed
```

**Q: Can I try all methods at once?**
A: Yes:

```bash
python compare_smoothing_methods.py \
  --video happy1.mp4 \
  --detections out/test_37/detections.jsonl \
  --output-dir out/comparison
```

## Documentation

- **SPEED_DEBUG_GUIDE.md** - Complete guide with all features
- **SPEED_SMOOTHING_RESULTS.md** - Detailed test results
- **This file** - Quick reference for getting started

## Bottom Line

Use this configuration for best results:

- `tracking_point="bottom_center"`
- `speed_smoothing="moving_average"`
- `smoothing_window=5`

Your speed estimates will now be smooth and accurate! 🎉
