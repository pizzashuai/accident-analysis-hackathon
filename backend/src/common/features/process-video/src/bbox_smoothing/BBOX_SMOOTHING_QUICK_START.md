# Bounding Box Smoothing - Quick Start Guide

## Overview

Bounding box smoothing eliminates jitter in vehicle detection boxes, resulting in:

- ✅ **22% smoother speed estimates**
- ✅ **51% more stable bounding boxes**
- ✅ **More accurate vehicle tracking**

## Quick Usage

### Default (Recommended)

The system now uses **Kalman filter** smoothing by default when processing videos with speed calculation:

```bash
python main.py happy1.mp4 --homography homography-points.json
```

That's it! Bbox smoothing is automatically enabled with optimal settings.

### Custom Configuration

```python
from src.annotate_video import annotate_video

annotate_video(
    video_path="video.mp4",
    detections_source="detections.jsonl",
    homography_file="homography-points.json",
    bbox_smoothing="kalman",  # Choose your method
    bbox_smoothing_window=5,
)
```

## Available Methods

| Method             | Best For               | Speed Improvement | Complexity |
| ------------------ | ---------------------- | ----------------- | ---------- |
| **kalman** 🏆      | Maximum speed accuracy | 22.2%             | Medium     |
| **exponential** 🥈 | Balance of all metrics | 19.8%             | Low        |
| **moving_average** | Simple averaging       | 17.3%             | Low        |
| **iou_weighted**   | Variable vehicle sizes | 16.0%             | Medium     |
| **none**           | No smoothing (debug)   | 0% (baseline)     | -          |

## When to Use Each Method

### Kalman Filter (Default) 🏆

```python
bbox_smoothing="kalman"
```

**Best for:**

- Speed estimation (primary use case)
- Maximum smoothness
- Statistical optimality

**Pros:**

- Lowest frame-to-frame speed changes (0.63 mph)
- Adaptive to noise
- Best overall performance

**Cons:**

- Slightly more computational overhead
- Requires parameter tuning for edge cases

### Exponential Moving Average 🥈

```python
bbox_smoothing="exponential"
```

**Best for:**

- Simpler implementation needs
- Fastest processing
- Best bbox stability

**Pros:**

- Simplest to implement and tune
- Best bbox size stability (1.02 px change)
- Lowest standard deviation (10.94 mph)
- Very close to Kalman performance

**Cons:**

- Slightly higher speed variation than Kalman

### Moving Average

```python
bbox_smoothing="moving_average"
bbox_smoothing_window=5  # Adjust window size
```

**Best for:**

- When you need predictable behavior
- Understanding exactly what's averaged

**Pros:**

- Simple and transparent
- No parameters except window size

**Cons:**

- Introduces lag
- Less adaptive than Kalman or EMA

### IOU-Weighted Average

```python
bbox_smoothing="iou_weighted"
bbox_smoothing_window=5
```

**Best for:**

- Vehicles with large size variations
- Handling occlusions

**Pros:**

- Adapts to size changes
- Good for complex scenarios

**Cons:**

- More computationally expensive
- Slightly lower performance than Kalman/EMA

### None (No Smoothing)

```python
bbox_smoothing="none"
```

**Use only for:**

- Debugging
- Raw data analysis
- Custom post-processing

## Comparison Test

Want to see the difference yourself? Run the comparison test:

```bash
# Test all methods on your video
python src/bbox_smoothing/test_bbox_smoothing.py happy1.mp4 out/test_37/detections.jsonl out/comparison

# Generate comparison plots
python src/bbox_smoothing/visualize_bbox_smoothing.py out/comparison
```

This generates:

- ✅ 5 annotated videos (one per method)
- ✅ Debug data with detailed metrics
- ✅ 4 comparison plots
- ✅ Summary statistics JSON

## Performance Metrics Explained

### Avg ΔSpeed (Frame-to-Frame Speed Change)

- **What**: Average absolute change in speed between consecutive frames
- **Lower is better** (indicates smoother speed estimates)
- **Baseline**: 0.81 mph
- **Best (Kalman)**: 0.63 mph (22% improvement)

### Speed Std (Speed Standard Deviation)

- **What**: Overall consistency of speed measurements
- **Lower is better** (indicates more consistent speeds)
- **Baseline**: 11.55 mph
- **Best (Exponential)**: 10.94 mph (5% improvement)

### Avg ΔBBox (BBox Size Change)

- **What**: Average change in bbox width+height per frame
- **Lower is better** (indicates more stable bbox)
- **Baseline**: 2.07 px
- **Best (Exponential)**: 1.02 px (51% improvement)

## Advanced Tuning

### Exponential Smoothing Alpha

Adjust responsiveness in `src/annotate_video.py`:

```python
self.bbox_ema_alpha = 0.3  # Default

# More smoothing (slower response):
self.bbox_ema_alpha = 0.2

# Less smoothing (faster response):
self.bbox_ema_alpha = 0.4
```

### Kalman Filter Parameters

Tune noise parameters in `src/annotate_video.py`:

```python
Q = 0.5  # Process noise (bbox variability)
R = 3.0  # Measurement noise (detection uncertainty)

# For more aggressive smoothing:
Q = 0.3
R = 5.0

# For more responsive tracking:
Q = 0.7
R = 2.0
```

## Integration with Existing Code

Bbox smoothing is **automatically enabled** in:

1. **main.py**: Default pipeline uses Kalman smoothing
2. **annotate_video.py**: VideoAnnotator class applies smoothing before speed calculation
3. **Speed calculation**: Uses smoothed bboxes for tracking points

No code changes needed in existing scripts!

## Troubleshooting

### Speed estimates still jumping?

1. Try exponential smoothing (sometimes better for specific videos):

   ```python
   bbox_smoothing="exponential"
   ```

2. Increase smoothing window for moving average methods:

   ```python
   bbox_smoothing="moving_average"
   bbox_smoothing_window=10  # Increased from 5
   ```

3. Adjust Kalman parameters (see Advanced Tuning above)

### Bboxes lagging behind vehicles?

- Reduce smoothing strength:

  ```python
  # For exponential:
  self.bbox_ema_alpha = 0.4  # More responsive

  # For Kalman:
  Q = 0.7  # Higher process noise
  R = 2.0  # Lower measurement noise
  ```

### Need raw bboxes for debugging?

- Disable smoothing temporarily:
  ```python
  bbox_smoothing="none"
  ```

## Files and Documentation

- **Implementation**: `src/annotate_video.py` (VideoAnnotator class)
- **Testing**: `test_bbox_smoothing.py` (comparison script)
- **Visualization**: `visualize_bbox_smoothing.py` (plot generation)
- **Full Analysis**: `BBOX_SMOOTHING_ANALYSIS.md` (detailed report)
- **This Guide**: `BBOX_SMOOTHING_QUICK_START.md`

## Examples

### Example 1: Process video with default settings

```bash
python main.py video.mp4 --homography homography-points.json
# Uses Kalman filter automatically
```

### Example 2: Try different smoothing method

```python
from src.annotate_video import annotate_video

annotate_video(
    video_path="video.mp4",
    detections_source="detections.jsonl",
    output_path="output.mp4",
    homography_file="homography-points.json",
    bbox_smoothing="exponential",  # Try exponential instead
)
```

### Example 3: Compare all methods

```bash
python src/bbox_smoothing/test_bbox_smoothing.py video.mp4 detections.jsonl output_dir homography.json
python src/bbox_smoothing/visualize_bbox_smoothing.py output_dir
```

## Results Summary

Based on testing with 98-frame video (19 vehicles, 1223 detections):

| Metric            | Baseline    | With Kalman | Improvement |
| ----------------- | ----------- | ----------- | ----------- |
| Speed Smoothness  | 0.81 mph    | 0.63 mph    | **22.2%** ✓ |
| BBox Stability    | 2.07 px     | 1.06 px     | **48.8%** ✓ |
| Speed Consistency | 11.55 mph σ | 10.99 mph σ | **4.8%** ✓  |

## Summary

🎯 **Recommendation**: Use the default (Kalman filter) for best results

🔧 **Alternative**: Use exponential for simpler implementation with nearly equal performance

📊 **Testing**: Run `test_bbox_smoothing.py` to compare methods on your data

✅ **Default Enabled**: Bbox smoothing is now active by default in the main pipeline

---

**Quick Start Complete!** Bbox smoothing is now protecting your speed estimates from detection jitter. 🚗💨
