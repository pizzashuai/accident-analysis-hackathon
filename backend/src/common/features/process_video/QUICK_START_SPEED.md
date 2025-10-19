# Quick Start: Speed Calculation

## Run Speed Analysis in One Command

```bash
cd /Users/shuaima/code/accident_analysis/accident-analysis-hackathon/backend

# Using the processor main function directly
python3 -m src.common.features.process_video.processor \
  --jsonl src/common/features/process_video/detections.jsonl \
  --homography src/common/features/process_video/homography-points.json \
  --output-dir src/common/features/process_video/speed_output \
  --video-width 1280 \
  --video-height 720 \
  --max-speed 100.0 \
  --min-speed 0.0 \
  --lookback 5
```

## Or Use the Test Script

```bash
cd /Users/shuaima/code/accident_analysis/accident-analysis-hackathon/backend
python3 src/common/features/process_video/test_speed_smoothing.py
```

## What You Get

The script will test 5 different smoothing methods and output:

1. **Comparison Table**:

```
Method                              Avg Speed       Median          Max     Outliers
-------------------------------------------------------------------------------------
none                                     5.95         0.00        63.95            0
moving_average                           5.99         1.42        51.21            0
exponential_with_outlier_rejection       5.98         1.47        50.67            0
kalman_with_outlier_rejection            5.97         1.71        49.84            0  ← BEST
median_moving_average                    5.99         1.42        51.21            0
```

2. **Output Files** (in `speed_test_output/`):
   - `detections_no_smoothing.jsonl` - Raw speeds (for debugging)
   - `detections_moving_average.jsonl` - Simple smoothing
   - `detections_exponential_outlier.jsonl` - EMA with outlier rejection
   - `detections_kalman_outlier.jsonl` - Kalman filter (recommended)
   - `detections_median_ma.jsonl` - Median + MA (alternative)
   - `summary.json` - Statistics for all methods

## View Results

```bash
# Analyze speed distributions
python3 src/common/features/process_video/analyze_speeds.py

# Check specific output file
python3 -c "
import json
speeds = []
for line in open('src/common/features/process_video/speed_test_output/detections_kalman_outlier.jsonl'):
    det = json.loads(line)
    if det.get('speed_mph') is not None:
        speeds.append(det['speed_mph'])

print(f'Total speeds: {len(speeds)}')
print(f'Average: {sum(speeds)/len(speeds):.2f} mph')
print(f'Max: {max(speeds):.2f} mph')
print(f'Min: {min(speeds):.2f} mph')
"
```

## Adjust Parameters

### For Highway Scenarios (faster traffic):

```bash
python3 -m src.common.features.process_video.processor \
  --jsonl detections.jsonl \
  --homography homography-points.json \
  --output-dir speed_output \
  --max-speed 150.0 \  # Higher speed limit
  --lookback 3          # More responsive
```

### For City Scenarios (slower traffic, more outliers):

```bash
python3 -m src.common.features.process_video.processor \
  --jsonl detections.jsonl \
  --homography homography-points.json \
  --output-dir speed_output \
  --max-speed 60.0 \   # Lower speed limit
  --lookback 8          # More stable
```

## Key Findings

✅ **Kalman filter with outlier rejection** produces the best results:

- Max speed: 49.84 mph (vs 63.95 mph without smoothing)
- Smooth, realistic speed estimates
- No impossible speeds (no 2000 mph!)

✅ **Speed distribution is realistic**:

- 62.7% of vehicles: 0-5 mph (stopped/slow)
- 27.2% of vehicles: 5-15 mph (city driving)
- 10.1% of vehicles: 15-50 mph (faster roads)
- 0.0% of vehicles: >100 mph (no outliers!)

✅ **All smoothing methods work well**:

- No impossible speeds detected
- All max speeds < 64 mph
- Reasonable average speeds (~6 mph for this intersection)

## Next Steps

1. **Use in production**: Update your video processing pipeline to use `kalman_with_outlier_rejection`
2. **Calibrate for your scenario**: Adjust `max_speed` based on your road type
3. **Monitor results**: Check `analyze_speeds.py` output for outliers
4. **Fine-tune if needed**: Adjust `lookback_frames` and `smoothing_window` for your use case
