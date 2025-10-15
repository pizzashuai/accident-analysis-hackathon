#!/usr/bin/env python3
"""
Quick test of speed smoothing on existing detections.
"""

from pathlib import Path
from ..annotate_video import annotate_video

# Paths
video_path = Path("happy1.mp4")
detections_path = Path("out/test_37/detections.jsonl")
homography_file = "homography-points.json"

# Test directory
test_dir = Path("out/test_37/speed_debug")
test_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Testing Speed Smoothing")
print("=" * 80)
print(f"Video: {video_path}")
print(f"Detections: {detections_path}")
print(f"Output: {test_dir}")
print()

# Test 1: No smoothing (raw speeds) with debug
print("\n1. Testing: No smoothing (raw speeds)")
print("-" * 80)
annotate_video(
    video_path=video_path,
    detections_source=detections_path,
    output_path=test_dir / "test_raw.mp4",
    homography_file=homography_file,
    speed_smoothing="none",
    tracking_point="center",
    debug_speed=True,
    debug_jsonl_path=test_dir / "debug_raw.jsonl",
)

# Test 2: Moving average with center
print("\n2. Testing: Moving average (window=5) with center tracking")
print("-" * 80)
annotate_video(
    video_path=video_path,
    detections_source=detections_path,
    output_path=test_dir / "test_ma_center.mp4",
    homography_file=homography_file,
    speed_smoothing="moving_average",
    smoothing_window=5,
    tracking_point="center",
    debug_speed=True,
    debug_jsonl_path=test_dir / "debug_ma_center.jsonl",
)

# Test 3: Moving average with bottom_center
print("\n3. Testing: Moving average (window=5) with bottom_center tracking")
print("-" * 80)
annotate_video(
    video_path=video_path,
    detections_source=detections_path,
    output_path=test_dir / "test_ma_bottom.mp4",
    homography_file=homography_file,
    speed_smoothing="moving_average",
    smoothing_window=5,
    tracking_point="bottom_center",
    debug_speed=True,
    debug_jsonl_path=test_dir / "debug_ma_bottom.jsonl",
)

# Test 4: Exponential smoothing
print("\n4. Testing: Exponential moving average")
print("-" * 80)
annotate_video(
    video_path=video_path,
    detections_source=detections_path,
    output_path=test_dir / "test_ema.mp4",
    homography_file=homography_file,
    speed_smoothing="exponential",
    tracking_point="bottom_center",
    debug_speed=True,
    debug_jsonl_path=test_dir / "debug_ema.jsonl",
)

# Test 5: Kalman filter
print("\n5. Testing: Kalman filter")
print("-" * 80)
annotate_video(
    video_path=video_path,
    detections_source=detections_path,
    output_path=test_dir / "test_kalman.mp4",
    homography_file=homography_file,
    speed_smoothing="kalman",
    tracking_point="bottom_center",
    debug_speed=True,
    debug_jsonl_path=test_dir / "debug_kalman.jsonl",
)

print("\n" + "=" * 80)
print("Testing Complete!")
print("=" * 80)
print(f"\nOutputs saved to: {test_dir}")
print("\nNext steps:")
print("1. Watch the videos to see which smoothing looks best")
print("2. Analyze debug files:")
print(
    f"   python src/speed_smoothing/analyze_speed_debug.py {test_dir}/debug_ma_center.jsonl"
)
print("3. Compare raw vs smoothed:")
print(
    f"   python src/speed_smoothing/analyze_speed_debug.py {test_dir}/debug_ma_center.jsonl --track-id 1 --detailed"
)
print()
