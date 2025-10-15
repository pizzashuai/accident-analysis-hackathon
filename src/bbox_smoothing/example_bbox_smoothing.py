#!/usr/bin/env python3
"""
Example script demonstrating different bbox smoothing methods.
Shows how to use bbox smoothing in your own code.
"""

from pathlib import Path
from ..annotate_video import annotate_video


# Example 1: Use default Kalman filter (recommended)
def example_kalman():
    """Example using Kalman filter (best for speed estimation)."""
    print("Example 1: Kalman Filter (Recommended)")
    print("-" * 60)

    annotate_video(
        video_path="happy1.mp4",
        detections_source="out/test_37/detections.jsonl",
        output_path="out/example_kalman.mp4",
        homography_file="homography-points.json",
        bbox_smoothing="kalman",  # Best for speed estimation
        bbox_smoothing_window=5,
        debug_speed=True,
        debug_jsonl_path=Path("out/example_kalman_debug.jsonl"),
    )

    print("✓ Kalman filter example complete")
    print("  Output: out/example_kalman.mp4")
    print("  Debug: out/example_kalman_debug.jsonl")
    print()


# Example 2: Use exponential smoothing (simpler alternative)
def example_exponential():
    """Example using Exponential Moving Average."""
    print("Example 2: Exponential Moving Average")
    print("-" * 60)

    annotate_video(
        video_path="happy1.mp4",
        detections_source="out/test_37/detections.jsonl",
        output_path="out/example_exponential.mp4",
        homography_file="homography-points.json",
        bbox_smoothing="exponential",  # Simpler, nearly as good
        bbox_smoothing_window=5,
        debug_speed=True,
        debug_jsonl_path=Path("out/example_exponential_debug.jsonl"),
    )

    print("✓ Exponential smoothing example complete")
    print("  Output: out/example_exponential.mp4")
    print("  Debug: out/example_exponential_debug.jsonl")
    print()


# Example 3: No smoothing (for comparison)
def example_no_smoothing():
    """Example with no bbox smoothing (baseline)."""
    print("Example 3: No Smoothing (Baseline)")
    print("-" * 60)

    annotate_video(
        video_path="happy1.mp4",
        detections_source="out/test_37/detections.jsonl",
        output_path="out/example_no_smoothing.mp4",
        homography_file="homography-points.json",
        bbox_smoothing="none",  # No smoothing
        debug_speed=True,
        debug_jsonl_path=Path("out/example_no_smoothing_debug.jsonl"),
    )

    print("✓ No smoothing example complete")
    print("  Output: out/example_no_smoothing.mp4")
    print("  Debug: out/example_no_smoothing_debug.jsonl")
    print()


# Example 4: Using VideoAnnotator class directly
def example_custom():
    """Example using VideoAnnotator class directly for more control."""
    print("Example 4: Custom Configuration")
    print("-" * 60)

    from ..annotate_video import VideoAnnotator

    # Create annotator with custom settings
    annotator = VideoAnnotator(
        trail_length=15,  # Longer trails
        homography_file="homography-points.json",
        speed_smoothing="exponential",  # Speed smoothing
        smoothing_window=7,  # Larger speed smoothing window
        bbox_smoothing="kalman",  # Bbox smoothing
        bbox_smoothing_window=5,
        tracking_point="bottom_center",  # Track bottom of vehicle
        debug_speed=True,
        debug_jsonl_path=Path("out/example_custom_debug.jsonl"),
    )

    # Annotate video
    annotator.annotate_video_from_jsonl(
        original_video_path="happy1.mp4",
        jsonl_path="out/test_37/detections.jsonl",
        output_path="out/example_custom.mp4",
        show_trails=True,
        show_labels=True,
        show_boxes=True,
    )

    print("✓ Custom configuration example complete")
    print("  Output: out/example_custom.mp4")
    print("  Debug: out/example_custom_debug.jsonl")
    print()


def compare_debug_results():
    """Compare debug results from different methods."""
    print("=" * 80)
    print("COMPARISON OF DEBUG RESULTS")
    print("=" * 80)
    print()

    import json

    methods = [
        ("No Smoothing", "out/example_no_smoothing_debug.jsonl"),
        ("Exponential", "out/example_exponential_debug.jsonl"),
        ("Kalman", "out/example_kalman_debug.jsonl"),
    ]

    for method_name, debug_file in methods:
        if not Path(debug_file).exists():
            print(f"⚠ {method_name}: Debug file not found")
            continue

        # Read debug data
        speeds = []
        speed_changes = []
        bbox_changes = []

        with open(debug_file, "r") as f:
            prev_record = None
            for line in f:
                record = json.loads(line)
                speed = record["speed_calc"]["smoothed_speed_mph"]
                speeds.append(speed)

                if prev_record and prev_record["track_id"] == record["track_id"]:
                    prev_speed = prev_record["speed_calc"]["smoothed_speed_mph"]
                    speed_changes.append(abs(speed - prev_speed))

                    prev_width = prev_record["bbox_size"]["width"]
                    prev_height = prev_record["bbox_size"]["height"]
                    curr_width = record["bbox_size"]["width"]
                    curr_height = record["bbox_size"]["height"]
                    bbox_changes.append(
                        abs(curr_width - prev_width) + abs(curr_height - prev_height)
                    )

                prev_record = record

        import numpy as np

        print(f"{method_name}:")
        print(f"  Avg Speed: {np.mean(speeds):.2f} mph")
        print(f"  Speed Std: {np.std(speeds):.2f} mph")
        print(f"  Avg Speed Change: {np.mean(speed_changes):.2f} mph")
        print(f"  Avg BBox Change: {np.mean(bbox_changes):.2f} px")
        print()

    print("=" * 80)
    print()


def main():
    """Run all examples."""
    print("=" * 80)
    print("BOUNDING BOX SMOOTHING EXAMPLES")
    print("=" * 80)
    print()
    print("This script demonstrates different bbox smoothing methods.")
    print("Each example generates an annotated video and debug data.")
    print()

    # Check if required files exist
    if not Path("happy1.mp4").exists():
        print("Error: happy1.mp4 not found")
        print("Please ensure the video file is in the current directory.")
        return

    if not Path("out/test_37/detections.jsonl").exists():
        print("Error: out/test_37/detections.jsonl not found")
        print("Please run main.py first to generate detections.")
        return

    if not Path("homography-points.json").exists():
        print("Warning: homography-points.json not found")
        print("Speed calculation will be disabled.")

    # Run examples
    try:
        example_kalman()
        example_exponential()
        example_no_smoothing()
        example_custom()

        # Compare results
        compare_debug_results()

        print("=" * 80)
        print("ALL EXAMPLES COMPLETE")
        print("=" * 80)
        print()
        print("Generated files:")
        print("  - out/example_kalman.mp4")
        print("  - out/example_exponential.mp4")
        print("  - out/example_no_smoothing.mp4")
        print("  - out/example_custom.mp4")
        print()
        print("Compare the videos to see the difference!")
        print()

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
