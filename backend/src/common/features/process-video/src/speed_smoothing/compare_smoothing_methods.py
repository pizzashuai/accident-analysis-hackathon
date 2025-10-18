#!/usr/bin/env python3
"""
Compare different speed smoothing methods and generate comparison outputs.

This script processes the same detection data with different smoothing algorithms
and generates annotated videos and debug JSONL files for comparison.
"""

import argparse
from pathlib import Path
from src.annotate_video import VideoAnnotator


def main():
    parser = argparse.ArgumentParser(description="Compare speed smoothing methods")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--detections", type=str, required=True, help="Path to detections JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/smoothing_comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default="homography-points.json",
        help="Path to homography file",
    )
    parser.add_argument(
        "--tracking-point",
        type=str,
        default="center",
        choices=["center", "bottom_center"],
        help="Point to track on vehicle (center or bottom_center)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for moving average smoothing",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    detections_path = Path(args.detections)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    if not detections_path.exists():
        print(f"Error: Detections file not found: {detections_path}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Smoothing methods to test
    smoothing_methods = [
        ("none", "No smoothing (raw speeds)"),
        ("moving_average", f"Moving average (window={args.smoothing_window})"),
        ("exponential", "Exponential moving average (alpha=0.3)"),
        ("kalman", "Kalman filter (Q=0.1, R=2.0)"),
    ]

    print("=" * 80)
    print("Speed Smoothing Comparison Tool")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Detections: {detections_path.name}")
    print(f"Tracking point: {args.tracking_point}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    for method_name, method_desc in smoothing_methods:
        print(f"\n{'=' * 80}")
        print(f"Processing with: {method_desc}")
        print("=" * 80)

        # Create output paths
        output_video = output_dir / f"annotated_{method_name}.mp4"
        debug_jsonl = output_dir / f"debug_{method_name}.jsonl"

        # Create annotator with this smoothing method
        annotator = VideoAnnotator(
            trail_length=10,
            homography_file=args.homography,
            speed_smoothing=method_name,
            smoothing_window=args.smoothing_window,
            tracking_point=args.tracking_point,
            debug_speed=True,
            debug_jsonl_path=debug_jsonl,
        )

        # Annotate video
        try:
            annotator.annotate_video_from_jsonl(
                original_video_path=video_path,
                jsonl_path=detections_path,
                output_path=output_video,
                show_trails=True,
                show_labels=True,
                show_boxes=True,
            )
            print(f"✓ Video saved: {output_video}")
            print(f"✓ Debug data saved: {debug_jsonl}")
        except Exception as e:
            print(f"✗ Error processing {method_name}: {e}")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nTo compare results:")
    print(f"  1. Watch the videos in {output_dir}")
    print("  2. Analyze debug JSONL files to see raw vs smoothed speeds")
    print("  3. Use the analysis script: python analyze_speed_debug.py")


if __name__ == "__main__":
    main()
