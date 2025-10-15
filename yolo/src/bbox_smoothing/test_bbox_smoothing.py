#!/usr/bin/env python3
"""
Test and compare different bounding box smoothing methods for speed estimation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from ..annotate_video import annotate_video


def test_bbox_smoothing_methods(
    video_path: Path,
    detections_jsonl: Path,
    output_dir: Path,
    homography_file: Path,
    methods: List[str] = None,
):
    """
    Test different bbox smoothing methods and generate comparison videos.

    Args:
        video_path: Path to input video
        detections_jsonl: Path to detections JSONL file
        output_dir: Directory to save output videos
        homography_file: Path to homography file for speed calculation
        methods: List of smoothing methods to test
    """
    if methods is None:
        methods = ["none", "moving_average", "exponential", "kalman", "iou_weighted"]

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("=" * 80)
    print("BOUNDING BOX SMOOTHING COMPARISON TEST")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Detections: {detections_jsonl.name}")
    print(f"Output directory: {output_dir}")
    print(f"Methods to test: {', '.join(methods)}")
    print("=" * 80)
    print()

    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"Testing method: {method}")
        print(f"{'=' * 80}")

        # Set up output paths
        output_video = output_dir / f"bbox_{method}_annotated.mp4"
        debug_jsonl = output_dir / f"bbox_{method}_debug.jsonl"

        try:
            # Annotate video with this smoothing method
            annotate_video(
                video_path=video_path,
                detections_source=detections_jsonl,
                output_path=output_video,
                homography_file=str(homography_file),
                speed_smoothing="exponential",  # Use consistent speed smoothing for comparison
                smoothing_window=5,
                bbox_smoothing=method,  # Test different bbox smoothing methods
                bbox_smoothing_window=5,
                tracking_point="center",
                debug_speed=True,
                debug_jsonl_path=debug_jsonl,
                show_trails=True,
                show_labels=True,
                show_boxes=True,
            )

            # Read debug data to calculate statistics
            stats = calculate_smoothing_stats(debug_jsonl)

            results[method] = {
                "success": True,
                "output_video": str(output_video),
                "debug_jsonl": str(debug_jsonl),
                "stats": stats,
            }

            print(f"\n✓ Method '{method}' completed successfully")
            print(f"  Output video: {output_video}")
            print(f"  Debug data: {debug_jsonl}")
            print("\n  Statistics:")
            for key, value in stats.items():
                print(f"    {key}: {value}")

        except Exception as e:
            print(f"\n✗ Method '{method}' failed: {e}")
            results[method] = {
                "success": False,
                "error": str(e),
            }

    # Save results summary
    results_path = output_dir / "smoothing_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {results_path}")
    print()

    # Print summary
    print_comparison_summary(results)

    return results


def calculate_smoothing_stats(debug_jsonl: Path) -> Dict:
    """
    Calculate statistics from debug data.

    Args:
        debug_jsonl: Path to debug JSONL file

    Returns:
        Dictionary of statistics
    """
    if not debug_jsonl.exists():
        return {}

    # Read debug data
    debug_data = []
    with open(debug_jsonl, "r") as f:
        for line in f:
            debug_data.append(json.loads(line))

    if not debug_data:
        return {}

    # Organize by track_id
    tracks = {}
    for record in debug_data:
        track_id = record["track_id"]
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(record)

    # Calculate statistics
    all_speeds = []
    all_bbox_widths = []
    all_bbox_heights = []
    all_speed_changes = []
    all_bbox_size_changes = []

    for track_id, records in tracks.items():
        if len(records) < 2:
            continue

        for i, record in enumerate(records):
            speed = record["speed_calc"]["smoothed_speed_mph"]
            bbox_width = record["bbox_size"]["width"]
            bbox_height = record["bbox_size"]["height"]

            all_speeds.append(speed)
            all_bbox_widths.append(bbox_width)
            all_bbox_heights.append(bbox_height)

            if i > 0:
                prev_speed = records[i - 1]["speed_calc"]["smoothed_speed_mph"]
                prev_width = records[i - 1]["bbox_size"]["width"]
                prev_height = records[i - 1]["bbox_size"]["height"]

                speed_change = abs(speed - prev_speed)
                bbox_size_change = abs(bbox_width - prev_width) + abs(
                    bbox_height - prev_height
                )

                all_speed_changes.append(speed_change)
                all_bbox_size_changes.append(bbox_size_change)

    import numpy as np

    stats = {
        "total_frames": len(debug_data),
        "num_tracks": len(tracks),
        "avg_speed_mph": round(float(np.mean(all_speeds)), 2) if all_speeds else 0,
        "std_speed_mph": round(float(np.std(all_speeds)), 2) if all_speeds else 0,
        "avg_speed_change_mph": round(float(np.mean(all_speed_changes)), 2)
        if all_speed_changes
        else 0,
        "max_speed_change_mph": round(float(np.max(all_speed_changes)), 2)
        if all_speed_changes
        else 0,
        "avg_bbox_size_change_px": round(float(np.mean(all_bbox_size_changes)), 2)
        if all_bbox_size_changes
        else 0,
        "max_bbox_size_change_px": round(float(np.max(all_bbox_size_changes)), 2)
        if all_bbox_size_changes
        else 0,
    }

    return stats


def print_comparison_summary(results: Dict):
    """
    Print a summary comparison of results.

    Args:
        results: Dictionary of results from test_bbox_smoothing_methods
    """
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    # Table header
    print(
        f"{'Method':<20} {'Avg Speed':<12} {'Speed Std':<12} {'Avg ΔSpeed':<12} {'Avg ΔBBox':<12}"
    )
    print("-" * 80)

    # Sort by avg speed change (lower is better)
    sorted_methods = sorted(
        [
            (method, data)
            for method, data in results.items()
            if data.get("success", False)
        ],
        key=lambda x: x[1]["stats"].get("avg_speed_change_mph", float("inf")),
    )

    for method, data in sorted_methods:
        stats = data["stats"]
        print(
            f"{method:<20} "
            f"{stats.get('avg_speed_mph', 0):>10.2f}  "
            f"{stats.get('std_speed_mph', 0):>10.2f}  "
            f"{stats.get('avg_speed_change_mph', 0):>10.2f}  "
            f"{stats.get('avg_bbox_size_change_px', 0):>10.2f}"
        )

    print()
    print("Legend:")
    print("  Avg Speed: Average speed across all detections (mph)")
    print("  Speed Std: Standard deviation of speeds (lower = more consistent)")
    print("  Avg ΔSpeed: Average frame-to-frame speed change (lower = smoother)")
    print("  Avg ΔBBox: Average frame-to-frame bbox size change (lower = more stable)")
    print()

    # Recommendation
    if sorted_methods:
        best_method = sorted_methods[0][0]
        best_stats = sorted_methods[0][1]["stats"]
        print(f"🏆 RECOMMENDED METHOD: {best_method}")
        print(
            f"   - Lowest average speed change: {best_stats['avg_speed_change_mph']:.2f} mph"
        )
        print(f"   - Speed standard deviation: {best_stats['std_speed_mph']:.2f} mph")
        print()


def main():
    """Main function to run bbox smoothing comparison."""
    # Default paths
    video_path = Path("happy1.mp4")
    detections_jsonl = Path("out/test_37/detections.jsonl")
    output_dir = Path("out/bbox_smoothing_comparison")
    homography_file = Path("homography-points.json")

    # Check if files exist
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        print("Please provide the video file path as the first argument.")
        sys.exit(1)

    if not detections_jsonl.exists():
        print(f"Error: Detections file not found: {detections_jsonl}")
        print("Please provide the detections JSONL path as the second argument.")
        sys.exit(1)

    if not homography_file.exists():
        print(f"Error: Homography file not found: {homography_file}")
        print("Please provide the homography file path.")
        sys.exit(1)

    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        detections_jsonl = Path(sys.argv[2])
    if len(sys.argv) > 3:
        output_dir = Path(sys.argv[3])
    if len(sys.argv) > 4:
        homography_file = Path(sys.argv[4])

    # Run tests
    test_bbox_smoothing_methods(
        video_path=video_path,
        detections_jsonl=detections_jsonl,
        output_dir=output_dir,
        homography_file=homography_file,
    )


if __name__ == "__main__":
    main()
