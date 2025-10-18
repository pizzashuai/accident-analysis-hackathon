#!/usr/bin/env python3
"""
Analyze speed debug data from JSONL files.

This script reads debug JSONL files generated during video annotation
and provides insights into speed calculation issues.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def read_debug_jsonl(jsonl_path):
    """Read debug data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def analyze_track(track_data, track_id):
    """Analyze a single track's speed data."""
    if not track_data:
        return None

    # Extract speed values
    raw_speeds = [d["speed_calc"]["raw_speed_mph"] for d in track_data]
    smoothed_speeds = [d["speed_calc"]["smoothed_speed_mph"] for d in track_data]
    frames = [d["frame"] for d in track_data]

    # Calculate statistics
    raw_mean = statistics.mean(raw_speeds)
    raw_stdev = statistics.stdev(raw_speeds) if len(raw_speeds) > 1 else 0
    raw_min = min(raw_speeds)
    raw_max = max(raw_speeds)

    smoothed_mean = statistics.mean(smoothed_speeds)
    smoothed_stdev = (
        statistics.stdev(smoothed_speeds) if len(smoothed_speeds) > 1 else 0
    )
    smoothed_min = min(smoothed_speeds)
    smoothed_max = max(smoothed_speeds)

    # Find largest speed jumps (frame to frame)
    max_raw_jump = 0
    max_raw_jump_frames = None
    max_smoothed_jump = 0
    max_smoothed_jump_frames = None

    for i in range(1, len(track_data)):
        raw_jump = abs(raw_speeds[i] - raw_speeds[i - 1])
        if raw_jump > max_raw_jump:
            max_raw_jump = raw_jump
            max_raw_jump_frames = (frames[i - 1], frames[i])

        smoothed_jump = abs(smoothed_speeds[i] - smoothed_speeds[i - 1])
        if smoothed_jump > max_smoothed_jump:
            max_smoothed_jump = smoothed_jump
            max_smoothed_jump_frames = (frames[i - 1], frames[i])

    # Analyze bbox size variations
    bbox_widths = [d["bbox_size"]["width"] for d in track_data]
    bbox_heights = [d["bbox_size"]["height"] for d in track_data]
    width_range = max(bbox_widths) - min(bbox_widths)
    height_range = max(bbox_heights) - min(bbox_heights)

    return {
        "track_id": track_id,
        "num_frames": len(track_data),
        "frame_range": (frames[0], frames[-1]),
        "raw_speeds": {
            "mean": raw_mean,
            "stdev": raw_stdev,
            "min": raw_min,
            "max": raw_max,
            "range": raw_max - raw_min,
            "max_jump": max_raw_jump,
            "max_jump_frames": max_raw_jump_frames,
        },
        "smoothed_speeds": {
            "mean": smoothed_mean,
            "stdev": smoothed_stdev,
            "min": smoothed_min,
            "max": smoothed_max,
            "range": smoothed_max - smoothed_min,
            "max_jump": max_smoothed_jump,
            "max_jump_frames": max_smoothed_jump_frames,
        },
        "bbox_variation": {
            "width_range": width_range,
            "height_range": height_range,
        },
        "smoothing_effectiveness": {
            "stdev_reduction": ((raw_stdev - smoothed_stdev) / raw_stdev * 100)
            if raw_stdev > 0
            else 0,
            "jump_reduction": ((max_raw_jump - max_smoothed_jump) / max_raw_jump * 100)
            if max_raw_jump > 0
            else 0,
        },
    }


def print_track_analysis(analysis):
    """Print analysis results for a track."""
    print(f"\n{'=' * 80}")
    print(f"Track ID: {analysis['track_id']}")
    print(
        f"Frames: {analysis['frame_range'][0]} - {analysis['frame_range'][1]} ({analysis['num_frames']} frames)"
    )
    print("=" * 80)

    print("\n📊 Raw Speed Statistics:")
    print(f"  Mean:      {analysis['raw_speeds']['mean']:.2f} mph")
    print(f"  Std Dev:   {analysis['raw_speeds']['stdev']:.2f} mph")
    print(
        f"  Range:     {analysis['raw_speeds']['min']:.2f} - {analysis['raw_speeds']['max']:.2f} mph"
    )
    print(f"  Variation: {analysis['raw_speeds']['range']:.2f} mph")
    print(
        f"  Max Jump:  {analysis['raw_speeds']['max_jump']:.2f} mph (frames {analysis['raw_speeds']['max_jump_frames']})"
    )

    print("\n📈 Smoothed Speed Statistics:")
    print(f"  Mean:      {analysis['smoothed_speeds']['mean']:.2f} mph")
    print(f"  Std Dev:   {analysis['smoothed_speeds']['stdev']:.2f} mph")
    print(
        f"  Range:     {analysis['smoothed_speeds']['min']:.2f} - {analysis['smoothed_speeds']['max']:.2f} mph"
    )
    print(f"  Variation: {analysis['smoothed_speeds']['range']:.2f} mph")
    print(
        f"  Max Jump:  {analysis['smoothed_speeds']['max_jump']:.2f} mph (frames {analysis['smoothed_speeds']['max_jump_frames']})"
    )

    print("\n✨ Smoothing Effectiveness:")
    print(
        f"  Std Dev Reduction: {analysis['smoothing_effectiveness']['stdev_reduction']:.1f}%"
    )
    print(
        f"  Jump Reduction:    {analysis['smoothing_effectiveness']['jump_reduction']:.1f}%"
    )

    print("\n📦 Bounding Box Variation:")
    print(f"  Width Range:  {analysis['bbox_variation']['width_range']:.1f} pixels")
    print(f"  Height Range: {analysis['bbox_variation']['height_range']:.1f} pixels")


def print_detailed_frames(track_data, num_frames=10):
    """Print detailed frame-by-frame data."""
    print(f"\n{'=' * 80}")
    print(f"Detailed Frame Data (first {num_frames} frames)")
    print("=" * 80)

    for i, frame_data in enumerate(track_data[:num_frames]):
        print(f"\nFrame {frame_data['frame']} (time: {frame_data['time']:.3f}s):")
        print(f"  Bbox: {frame_data['bbox_xyxy']}")
        print(
            f"  Size: {frame_data['bbox_size']['width']:.1f} x {frame_data['bbox_size']['height']:.1f}"
        )
        print(
            f"  Track Point: ({frame_data['track_point_pixel']['x']:.1f}, {frame_data['track_point_pixel']['y']:.1f})"
        )
        print(
            f"  Normalized: ({frame_data['track_point_norm']['x']:.4f}, {frame_data['track_point_norm']['y']:.4f})"
        )

        if "speed_calc" in frame_data:
            calc = frame_data["speed_calc"]
            print("  Speed Calculation:")
            print(
                f"    Old Frame: {calc['old_frame']} -> New Frame: {calc['new_frame']}"
            )
            print(f"    Distance: {calc['distance_meters']:.2f} m")
            print(f"    Time Diff: {calc['time_diff']:.3f} s")
            print(f"    Raw Speed: {calc['raw_speed_mph']:.2f} mph")
            print(f"    Smoothed: {calc['smoothed_speed_mph']:.2f} mph")


def main():
    parser = argparse.ArgumentParser(description="Analyze speed debug data")
    parser.add_argument("debug_jsonl", type=str, help="Path to debug JSONL file")
    parser.add_argument(
        "--track-id", type=int, help="Analyze specific track ID (default: all tracks)"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed frame-by-frame data"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to show in detailed view (default: 10)",
    )

    args = parser.parse_args()

    debug_path = Path(args.debug_jsonl)
    if not debug_path.exists():
        print(f"Error: Debug file not found: {debug_path}")
        return

    print("=" * 80)
    print("Speed Debug Data Analysis")
    print("=" * 80)
    print(f"File: {debug_path}")
    print()

    # Read debug data
    debug_data = read_debug_jsonl(debug_path)
    print(f"Total records: {len(debug_data)}")

    # Group by track ID
    tracks = defaultdict(list)
    for record in debug_data:
        tracks[record["track_id"]].append(record)

    print(f"Unique tracks: {len(tracks)}")

    # Analyze tracks
    if args.track_id is not None:
        # Analyze specific track
        if args.track_id not in tracks:
            print(f"Error: Track ID {args.track_id} not found")
            return

        track_data = tracks[args.track_id]
        analysis = analyze_track(track_data, args.track_id)
        print_track_analysis(analysis)

        if args.detailed:
            print_detailed_frames(track_data, args.num_frames)

    else:
        # Analyze all tracks
        all_analyses = []
        for track_id in sorted(tracks.keys()):
            track_data = tracks[track_id]
            analysis = analyze_track(track_data, track_id)
            all_analyses.append(analysis)

        # Print summary
        print(f"\n{'=' * 80}")
        print("Summary of All Tracks")
        print("=" * 80)
        print(
            f"{'Track ID':<10} {'Frames':<8} {'Raw Mean':<12} {'Raw Jump':<12} {'Smoothed Mean':<15} {'Smoothed Jump':<15}"
        )
        print("-" * 80)

        for analysis in all_analyses:
            print(
                f"{analysis['track_id']:<10} "
                f"{analysis['num_frames']:<8} "
                f"{analysis['raw_speeds']['mean']:>6.2f} mph   "
                f"{analysis['raw_speeds']['max_jump']:>6.2f} mph   "
                f"{analysis['smoothed_speeds']['mean']:>6.2f} mph      "
                f"{analysis['smoothed_speeds']['max_jump']:>6.2f} mph"
            )

        # Find most problematic tracks
        print(f"\n{'=' * 80}")
        print("Most Problematic Tracks (highest raw speed jumps)")
        print("=" * 80)

        sorted_by_jump = sorted(
            all_analyses, key=lambda x: x["raw_speeds"]["max_jump"], reverse=True
        )
        for analysis in sorted_by_jump[:5]:
            print(f"\nTrack {analysis['track_id']}:")
            print(
                f"  Raw speed jump: {analysis['raw_speeds']['max_jump']:.2f} mph at frames {analysis['raw_speeds']['max_jump_frames']}"
            )
            print(
                f"  After smoothing: {analysis['smoothed_speeds']['max_jump']:.2f} mph"
            )
            print(
                f"  Bbox width variation: {analysis['bbox_variation']['width_range']:.1f} pixels"
            )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
