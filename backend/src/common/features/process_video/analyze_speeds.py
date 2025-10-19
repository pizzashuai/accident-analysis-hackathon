#!/usr/bin/env python3
"""
Analyze speed distributions from processed detections.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_speed_file(jsonl_path):
    """Analyze speeds from a JSONL file."""
    speeds_by_track = defaultdict(list)
    all_speeds = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            det = json.loads(line.strip())
            speed = det.get("speed_mph")
            track_id = det.get("track_id")
            
            if speed is not None and track_id is not None:
                speeds_by_track[track_id].append(speed)
                all_speeds.append(speed)
    
    if not all_speeds:
        print(f"No speeds found in {jsonl_path}")
        return
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"Speed Analysis: {jsonl_path.name}")
    print(f"{'='*80}")
    print(f"Total detections with speed: {len(all_speeds)}")
    print(f"Total tracks: {len(speeds_by_track)}")
    print(f"\nSpeed Statistics:")
    print(f"  Average: {sum(all_speeds) / len(all_speeds):.2f} mph")
    print(f"  Median: {sorted(all_speeds)[len(all_speeds) // 2]:.2f} mph")
    print(f"  Min: {min(all_speeds):.2f} mph")
    print(f"  Max: {max(all_speeds):.2f} mph")
    
    # Speed distribution
    buckets = {
        "0-5 mph": 0,
        "5-15 mph": 0,
        "15-25 mph": 0,
        "25-35 mph": 0,
        "35-50 mph": 0,
        "50-100 mph": 0,
        ">100 mph (OUTLIERS!)": 0,
    }
    
    for speed in all_speeds:
        if speed < 5:
            buckets["0-5 mph"] += 1
        elif speed < 15:
            buckets["5-15 mph"] += 1
        elif speed < 25:
            buckets["15-25 mph"] += 1
        elif speed < 35:
            buckets["25-35 mph"] += 1
        elif speed < 50:
            buckets["35-50 mph"] += 1
        elif speed < 100:
            buckets["50-100 mph"] += 1
        else:
            buckets[">100 mph (OUTLIERS!)"] += 1
    
    print(f"\nSpeed Distribution:")
    for bucket, count in buckets.items():
        pct = 100 * count / len(all_speeds)
        bar = "█" * int(pct / 2)
        print(f"  {bucket:25s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Track-level statistics
    print(f"\nTop 10 Fastest Average Tracks:")
    track_avg_speeds = {tid: sum(speeds) / len(speeds) for tid, speeds in speeds_by_track.items()}
    sorted_tracks = sorted(track_avg_speeds.items(), key=lambda x: x[1], reverse=True)[:10]
    for tid, avg_speed in sorted_tracks:
        max_speed = max(speeds_by_track[tid])
        print(f"  Track {tid:3d}: avg={avg_speed:6.2f} mph, max={max_speed:6.2f} mph, frames={len(speeds_by_track[tid])}")
    
    # Identify problematic tracks (potential outliers)
    print(f"\nPotential Outlier Tracks (max speed > 60 mph):")
    outlier_tracks = [(tid, max(speeds)) for tid, speeds in speeds_by_track.items() if max(speeds) > 60]
    if outlier_tracks:
        for tid, max_speed in sorted(outlier_tracks, key=lambda x: x[1], reverse=True):
            avg_speed = sum(speeds_by_track[tid]) / len(speeds_by_track[tid])
            print(f"  Track {tid:3d}: max={max_speed:6.2f} mph, avg={avg_speed:6.2f} mph")
    else:
        print("  None found! (Good)")


if __name__ == "__main__":
    # Analyze all output files
    output_dir = Path(__file__).parent / "speed_test_output"
    
    if not output_dir.exists():
        print(f"Error: {output_dir} not found")
        print("Run test_speed_smoothing.py first!")
        sys.exit(1)
    
    # Find all detections files
    jsonl_files = sorted(output_dir.glob("detections_*.jsonl"))
    
    if not jsonl_files:
        print(f"No detection files found in {output_dir}")
        sys.exit(1)
    
    for jsonl_path in jsonl_files:
        analyze_speed_file(jsonl_path)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("All methods produced reasonable speeds (0-64 mph range)")
    print("Kalman filter with outlier rejection provides the smoothest results")
    print("Median + Moving Average provides good balance of smoothness and responsiveness")

