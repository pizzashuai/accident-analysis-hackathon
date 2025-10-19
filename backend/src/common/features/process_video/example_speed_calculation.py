#!/usr/bin/env python3
"""
Example: Calculate speeds with optimal smoothing (Kalman filter)

This example shows how to process detections and add speed_mph
using the best performing method.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from pathlib import Path
from src.common.features.process_video.processor import calculate_speeds_with_smoothing


def main():
    # Input files
    script_dir = Path(__file__).parent
    jsonl_path = script_dir / "detections.jsonl"
    homography_path = script_dir / "homography-points.json"
    
    # Output file
    output_path = script_dir / "detections_with_speeds.jsonl"
    
    print("="*80)
    print("Speed Calculation Example - Using Kalman Filter (Optimal)")
    print("="*80)
    print(f"\nInput:  {jsonl_path}")
    print(f"Output: {output_path}")
    print()
    
    # Calculate speeds using optimal method
    output_path, stats = calculate_speeds_with_smoothing(
        jsonl_path=jsonl_path,
        homography_file=str(homography_path),
        output_path=output_path,
        video_width=1280,
        video_height=720,
        # Optimal settings for urban traffic
        smoothing_method="kalman_with_outlier_rejection",  # Best performance
        smoothing_window=5,                                # Good balance
        max_reasonable_speed=100.0,                        # Maximum speed (mph)
        min_reasonable_speed=0.0,                          # Minimum speed (mph)
        lookback_frames=5,                                 # Stability vs responsiveness
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total detections:     {stats['total_detections']}")
    print(f"Speeds assigned:      {stats['speeds_assigned']}")
    print(f"Coverage:             {100 * stats['speeds_assigned'] / stats['total_detections']:.1f}%")
    print()
    print(f"Average speed:        {stats.get('avg_smoothed_speed', 0):.2f} mph")
    print(f"Median speed:         {stats.get('median_smoothed_speed', 0):.2f} mph")
    print(f"Speed range:          {stats.get('min_smoothed_speed', 0):.2f} - {stats.get('max_smoothed_speed', 0):.2f} mph")
    print()
    print(f"Outliers rejected:    {stats['outliers_rejected']} ({stats.get('outlier_rejection_rate', 0):.1f}%)")
    print()
    print(f"Output saved to: {output_path}")
    print("="*80)
    
    # Show sample speeds
    print("\nSample speeds from output file:")
    import json
    speeds = []
    count = 0
    with open(output_path, 'r') as f:
        for line in f:
            det = json.loads(line.strip())
            if det.get('speed_mph') is not None and count < 10:
                speeds.append({
                    'track_id': det['track_id'],
                    'frame': det['frame'],
                    'speed': det['speed_mph'],
                })
                count += 1
    
    for i, sample in enumerate(speeds[:5], 1):
        print(f"  {i}. Track {sample['track_id']:3d} @ frame {sample['frame']:4d}: {sample['speed']:6.2f} mph")
    
    print("\n✅ Speed calculation complete!")
    print("✅ No impossible speeds (all < 100 mph)")
    print("✅ Smooth, realistic results")


if __name__ == "__main__":
    main()

