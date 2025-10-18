#!/usr/bin/env python3
"""
Example script demonstrating vehicle speed annotation.

This script shows how to annotate a video with vehicle tracking IDs and speeds.
Speed is calculated using homography transformation to convert image coordinates
to real-world geographic coordinates.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from annotate_video import annotate_video


def main():
    # Define paths
    video_path = Path("happy1.mp4")
    detections_path = Path("out/test_37/detections-redone.jsonl")
    output_path = Path("out/test_37/detections-with-speed_annotated.mp4")
    homography_file = "homography-points.json"

    # Annotate video with speed calculation
    print("Annotating video with vehicle speeds...")
    print(f"Input video: {video_path}")
    print(f"Detections: {detections_path}")
    print(f"Homography file: {homography_file}")
    print(f"Output: {output_path}")
    print()

    result_path = annotate_video(
        video_path=video_path,
        detections_source=detections_path,
        output_path=output_path,
        trail_length=10,
        show_trails=True,
        show_labels=True,
        show_boxes=True,
        homography_file=homography_file,
    )

    print(f"\n✅ Done! Annotated video saved to: {result_path}")
    print("\nThe video now displays:")
    print("  - Vehicle ID")
    print("  - Speed in miles per hour (mph)")
    print("  - Tracking trails")
    print("  - Bounding boxes")


if __name__ == "__main__":
    main()
