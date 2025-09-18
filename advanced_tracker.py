#!/usr/bin/env python3
"""
Advanced multi-object tracking using motpy library with Kalman filters.

This replaces the basic ByteTracker with a more sophisticated tracking system
that uses:
1. Kalman filters for motion prediction
2. Hungarian algorithm for optimal assignment
3. Better handling of occlusions and missed detections
4. More consistent ID management

The motpy library provides state-of-the-art tracking algorithms that should
significantly reduce ID switching compared to the basic ByteTracker.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Import motpy for advanced tracking
from motpy import MultiObjectTracker, Detection


def load_detections(detections_path: Path) -> List[Dict[str, Any]]:
    """Load detections from JSONL file."""
    detections = []
    with detections_path.open("r") as f:
        for line in f:
            if line.strip():
                detections.append(json.loads(line))
    return detections


def group_detections_by_frame(
    detections: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group detections by frame number."""
    frame_detections = defaultdict(list)
    for det in detections:
        frame_detections[det["frame"]].append(det)
    return frame_detections


def convert_to_motpy_format(detection: Dict[str, Any]) -> Detection:
    """
    Convert detection to motpy Detection object.
    """
    bbox = detection["bbox"]
    score = detection["score"]
    return Detection(box=bbox, score=score)


def convert_from_motpy_format(
    track, frame_idx: int, timestamp: float, id_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Convert motpy track back to our format.

    motpy track object has:
    - track.id: unique track ID (UUID string)
    - track.box: current bounding box [x1, y1, x2, y2]
    - track.score: confidence score
    """
    # Convert UUID to integer ID
    track_id_str = str(track.id)
    if track_id_str not in id_mapping:
        id_mapping[track_id_str] = len(id_mapping) + 1

    return {
        "obj_id": id_mapping[track_id_str],
        "bbox": [float(x) for x in track.box],
        "score": float(track.score) if hasattr(track, "score") else 1.0,
        "cls": 2,  # Assume vehicle class (this would need to be tracked separately for multi-class)
        "frame": frame_idx,
        "t": timestamp,
    }


default_model_spec = {"order_pos": 1, "dim_pos": 2, "order_size": 0, "dim_size": 2}


def track_with_motpy(
    detections_path: Path,
    output_path: Path,
    dt: float = 1 / 30.0,  # Time step (1/fps)
    model_spec: Dict = default_model_spec,
    max_staleness: int = 12,  # How long to keep lost tracks
):
    """
    Run advanced tracking using motpy library.

    Parameters:
    - dt: Time step between frames
    - model_spec: Motion model specification dict
    - max_staleness: Frames to keep lost tracks before deletion
    """

    print(f"Loading detections from {detections_path}...")
    detections = load_detections(detections_path)
    frame_detections = group_detections_by_frame(detections)

    print(f"Loaded {len(detections)} detections across {len(frame_detections)} frames")

    # Initialize motpy tracker with parameters for vehicle tracking
    tracker_kwargs = {
        "max_staleness": max_staleness,
    }

    tracker = MultiObjectTracker(
        dt=dt,
        model_spec=model_spec,
        tracker_kwargs=tracker_kwargs,
    )

    print("Running advanced tracking...")

    # Load video info for FPS
    video_info_path = Path("out/video_info.json")
    if video_info_path.exists():
        with video_info_path.open("r") as f:
            video_info = json.load(f)
        fps = video_info["fps"]
    else:
        fps = 30.0  # fallback

    all_tracks = []
    id_mapping = {}  # Map UUID strings to integer IDs

    # Process each frame
    for frame_idx in sorted(frame_detections.keys()):
        frame_dets = frame_detections[frame_idx]
        timestamp = frame_idx / fps if fps > 0 else 0.0

        # Convert detections to motpy format
        motpy_detections = [convert_to_motpy_format(det) for det in frame_dets]

        # Update tracker
        tracker.step(motpy_detections)

        # Get active tracks
        active_tracks = tracker.active_tracks()

        # Convert back to our format
        for track in active_tracks:
            track_data = convert_from_motpy_format(
                track, frame_idx, timestamp, id_mapping
            )
            all_tracks.append(track_data)

        if frame_idx % 20 == 0:
            print(f"Processed frame {frame_idx}, active tracks: {len(active_tracks)}")

    # Save tracks
    print(f"Saving {len(all_tracks)} tracks to {output_path}...")
    with output_path.open("w") as outf:
        for track in all_tracks:
            outf.write(json.dumps(track, separators=(",", ":")) + "\n")

    print("Advanced tracking complete!")

    # Print statistics
    unique_ids = set(track["obj_id"] for track in all_tracks)
    print("\nTracking Statistics:")
    print(f"Total tracks: {len(all_tracks)}")
    print(f"Unique IDs: {len(unique_ids)}")

    # Track length analysis
    id_counts = defaultdict(int)
    for track in all_tracks:
        id_counts[track["obj_id"]] += 1

    if id_counts:
        avg_length = sum(id_counts.values()) / len(id_counts)
        max_length = max(id_counts.values())
        print(f"Average track length: {avg_length:.1f} frames")
        print(f"Longest track: {max_length} frames")

        # Show longest tracks
        sorted_tracks = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nLongest tracks:")
        for track_id, length in sorted_tracks[:10]:
            print(f"  ID {track_id}: {length} frames")


def main():
    parser = argparse.ArgumentParser(description="Advanced object tracking with motpy")
    parser.add_argument(
        "--detections", type=str, required=True, help="Input detections JSONL file"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output tracks JSONL file"
    )

    # Motpy parameters
    parser.add_argument(
        "--dt", type=float, default=1 / 30.0, help="Time step between frames"
    )
    parser.add_argument(
        "--max-staleness", type=int, default=12, help="Frames to keep lost tracks"
    )

    args = parser.parse_args()

    detections_path = Path(args.detections)
    output_path = Path(args.out)

    if not detections_path.exists():
        print(f"Detections file not found: {detections_path}", file=sys.stderr)
        sys.exit(1)

    try:
        track_with_motpy(
            detections_path=detections_path,
            output_path=output_path,
            dt=args.dt,
            max_staleness=args.max_staleness,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
