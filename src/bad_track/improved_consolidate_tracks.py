#!/usr/bin/env python3
"""
Improved track consolidation that prevents merging different objects.
Uses motion prediction, stricter spatial criteria, and temporal consistency.
"""

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def load_tracks(tracks_file: str) -> List[dict]:
    """Load all tracking data."""
    tracks = []
    with open(tracks_file, "r") as f:
        for line in f:
            tracks.append(json.loads(line.strip()))
    return tracks


def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1) * (y2 - y1))


def calculate_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate distance between centers of two bounding boxes."""
    center1 = calculate_bbox_center(bbox1)
    center2 = calculate_bbox_center(bbox2)
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU (Intersection over Union) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def predict_next_position(
    bbox: List[float], velocity: Tuple[float, float]
) -> List[float]:
    """Predict next position based on current bbox and velocity."""
    x1, y1, x2, y2 = bbox
    vx, vy = velocity
    return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]


def calculate_velocity(
    bbox1: List[float], bbox2: List[float], frame_gap: int
) -> Tuple[float, float]:
    """Calculate velocity between two bboxes."""
    center1 = calculate_bbox_center(bbox1)
    center2 = calculate_bbox_center(bbox2)
    if frame_gap == 0:
        return (0.0, 0.0)
    return (
        (center2[0] - center1[0]) / frame_gap,
        (center2[1] - center1[1]) / frame_gap,
    )


def is_valid_track_continuation(
    track1: dict,
    track2: dict,
    max_distance: float = 30.0,
    min_iou: float = 0.3,
    max_velocity_change: float = 50.0,
    max_size_change: float = 0.5,
) -> bool:
    """
    Check if track2 is a valid continuation of track1.
    Uses multiple criteria to prevent merging different objects.
    """
    bbox1, bbox2 = track1["bbox"], track2["bbox"]
    frame_gap = track2["frame"] - track1["frame"]

    if frame_gap <= 0:
        return False

    # 1. Distance check - must be close enough
    distance = calculate_distance(bbox1, bbox2)
    if distance > max_distance:
        return False

    # 2. IoU check - must have significant overlap
    iou = calculate_bbox_overlap(bbox1, bbox2)
    if iou < min_iou:
        return False

    # 3. Velocity consistency check
    velocity = calculate_velocity(bbox1, bbox2, frame_gap)
    velocity_magnitude = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

    # If velocity is too high, it's probably a different object
    if velocity_magnitude > max_velocity_change:
        return False

    # 4. Size consistency check
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    if area1 > 0:
        size_ratio = area2 / area1
        if size_ratio < (1 - max_size_change) or size_ratio > (1 + max_size_change):
            return False

    return True


def find_track_continuations(
    tracks: List[dict],
    max_distance: float = 30.0,
    min_iou: float = 0.3,
    max_velocity_change: float = 50.0,
    max_size_change: float = 0.5,
    max_frame_gap: int = 3,
) -> Dict[int, int]:
    """
    Find track continuations using improved criteria.
    Returns mapping from old_id to new_id for tracks that should be merged.
    """
    # Group tracks by frame
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track["frame"]].append(track)

    # Build track chains
    track_chains = {}  # old_id -> list of (frame, bbox, track)
    id_mapping = {}  # old_id -> representative_id

    frames = sorted(tracks_by_frame.keys())

    for frame in frames:
        current_tracks = tracks_by_frame[frame]

        for track in current_tracks:
            old_id = track["obj_id"]
            bbox = track["bbox"]

            # Try to find a continuation from previous frames
            best_continuation = None
            best_score = -1

            for prev_frame in range(max(0, frame - max_frame_gap), frame):
                if prev_frame not in tracks_by_frame:
                    continue

                for prev_track in tracks_by_frame[prev_frame]:
                    if is_valid_track_continuation(
                        prev_track,
                        track,
                        max_distance,
                        min_iou,
                        max_velocity_change,
                        max_size_change,
                    ):
                        # Calculate a score based on IoU and distance
                        iou = calculate_bbox_overlap(prev_track["bbox"], bbox)
                        distance = calculate_distance(prev_track["bbox"], bbox)
                        score = iou - (distance / 100.0)  # Normalize distance

                        if score > best_score:
                            best_score = score
                            best_continuation = prev_track["obj_id"]

            if best_continuation is not None:
                # Use the representative ID of the continuation
                representative_id = id_mapping.get(best_continuation, best_continuation)
                id_mapping[old_id] = representative_id

                # Update the chain
                if representative_id not in track_chains:
                    track_chains[representative_id] = []
                track_chains[representative_id].append((frame, bbox, track))
            else:
                # This is a new track
                id_mapping[old_id] = old_id
                track_chains[old_id] = [(frame, bbox, track)]

    return id_mapping


def consolidate_tracks(tracks: List[dict], id_mapping: Dict[int, int]) -> List[dict]:
    """Consolidate tracks by remapping IDs."""
    consolidated = []

    for track in tracks:
        new_track = track.copy()
        old_id = track["obj_id"]
        new_track["obj_id"] = id_mapping.get(old_id, old_id)
        consolidated.append(new_track)

    return consolidated


def validate_consolidation(tracks: List[dict]) -> bool:
    """Validate that no frame has duplicate IDs."""
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track["frame"]].append(track)

    for frame, frame_tracks in tracks_by_frame.items():
        ids = [t["obj_id"] for t in frame_tracks]
        if len(ids) != len(set(ids)):
            print(f"ERROR: Frame {frame} has duplicate IDs: {ids}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Improved track consolidation")
    parser.add_argument("--tracks", required=True, help="Input tracks JSONL file")
    parser.add_argument(
        "--out", required=True, help="Output consolidated tracks JSONL file"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=30.0,
        help="Maximum distance between centers to consider merging",
    )
    parser.add_argument(
        "--min-iou", type=float, default=0.3, help="Minimum IoU to consider merging"
    )
    parser.add_argument(
        "--max-velocity-change",
        type=float,
        default=50.0,
        help="Maximum velocity change to consider merging",
    )
    parser.add_argument(
        "--max-size-change",
        type=float,
        default=0.5,
        help="Maximum size change ratio to consider merging",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=3,
        help="Maximum frame gap to consider merging",
    )

    args = parser.parse_args()

    print(f"Loading tracks from {args.tracks}...")
    tracks = load_tracks(args.tracks)
    print(f"Loaded {len(tracks)} track records")

    # Count unique IDs before consolidation
    original_ids = set(track["obj_id"] for track in tracks)
    print(f"Original unique IDs: {len(original_ids)}")

    print("Finding track continuations with improved criteria...")
    id_mapping = find_track_continuations(
        tracks,
        args.max_distance,
        args.min_iou,
        args.max_velocity_change,
        args.max_size_change,
        args.max_frame_gap,
    )

    print(f"Found {len(id_mapping)} ID mappings")

    print("Consolidating tracks...")
    consolidated_tracks = consolidate_tracks(tracks, id_mapping)

    # Validate consolidation
    if not validate_consolidation(consolidated_tracks):
        print("ERROR: Consolidation validation failed!")
        return 1

    # Count unique IDs after consolidation
    final_ids = set(track["obj_id"] for track in consolidated_tracks)
    print(f"Consolidated unique IDs: {len(final_ids)}")
    print(f"Reduced from {len(original_ids)} to {len(final_ids)} IDs")

    # Save consolidated tracks
    print(f"Saving consolidated tracks to {args.out}...")
    with open(args.out, "w") as f:
        for track in consolidated_tracks:
            f.write(json.dumps(track) + "\n")

    print("Consolidation complete!")

    # Show some statistics about the consolidation
    id_counts = defaultdict(int)
    for track in consolidated_tracks:
        id_counts[track["obj_id"]] += 1

    print("\nTrack length distribution after consolidation:")
    for obj_id, count in sorted(id_counts.items()):
        print(f"ID {obj_id}: {count} frames")

    return 0


if __name__ == "__main__":
    exit(main())
