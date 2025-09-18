#!/usr/bin/env python3
"""
Fix track ID consistency by building track chains and reassigning IDs.

This script addresses the ByteTracker issue where objects get new IDs every ~30 frames
instead of maintaining consistent tracking. The approach:

1. Load all tracks and group by frame
2. Build track chains using spatial-temporal continuity
3. Reassign IDs to ensure consistency across the entire video
4. Never create duplicate IDs in the same frame

Key insight: Instead of trying to merge existing IDs (which creates conflicts),
we rebuild the ID assignments from scratch using track continuity analysis.
"""

import argparse
import json
import math
from collections import defaultdict
from typing import List, Tuple, Optional


def load_tracks(tracks_file: str) -> List[dict]:
    """Load all tracking data."""
    tracks = []
    with open(tracks_file, "r") as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line.strip()))
    return tracks


def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Euclidean distance between centers of two bounding boxes."""
    center1 = calculate_bbox_center(bbox1)
    center2 = calculate_bbox_center(bbox2)
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
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


def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1) * (y2 - y1))


def is_good_continuation(
    track1: dict,
    track2: dict,
    max_distance: float = 80.0,
    min_iou: float = 0.05,
    max_velocity: float = 100.0,
    max_size_ratio: float = 2.0,
) -> bool:
    """
    Check if track2 could be a continuation of track1.
    Uses conservative thresholds to avoid merging different objects.
    """
    frame_gap = track2["frame"] - track1["frame"]

    # Must be in future frames and within reasonable gap
    if frame_gap <= 0 or frame_gap > 10:
        return False

    # Must be same class
    if track1["cls"] != track2["cls"]:
        return False

    bbox1, bbox2 = track1["bbox"], track2["bbox"]

    # Distance check
    distance = calculate_distance(bbox1, bbox2)
    if distance > max_distance:
        return False

    # IoU check - must have some overlap potential
    iou = calculate_iou(bbox1, bbox2)
    if iou < min_iou:
        return False

    # Velocity check - movement shouldn't be too fast
    velocity = distance / frame_gap
    if velocity > max_velocity:
        return False

    # Size consistency check
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    if area1 > 0 and area2 > 0:
        size_ratio = max(area1, area2) / min(area1, area2)
        if size_ratio > max_size_ratio:
            return False

    return True


def find_best_continuation(track: dict, candidate_tracks: List[dict]) -> Optional[dict]:
    """Find the best continuation for a track from a list of candidates."""
    best_track = None
    best_score = -1

    for candidate in candidate_tracks:
        if is_good_continuation(track, candidate):
            # Score based on IoU and distance (higher IoU, lower distance = better)
            iou = calculate_iou(track["bbox"], candidate["bbox"])
            distance = calculate_distance(track["bbox"], candidate["bbox"])

            # Normalize distance by dividing by frame gap to account for time
            frame_gap = candidate["frame"] - track["frame"]
            normalized_distance = distance / frame_gap if frame_gap > 0 else distance

            # Higher score is better: high IoU, low normalized distance
            score = iou - (normalized_distance / 200.0)  # Scale distance appropriately

            if score > best_score:
                best_score = score
                best_track = candidate

    return best_track


def build_track_chains(tracks: List[dict]) -> List[List[dict]]:
    """
    Build track chains by linking tracks that represent the same object.
    Returns a list of track chains, where each chain is a list of tracks
    representing the same object across time.
    """
    # Group tracks by frame for efficient lookup
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track["frame"]].append(track)

    # Sort frames
    sorted_frames = sorted(tracks_by_frame.keys())

    # Build chains
    chains = []
    used_tracks = set()  # Keep track of which tracks are already in chains

    # Start chains from the first frame
    for track in tracks_by_frame[sorted_frames[0]]:
        if id(track) in used_tracks:
            continue

        chain = [track]
        used_tracks.add(id(track))
        current_track = track

        # Try to extend the chain forward through subsequent frames
        for frame in sorted_frames[1:]:
            if frame <= current_track["frame"]:
                continue

            # Find unused tracks in this frame
            available_tracks = [
                t for t in tracks_by_frame[frame] if id(t) not in used_tracks
            ]

            if not available_tracks:
                continue

            # Find best continuation
            next_track = find_best_continuation(current_track, available_tracks)

            if next_track:
                chain.append(next_track)
                used_tracks.add(id(next_track))
                current_track = next_track

        chains.append(chain)

    # Handle remaining unused tracks (start new chains from any frame)
    for frame in sorted_frames:
        unused_tracks = [t for t in tracks_by_frame[frame] if id(t) not in used_tracks]

        for track in unused_tracks:
            chain = [track]
            used_tracks.add(id(track))
            current_track = track

            # Try to extend forward
            for future_frame in sorted_frames:
                if future_frame <= current_track["frame"]:
                    continue

                available_tracks = [
                    t for t in tracks_by_frame[future_frame] if id(t) not in used_tracks
                ]

                if not available_tracks:
                    continue

                next_track = find_best_continuation(current_track, available_tracks)

                if next_track:
                    chain.append(next_track)
                    used_tracks.add(id(next_track))
                    current_track = next_track

            chains.append(chain)

    return chains


def reassign_ids(chains: List[List[dict]]) -> List[dict]:
    """
    Reassign IDs to track chains to ensure consistency.
    Each chain gets a single ID, and tracks are returned in frame order.
    """
    new_tracks = []

    for chain_id, chain in enumerate(chains, 1):
        for track in chain:
            new_track = track.copy()
            new_track["obj_id"] = chain_id
            new_tracks.append(new_track)

    # Sort by frame to maintain original order
    new_tracks.sort(key=lambda t: (t["frame"], t["obj_id"]))

    return new_tracks


def validate_tracks(tracks: List[dict]) -> bool:
    """Validate that no frame has duplicate IDs."""
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track["frame"]].append(track["obj_id"])

    is_valid = True
    for frame, ids in tracks_by_frame.items():
        if len(ids) != len(set(ids)):
            print(f"ERROR: Frame {frame} has duplicate IDs: {ids}")
            is_valid = False

    return is_valid


def print_statistics(original_tracks: List[dict], fixed_tracks: List[dict]):
    """Print before/after statistics."""
    original_ids = set(t["obj_id"] for t in original_tracks)
    fixed_ids = set(t["obj_id"] for t in fixed_tracks)

    print(f"Original unique IDs: {len(original_ids)}")
    print(f"Fixed unique IDs: {len(fixed_ids)}")
    print(
        f"Reduction: {len(original_ids)} -> {len(fixed_ids)} ({len(original_ids) - len(fixed_ids)} fewer IDs)"
    )

    # Track length analysis
    original_id_counts = defaultdict(int)
    fixed_id_counts = defaultdict(int)

    for track in original_tracks:
        original_id_counts[track["obj_id"]] += 1

    for track in fixed_tracks:
        fixed_id_counts[track["obj_id"]] += 1

    print(
        f"\nOriginal avg track length: {sum(original_id_counts.values()) / len(original_id_counts):.1f}"
    )
    print(
        f"Fixed avg track length: {sum(fixed_id_counts.values()) / len(fixed_id_counts):.1f}"
    )

    # Show longest tracks in fixed version
    print("\nLongest tracks in fixed version:")
    sorted_fixed = sorted(fixed_id_counts.items(), key=lambda x: x[1], reverse=True)
    for obj_id, count in sorted_fixed[:10]:
        print(f"  ID {obj_id}: {count} frames")


def main():
    parser = argparse.ArgumentParser(description="Fix track ID consistency")
    parser.add_argument("--tracks", required=True, help="Input tracks JSONL file")
    parser.add_argument("--out", required=True, help="Output fixed tracks JSONL file")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=80.0,
        help="Maximum distance between tracks to consider linking",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.05,
        help="Minimum IoU to consider linking tracks",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=100.0,
        help="Maximum velocity (pixels/frame) to consider linking",
    )
    parser.add_argument(
        "--max-size-ratio",
        type=float,
        default=2.0,
        help="Maximum size ratio to consider linking tracks",
    )

    args = parser.parse_args()

    print(f"Loading tracks from {args.tracks}...")
    tracks = load_tracks(args.tracks)
    print(f"Loaded {len(tracks)} track records")

    print("Building track chains...")
    chains = build_track_chains(tracks)
    print(f"Built {len(chains)} track chains")

    print("Reassigning IDs...")
    fixed_tracks = reassign_ids(chains)

    print("Validating fixed tracks...")
    if not validate_tracks(fixed_tracks):
        print("ERROR: Validation failed!")
        return 1

    print("Validation passed!")

    # Print statistics
    print_statistics(tracks, fixed_tracks)

    # Save fixed tracks
    print(f"\nSaving fixed tracks to {args.out}...")
    with open(args.out, "w") as f:
        for track in fixed_tracks:
            f.write(json.dumps(track, separators=(",", ":")) + "\n")

    print("Track ID fixing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
