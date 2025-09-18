#!/usr/bin/env python3
"""
Consolidate fragmented tracking data by merging IDs that belong to the same object.
Uses spatial proximity and temporal continuity to identify track fragments.
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import math


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


def find_track_fragments(
    tracks: List[dict],
    max_distance: float = 50.0,
    min_iou: float = 0.1,
    max_frame_gap: int = 5,
) -> Dict[int, Set[int]]:
    """
    Find track fragments that should be merged.
    Returns a mapping from representative ID to set of IDs to merge.
    """
    # Group tracks by frame
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track["frame"]].append(track)

    # Build adjacency graph of IDs that are close in space and time
    id_connections = defaultdict(set)

    frames = sorted(tracks_by_frame.keys())
    for i in range(len(frames) - 1):
        current_frame = frames[i]
        next_frame = frames[i + 1]
        frame_gap = next_frame - current_frame

        if frame_gap > max_frame_gap:
            continue

        current_tracks = tracks_by_frame[current_frame]
        next_tracks = tracks_by_frame[next_frame]

        for track1 in current_tracks:
            for track2 in next_tracks:
                # Skip if same ID
                if track1["obj_id"] == track2["obj_id"]:
                    continue

                # Check if tracks are close enough to be the same object
                distance = calculate_distance(track1["bbox"], track2["bbox"])
                iou = calculate_bbox_overlap(track1["bbox"], track2["bbox"])

                if distance <= max_distance and iou >= min_iou:
                    id1, id2 = track1["obj_id"], track2["obj_id"]
                    id_connections[id1].add(id2)
                    id_connections[id2].add(id1)

    # Find connected components (groups of IDs that should be merged)
    visited = set()
    id_groups = []

    for obj_id in id_connections:
        if obj_id in visited:
            continue

        # BFS to find all connected IDs
        group = set()
        queue = [obj_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue

            visited.add(current_id)
            group.add(current_id)

            for connected_id in id_connections[current_id]:
                if connected_id not in visited:
                    queue.append(connected_id)

        if len(group) > 1:  # Only consider groups with multiple IDs
            id_groups.append(group)

    # Create mapping from each ID to its representative ID
    id_to_representative = {}
    for group in id_groups:
        representative = min(group)  # Use smallest ID as representative
        for obj_id in group:
            id_to_representative[obj_id] = representative

    return id_to_representative


def consolidate_tracks(tracks: List[dict], id_mapping: Dict[int, int]) -> List[dict]:
    """Consolidate tracks by remapping IDs."""
    consolidated = []

    for track in tracks:
        new_track = track.copy()
        old_id = track["obj_id"]
        new_track["obj_id"] = id_mapping.get(old_id, old_id)
        consolidated.append(new_track)

    return consolidated


def main():
    parser = argparse.ArgumentParser(description="Consolidate fragmented tracking data")
    parser.add_argument("--tracks", required=True, help="Input tracks JSONL file")
    parser.add_argument(
        "--out", required=True, help="Output consolidated tracks JSONL file"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Maximum distance between centers to consider merging",
    )
    parser.add_argument(
        "--min-iou", type=float, default=0.1, help="Minimum IoU to consider merging"
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=5,
        help="Maximum frame gap to consider merging",
    )

    args = parser.parse_args()

    print(f"Loading tracks from {args.tracks}...")
    tracks = load_tracks(args.tracks)
    print(f"Loaded {len(tracks)} track records")

    # Count unique IDs before consolidation
    original_ids = set(track["obj_id"] for track in tracks)
    print(f"Original unique IDs: {len(original_ids)}")

    print("Finding track fragments to consolidate...")
    id_mapping = find_track_fragments(
        tracks, args.max_distance, args.min_iou, args.max_frame_gap
    )

    print(f"Found {len(id_mapping)} IDs to remap")

    # Add identity mapping for IDs that don't need remapping
    all_ids = set(track["obj_id"] for track in tracks)
    for obj_id in all_ids:
        if obj_id not in id_mapping:
            id_mapping[obj_id] = obj_id

    print("Consolidating tracks...")
    consolidated_tracks = consolidate_tracks(tracks, id_mapping)  # type: ignore

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


if __name__ == "__main__":
    main()
