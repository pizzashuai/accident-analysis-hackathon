#!/usr/bin/env python3
"""
Visual overlay script for rendering tracking data on video.
Shows object IDs, bounding boxes, and trails.
"""

import argparse
import json
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Deque


def load_tracks(tracks_file: str) -> Dict[int, List[dict]]:
    """Load tracking data grouped by frame."""
    tracks_by_frame = defaultdict(list)

    with open(tracks_file, "r") as f:
        for line in f:
            track = json.loads(line.strip())
            frame_num = track["frame"]
            tracks_by_frame[frame_num].append(track)

    return tracks_by_frame


def get_trail_positions(
    track_history: Dict[int, Deque], obj_id: int, trail_length: int
) -> List[Tuple[int, int]]:
    """Get trail positions for an object."""
    if obj_id not in track_history:
        return []

    history = track_history[obj_id]
    trail_positions = []

    for bbox in list(history)[-trail_length:]:
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        trail_positions.append((center_x, center_y))

    return trail_positions


def draw_trail(
    img: np.ndarray,
    trail_positions: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    """Draw trail lines connecting positions."""
    if len(trail_positions) < 2:
        return

    for i in range(1, len(trail_positions)):
        pt1 = trail_positions[i - 1]
        pt2 = trail_positions[i]
        cv2.line(img, pt1, pt2, color, thickness)


def get_color_for_id(obj_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for an object ID."""
    # Use a simple hash to get consistent colors
    np.random.seed(obj_id)
    color = np.random.randint(0, 255, 3).tolist()
    return tuple(map(int, color))


def main():
    parser = argparse.ArgumentParser(description="Render tracking overlay on video")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--tracks", required=True, help="Tracks JSONL file")
    parser.add_argument("--out", required=True, help="Output video file")
    parser.add_argument("--trail", type=int, default=10, help="Trail length in frames")
    parser.add_argument(
        "--box-thickness", type=int, default=2, help="Bounding box thickness"
    )
    parser.add_argument("--text-scale", type=float, default=0.7, help="Text scale")
    parser.add_argument("--text-thickness", type=int, default=2, help="Text thickness")

    args = parser.parse_args()

    # Load tracking data
    print(f"Loading tracks from {args.tracks}...")
    tracks_by_frame = load_tracks(args.tracks)
    print(f"Loaded tracks for {len(tracks_by_frame)} frames")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return 1

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Track history for trails
    track_history: Dict[int, Deque] = defaultdict(lambda: deque(maxlen=args.trail))

    frame_count = 0
    processed_frames = 0

    print("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get tracks for current frame
        current_tracks = tracks_by_frame.get(frame_count, [])

        # Update track history and draw trails
        for track in current_tracks:
            obj_id = track["obj_id"]
            bbox = track["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Add to history
            track_history[obj_id].append(bbox)

            # Get color for this object
            color = get_color_for_id(obj_id)

            # Draw trail
            trail_positions = get_trail_positions(track_history, obj_id, args.trail)
            draw_trail(frame, trail_positions, color, args.box_thickness)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.box_thickness)

            # Draw object ID
            label = f"ID:{obj_id}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, args.text_scale, args.text_thickness
            )[0]

            # Position label above the bounding box
            label_x = x1
            label_y = max(y1 - 10, label_size[1] + 10)

            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (label_x, label_y - label_size[1] - 5),
                (label_x + label_size[0] + 10, label_y + 5),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                frame,
                label,
                (label_x + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                args.text_scale,
                (255, 255, 255),
                args.text_thickness,
            )

        # Write frame
        out.write(frame)
        processed_frames += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Overlay video saved to {args.out}")
    print(f"Processed {processed_frames} frames")

    return 0


if __name__ == "__main__":
    exit(main())
