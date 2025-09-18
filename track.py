#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Optional imports for real-time detection
try:
    from ultralytics import YOLO
    import cv2

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class ByteTracker:
    """Simple ByteTrack implementation for object tracking."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        self.tracked_objects = {}  # {track_id: {'bbox': [x1,y1,x2,y2], 'score': float, 'frame_count': int}}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracker with new detections."""
        self.frame_count += 1

        # Filter detections by confidence
        high_conf_dets = [d for d in detections if d["score"] >= self.track_thresh]
        low_conf_dets = [d for d in detections if d["score"] < self.track_thresh]

        # Update existing tracks
        updated_tracks = []
        for track_id, track in self.tracked_objects.items():
            track["frame_count"] += 1

            # Find best matching detection
            best_match = None
            best_iou = 0

            for i, det in enumerate(high_conf_dets):
                if det.get("assigned", False):
                    continue

                iou = self._calculate_iou(track["bbox"], det["bbox"])
                if iou > self.match_thresh and iou > best_iou:
                    best_iou = iou
                    best_match = (i, det)

            if best_match:
                # Update track
                i, det = best_match
                track["bbox"] = det["bbox"]
                track["score"] = det["score"]
                track["cls"] = det["cls"]
                high_conf_dets[i]["assigned"] = True

                updated_tracks.append(
                    {
                        "obj_id": track_id,
                        "bbox": track["bbox"],
                        "score": track["score"],
                        "cls": track["cls"],
                        "frame": self.frame_count - 1,
                        "t": det["t"],
                    }
                )
            else:
                # Track lost, but keep for a few frames
                if track["frame_count"] <= self.track_buffer:
                    updated_tracks.append(
                        {
                            "obj_id": track_id,
                            "bbox": track["bbox"],
                            "score": track["score"],
                            "cls": track["cls"],
                            "frame": self.frame_count - 1,
                            "t": detections[0]["t"] if detections else 0.0,
                        }
                    )

        # Create new tracks for unassigned high confidence detections
        for det in high_conf_dets:
            if not det.get("assigned", False):
                track_id = self.next_id
                self.next_id += 1

                self.tracked_objects[track_id] = {
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "cls": det["cls"],
                    "frame_count": 1,
                }

                updated_tracks.append(
                    {
                        "obj_id": track_id,
                        "bbox": det["bbox"],
                        "score": det["score"],
                        "cls": det["cls"],
                        "frame": self.frame_count - 1,
                        "t": det["t"],
                    }
                )

        # Clean up old tracks
        self.tracked_objects = {
            tid: track
            for tid, track in self.tracked_objects.items()
            if track["frame_count"] <= self.track_buffer
        }

        return updated_tracks

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
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


def track_video(
    video_path: Path,
    config_path: Path,
    tracker_config_path: Path,
    output_path: Path,
    detections_path: Path = None,
):
    """Run tracking on video with detections."""

    # Load detection config
    with config_path.open("r") as f:
        detect_config = json.load(f)

    # Load tracker config
    if tracker_config_path.suffix == ".yaml" or tracker_config_path.suffix == ".yml":
        import yaml

        with tracker_config_path.open("r") as f:
            tracker_config = yaml.safe_load(f)
    else:
        with tracker_config_path.open("r") as f:
            tracker_config = json.load(f)

    # Initialize tracker
    tracker = ByteTracker(
        track_thresh=tracker_config.get("track_thresh", 0.5),
        track_buffer=tracker_config.get("track_buffer", 30),
        match_thresh=tracker_config.get("match_thresh", 0.8),
    )

    # Load video info
    video_info_path = Path("out/video_info.json")
    if video_info_path.exists():
        with video_info_path.open("r") as f:
            video_info = json.load(f)
        fps = video_info["fps"]
    else:
        fps = 30.0  # fallback

    # If detections file provided, use pre-computed detections
    if detections_path and detections_path.exists():
        print(f"Using pre-computed detections from {detections_path}")
        detections = load_detections(detections_path)
        frame_detections = group_detections_by_frame(detections)

        # Process each frame
        with output_path.open("w") as outf:
            for frame_idx in sorted(frame_detections.keys()):
                frame_dets = frame_detections[frame_idx]
                timestamp = frame_idx / fps if fps > 0 else 0.0

                # Add timestamp to detections
                for det in frame_dets:
                    det["t"] = timestamp

                # Update tracker
                tracks = tracker.update(frame_dets)

                # Write tracks to output
                for track in tracks:
                    outf.write(json.dumps(track, separators=(",", ":")) + "\n")

    else:
        # Run detection and tracking in real-time
        print("Running detection and tracking in real-time...")

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Load model
        model = YOLO(detect_config["model"])

        with output_path.open("w") as outf:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection
                results = model(
                    frame,
                    conf=detect_config["conf"],
                    classes=detect_config["classes"],
                    verbose=False,
                )[0]

                # Calculate timestamp
                timestamp = frame_idx / fps if fps > 0 else 0.0

                # Process detections
                detections = []
                if results.boxes is not None and len(results.boxes) > 0:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy()

                    for bbox, score, cls_id in zip(xyxy, scores, cls_ids):
                        x1, y1, x2, y2 = bbox
                        detections.append(
                            {
                                "frame": frame_idx,
                                "t": timestamp,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "cls": int(cls_id),
                                "score": float(score),
                            }
                        )

                # Update tracker
                tracks = tracker.update(detections)

                # Write tracks to output
                for track in tracks:
                    outf.write(json.dumps(track, separators=(",", ":")) + "\n")

                frame_idx += 1

        cap.release()

    print(f"Tracking complete. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Object tracking with ByteTrack")
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--config",
        type=str,
        default="src/detect.config.json",
        help="Detection config file",
    )
    parser.add_argument(
        "--tracker", type=str, default="bytetrack.yaml", help="Tracker config file"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output tracks JSONL file"
    )
    parser.add_argument(
        "--dets", type=str, help="Pre-computed detections JSONL file (optional)"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)
    tracker_config_path = Path(args.tracker)
    output_path = Path(args.out)
    detections_path = Path(args.dets) if args.dets else None

    if not video_path.exists():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if not tracker_config_path.exists():
        print(f"Tracker config file not found: {tracker_config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        track_video(
            video_path, config_path, tracker_config_path, output_path, detections_path
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
