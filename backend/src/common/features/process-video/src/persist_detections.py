import json
from pathlib import Path
from typing import List, Dict, Any
import supervision as sv


def write_detection(detections, model, frame_count, fps, video_id, jsonl_path):
    """Write detection data to JSONL file (legacy function for backward compatibility)."""
    jsonl = open(jsonl_path, "a", encoding="utf-8")
    time_sec = frame_count / fps

    # class name map if available
    class_names = getattr(getattr(model, "model", None), "names", {}) or {}

    for di in range(len(detections)):
        # tracker id
        tid = None if detections.tracker_id is None else detections.tracker_id[di]
        # bbox
        x1, y1, x2, y2 = map(float, detections.xyxy[di])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # confidence
        if hasattr(detections, "confidence") and detections.confidence is not None:
            conf = float(detections.confidence[di])

        # class id / name
        if detections.class_id is not None:
            cls_id = int(detections.class_id[di])
        cls_name = class_names.get(cls_id, str(cls_id))

        rec = {
            "video_id": video_id,
            "frame": frame_count,  # 0-based
            "time": round(time_sec, 3),
            "track_id": int(tid) if tid is not None else None,
            "det_idx": di,
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
            "center": [cx, cy],
        }
        jsonl.write(json.dumps(rec) + "\n")
    jsonl.close()


def write_detections_to_jsonl(
    detections: sv.Detections,
    model,
    frame_count: int,
    fps: float,
    video_id: str,
    jsonl_path: Path,
) -> None:
    """
    Write detection data to JSONL file.

    Args:
        detections: Supervision detections object
        model: YOLO model instance
        frame_count: Current frame number (0-based)
        fps: Video frame rate
        video_id: Video identifier
        jsonl_path: Path to JSONL file
    """
    time_sec = frame_count / fps

    # class name map if available
    class_names = getattr(getattr(model, "model", None), "names", {}) or {}

    with open(jsonl_path, "a", encoding="utf-8") as jsonl:
        for di in range(len(detections)):
            # tracker id
            tid = None if detections.tracker_id is None else detections.tracker_id[di]
            # bbox
            x1, y1, x2, y2 = map(float, detections.xyxy[di])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # confidence
            conf = 0.0
            if hasattr(detections, "confidence") and detections.confidence is not None:
                conf = float(detections.confidence[di])

            # class id / name
            cls_id = 0
            if detections.class_id is not None:
                cls_id = int(detections.class_id[di])
            cls_name = class_names.get(cls_id, str(cls_id))

            rec = {
                "video_id": video_id,
                "frame": frame_count,  # 0-based
                "time": round(time_sec, 3),
                "track_id": int(tid) if tid is not None else None,
                "det_idx": di,
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": round(conf, 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center": [cx, cy],
            }
            jsonl.write(json.dumps(rec) + "\n")


def read_detections_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Read detection data from JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of detection dictionaries
    """
    detections = []
    if not jsonl_path.exists():
        return detections

    with open(jsonl_path, "r", encoding="utf-8") as jsonl:
        for line in jsonl:
            line = line.strip()
            if line:
                detections.append(json.loads(line))

    return detections


def get_detections_by_frame(
    detections: List[Dict[str, Any]], frame_number: int
) -> List[Dict[str, Any]]:
    """
    Filter detections by frame number.

    Args:
        detections: List of detection dictionaries
        frame_number: Frame number to filter by

    Returns:
        List of detections for the specified frame
    """
    return [det for det in detections if det["frame"] == frame_number]


def get_detections_by_track_id(
    detections: List[Dict[str, Any]], track_id: int
) -> List[Dict[str, Any]]:
    """
    Filter detections by track ID.

    Args:
        detections: List of detection dictionaries
        track_id: Track ID to filter by

    Returns:
        List of detections for the specified track ID
    """
    return [det for det in detections if det.get("track_id") == track_id]


def convert_jsonl_to_supervision_detections(
    detections: List[Dict[str, Any]],
) -> sv.Detections:
    """
    Convert JSONL detection data back to supervision Detections format.

    Args:
        detections: List of detection dictionaries

    Returns:
        Supervision Detections object
    """
    if not detections:
        return sv.Detections.empty()

    # Extract data
    xyxy = []
    confidence = []
    class_id = []
    tracker_id = []

    for det in detections:
        xyxy.append(det["bbox_xyxy"])
        confidence.append(det["conf"])
        class_id.append(det["class_id"])
        tracker_id.append(det.get("track_id"))

    # Convert to numpy arrays
    import numpy as np

    xyxy = np.array(xyxy)
    confidence = np.array(confidence)
    class_id = np.array(class_id)
    tracker_id = (
        np.array(tracker_id) if any(tid is not None for tid in tracker_id) else None
    )

    return sv.Detections(
        xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id
    )
