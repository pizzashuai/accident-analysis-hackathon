from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
import json
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
# Removed shapely dependency - implementing polygon intersection manually

from persist_detections import read_detections_from_jsonl


def get_track_to_detections(
    detections: list[dict[str, Any]],
    interested_track_ids: list[int] = [],
) -> dict[int, list[dict[str, Any]]]:
    track_to_detections = defaultdict(list)
    for detection in detections:
        if interested_track_ids and detection["track_id"] not in interested_track_ids:
            continue
        track_to_detections[detection["track_id"]].append(detection)

    # Sort detections by frame
    for _, detections in track_to_detections.items():
        detections.sort(key=lambda x: x["frame"])

    return track_to_detections


def get_frame_to_detected_track_ids(
    detections: list[dict[str, Any]],
    interested_track_ids: list[int] = [],
) -> dict[int, list[int]]:
    frame_to_detected_track_ids = defaultdict(list)
    for detection in detections:
        if interested_track_ids and detection["track_id"] not in interested_track_ids:
            continue
        frame_to_detected_track_ids[detection["frame"]].append(detection["track_id"])
    return frame_to_detected_track_ids


def build_segments(
    track_detections: list[dict[str, Any]], max_gap: int = 1, min_len: int = 3
) -> list[dict]:
    """
    Build contiguous segments from track detections.

    Args:
        track_detections: List of detections for one track_id, sorted by frame
        max_gap: Maximum gap between frames to consider contiguous
        min_len: Minimum segment length to keep

    Returns:
        List of segments with start, end, len, and mean_conf
    """
    if not track_detections:
        return []

    segments = []
    current_segment = [track_detections[0]]

    for i in range(1, len(track_detections)):
        prev_frame = track_detections[i - 1]["frame"]
        curr_frame = track_detections[i]["frame"]

        if curr_frame - prev_frame <= max_gap:
            current_segment.append(track_detections[i])
        else:
            # End current segment and start new one
            if len(current_segment) >= min_len:
                segment = {
                    "start": current_segment[0]["frame"],
                    "end": current_segment[-1]["frame"],
                    "len": len(current_segment),
                    "mean_conf": sum(d["conf"] for d in current_segment)
                    / len(current_segment),
                }
                segments.append(segment)
            current_segment = [track_detections[i]]

    # Add the last segment
    if len(current_segment) >= min_len:
        segment = {
            "start": current_segment[0]["frame"],
            "end": current_segment[-1]["frame"],
            "len": len(current_segment),
            "mean_conf": sum(d["conf"] for d in current_segment) / len(current_segment),
        }
        segments.append(segment)

    return segments


def pick_anchor(segments: list[dict]) -> dict | None:
    """
    Pick the best anchor segment from a list of segments.

    Args:
        segments: List of segments from build_segments

    Returns:
        The chosen anchor segment with a 'reason' field, or None if no segments
    """
    if not segments:
        return None

    # Sort by length (descending), then by mean_conf (descending)
    sorted_segments = sorted(segments, key=lambda s: (-s["len"], -s["mean_conf"]))

    anchor = sorted_segments[0].copy()
    anchor["reason"] = "longest_then_conf"

    return anchor


def fit_motion_model(track_detections: list[dict[str, Any]], anchor: dict) -> dict:
    """
    Fit a constant-velocity + linear size drift model on the anchor segment.

    Args:
        track_detections: List of detections for one track_id
        anchor: Anchor segment from pick_anchor

    Returns:
        Dictionary with params, rmse, and prediction function
    """
    # Extract detections in anchor range
    anchor_detections = [
        d for d in track_detections if anchor["start"] <= d["frame"] <= anchor["end"]
    ]

    if len(anchor_detections) < 2:
        raise ValueError("Need at least 2 detections to fit motion model")

    # Build arrays
    frames = np.array([d["frame"] for d in anchor_detections])
    bboxes = np.array([d["bbox_xyxy"] for d in anchor_detections])

    # Convert to centers and dimensions
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    # Fit linear models: y = ax + b where x is frame, y is cx/cy/w/h
    # Using least squares: b = (X^T X)^(-1) X^T y
    X = np.column_stack([frames, np.ones(len(frames))])

    # Fit cx, cy, w, h
    coeffs_cx = np.linalg.lstsq(X, cx, rcond=None)[0]
    coeffs_cy = np.linalg.lstsq(X, cy, rcond=None)[0]
    coeffs_w = np.linalg.lstsq(X, w, rcond=None)[0]
    coeffs_h = np.linalg.lstsq(X, h, rcond=None)[0]

    # Extract parameters
    params = {
        "vx": coeffs_cx[0],  # velocity in x
        "vy": coeffs_cy[0],  # velocity in y
        "dw": coeffs_w[0],  # width drift
        "dh": coeffs_h[0],  # height drift
        "ax": coeffs_cx[1],  # intercept x
        "ay": coeffs_cy[1],  # intercept y
        "aw": coeffs_w[1],  # intercept width
        "ah": coeffs_h[1],  # intercept height
    }

    # Compute RMSE
    pred_cx = coeffs_cx[0] * frames + coeffs_cx[1]
    pred_cy = coeffs_cy[0] * frames + coeffs_cy[1]
    pred_w = coeffs_w[0] * frames + coeffs_w[1]
    pred_h = coeffs_h[0] * frames + coeffs_h[1]

    rmse = {
        "cx": float(np.sqrt(np.mean((cx - pred_cx) ** 2))),
        "cy": float(np.sqrt(np.mean((cy - pred_cy) ** 2))),
        "w": float(np.sqrt(np.mean((w - pred_w) ** 2))),
        "h": float(np.sqrt(np.mean((h - pred_h) ** 2))),
    }

    return {"params": params, "rmse": rmse}


def predict_bbox(params: dict, frame: int) -> list[float]:
    """
    Predict bbox at given frame using the motion model parameters.

    Args:
        params: Parameters from fit_motion_model
        frame: Frame number to predict

    Returns:
        [x1, y1, x2, y2] bbox coordinates
    """
    # Predict center and dimensions
    cx = params["ax"] + params["vx"] * frame
    cy = params["ay"] + params["vy"] * frame
    w = params["aw"] + params["dw"] * frame
    h = params["ah"] + params["dh"] * frame

    # Convert to bbox format
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return [x1, y1, x2, y2]


def make_rois(
    missing_frames: list[int],
    predict_bbox: Callable[[int], list[float]],
    img_size: dict[str, int],
    gap_to_anchor: dict[int, int],
    roi_policy: dict[str, float],
    aoi_polygon: list[list[float]] | None = None,
) -> list[dict]:
    """
    Task 4: Define robust ROI window per missing frame.

    Args:
        missing_frames: List of frame numbers that are missing
        predict_bbox: Function that takes frame number and returns predicted bbox [x1,y1,x2,y2]
        img_size: Dictionary with "width" and "height" keys
        gap_to_anchor: Dictionary mapping frame numbers to gap distances from anchor
        roi_policy: Dictionary with "base_scale", "per_gap_scale", "max_scale" keys
        aoi_polygon: Optional polygon defining area of interest

    Returns:
        List of ROI dictionaries with frame, pred_bbox, scale, roi_xyxy, aoi_intersected
    """
    rois = []
    width, height = img_size["width"], img_size["height"]

    # AOI polygon will be used for intersection check

    for frame in missing_frames:
        # Get predicted bbox
        pred_bbox = predict_bbox(frame)

        # Calculate scale based on gap
        gap = gap_to_anchor.get(frame, 1)
        scale = min(
            roi_policy["base_scale"] + roi_policy["per_gap_scale"] * gap,
            roi_policy["max_scale"],
        )

        # Expand bbox by scale about its center
        x1, y1, x2, y2 = pred_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Expand dimensions
        new_w = w * scale
        new_h = h * scale

        # Calculate new bbox centered on original center
        roi_x1 = center_x - new_w / 2
        roi_y1 = center_y - new_h / 2
        roi_x2 = center_x + new_w / 2
        roi_y2 = center_y + new_h / 2

        # Clip to image bounds
        roi_x1 = max(0, int(roi_x1))
        roi_y1 = max(0, int(roi_y1))
        roi_x2 = min(width, int(roi_x2))
        roi_y2 = min(height, int(roi_y2))

        roi_xyxy = [roi_x1, roi_y1, roi_x2, roi_y2]

        # Check AOI intersection
        aoi_intersected = False
        if aoi_polygon:
            roi_poly = [
                [roi_x1, roi_y1],
                [roi_x2, roi_y1],
                [roi_x2, roi_y2],
                [roi_x1, roi_y2],
            ]
            aoi_intersected = polygon_intersects(roi_poly, aoi_polygon)  # type: ignore

        rois.append(
            {
                "frame": frame,
                "pred_bbox": [round(x, 1) for x in pred_bbox],
                "scale": round(scale, 2),
                "roi_xyxy": roi_xyxy,
                "aoi_intersected": aoi_intersected,
            }
        )

    return rois


def map_crop_to_fullframe(crop_bbox: list[float], roi_xyxy: list[int]) -> list[float]:
    """
    Map crop-space bbox back to full-frame coordinates.

    Args:
        crop_bbox: Bbox in crop coordinates [x1,y1,x2,y2]
        roi_xyxy: ROI coordinates in full frame [x1,y1,x2,y2]

    Returns:
        Bbox mapped to full-frame coordinates
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_xyxy

    # Add ROI offset to crop bbox
    full_x1 = crop_bbox[0] + roi_x1
    full_y1 = crop_bbox[1] + roi_y1
    full_x2 = crop_bbox[2] + roi_x1
    full_y2 = crop_bbox[3] + roi_y1

    return [full_x1, full_y1, full_x2, full_y2]


def calculate_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate IoU between two bboxes."""
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
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_center_distance(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate center distance between two bboxes."""
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2

    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def calculate_size_ratio(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate size ratio between two bboxes."""
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    w_ratio = min(w1 / w2, w2 / w1) if w2 > 0 else 0
    h_ratio = min(h1 / h2, h2 / h1) if h2 > 0 else 0

    return w_ratio * h_ratio


def polygon_intersects(poly1: list[list[float]], poly2: list[list[float]]) -> bool:
    """Check if two polygons intersect using simple bounding box check."""

    # Get bounding boxes
    def get_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    bbox1 = get_bbox(poly1)
    bbox2 = get_bbox(poly2)

    # Check if bounding boxes overlap
    return not (
        bbox1[2] < bbox2[0]
        or bbox2[2] < bbox1[0]
        or bbox1[3] < bbox2[1]
        or bbox2[3] < bbox1[1]
    )


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def redetect_in_rois(
    rois: list[dict],
    detector_handle: Any,
    detect_params: dict[str, Any],
    class_id: int,
    video_path: str,
    gap_to_anchor: dict[int, int],
    aoi_polygon: list[list[float]] | None = None,
) -> dict:
    """
    Task 5: Run detector only inside each ROI with lowered thresholds and pick best candidate.

    Args:
        rois: List of ROI dictionaries from make_rois
        detector_handle: YOLO detector instance
        detect_params: Dictionary with conf, iou, upsample, score_min keys
        class_id: Class ID to enforce for detections
        video_path: Path to the video file
        aoi_polygon: Optional polygon defining area of interest

    Returns:
        Dictionary with fills and summary
    """
    fills = []
    attempted = 0
    filled = 0
    missed = 0

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Load detector if not provided
    if detector_handle is None:
        detector_handle = YOLO("yolov8s.pt")

    for roi in rois:
        frame_num = roi["frame"]
        pred_bbox = roi["pred_bbox"]
        roi_xyxy = roi["roi_xyxy"]

        attempted += 1

        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            fills.append(
                {"frame": frame_num, "status": "miss", "reason": "frame_not_found"}
            )
            missed += 1
            continue

        # Crop the ROI
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_xyxy
        roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi_crop.size == 0:
            fills.append({"frame": frame_num, "status": "miss", "reason": "empty_roi"})
            missed += 1
            continue

        # Run detection on ROI crop
        results = detector_handle(
            roi_crop,
            conf=detect_params["conf"],
            iou=detect_params["iou"],
            classes=[class_id],
            verbose=False,
        )[0]

        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)

        # Debug: print detection info
        # print(f"Frame {frame_num}: Found {len(detections)} detections in ROI")

        # Map detections back to full frame coordinates
        candidates = []
        for i in range(len(detections)):
            if detections.class_id is not None and detections.class_id[i] == class_id:
                # Get bbox in crop coordinates
                crop_bbox = detections.xyxy[i].tolist()

                # Map to full frame coordinates
                full_bbox = map_crop_to_fullframe(crop_bbox, roi_xyxy)

                # Get confidence
                conf = (
                    detections.confidence[i]
                    if detections.confidence is not None
                    else 0.0
                )

                candidates.append(
                    {
                        "bbox_xyxy": full_bbox,
                        "class_id": class_id,
                        "conf": conf,
                    }
                )

        # print(f"Frame {frame_num}: {len(candidates)} candidates after class filtering")

        # Filter by AOI if provided
        if aoi_polygon and candidates:
            filtered_candidates = []
            for det in candidates:
                center_x = (det["bbox_xyxy"][0] + det["bbox_xyxy"][2]) / 2
                center_y = (det["bbox_xyxy"][1] + det["bbox_xyxy"][3]) / 2
                if point_in_polygon((center_x, center_y), aoi_polygon):
                    filtered_candidates.append(det)
            candidates = filtered_candidates

        if not candidates:
            fills.append(
                {"frame": frame_num, "status": "miss", "reason": "no_candidate_passed"}
            )
            missed += 1
            continue

        # Score candidates
        best_candidate = None
        best_score = -1

        for candidate in candidates:
            det_bbox = candidate["bbox_xyxy"]

            # Calculate metrics
            iou = calculate_iou(det_bbox, pred_bbox)
            cd = calculate_center_distance(det_bbox, pred_bbox)
            sr = calculate_size_ratio(det_bbox, pred_bbox)

            # Hard gates - more lenient for early frames
            pred_w = pred_bbox[2] - pred_bbox[0]
            pred_h = pred_bbox[3] - pred_bbox[1]
            pred_diag = np.sqrt(pred_w**2 + pred_h**2)

            # Very lenient thresholds
            iou_threshold = 0.0  # No IoU requirement
            cd_threshold = 2.0 * pred_diag  # Very lenient center distance

            if iou < iou_threshold and cd > cd_threshold:
                # print(f"Frame {frame_num}: Candidate rejected - IoU={iou:.3f}, CD={cd:.1f} (thresholds: IoU>{iou_threshold:.3f}, CD<{cd_threshold:.1f})")
                continue  # Skip this candidate

            # Composite score - more lenient for early frames
            score = (
                1.2 * iou + 0.6 * sr - 0.0005 * cd + 0.1 * candidate["conf"]
            )  # Reduced CD penalty
            # print(f"Frame {frame_num}: Candidate score={score:.3f} (IoU={iou:.3f}, SR={sr:.3f}, CD={cd:.1f}, conf={candidate['conf']:.3f})")

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # Check if best candidate passes threshold
        if best_candidate and best_score >= detect_params["score_min"]:
            fills.append(
                {
                    "frame": frame_num,
                    "status": "filled",
                    "bbox_xyxy": [round(x, 1) for x in best_candidate["bbox_xyxy"]],
                    "score": round(best_score, 2),
                    "source": "roi_v1_lowconf",
                }
            )
            filled += 1
        else:
            fills.append(
                {"frame": frame_num, "status": "miss", "reason": "no_candidate_passed"}
            )
            missed += 1

    cap.release()

    return {
        "fills": fills,
        "summary": {"attempted": attempted, "filled": filled, "missed": missed},
    }


def find_neighboring_missing_frames(
    track_detections: list[dict], max_frame: int, max_gap: int = 5
) -> list[int]:
    """
    Find missing frames that are close to existing detections (neighboring frames).

    Args:
        track_detections: List of detections for the track
        max_frame: Maximum frame number to consider
        max_gap: Maximum gap from existing detections to consider

    Returns:
        List of missing frame numbers that are close to existing detections
    """
    if not track_detections:
        return []

    # Get all frames where track is detected
    detected_frames = set(d["frame"] for d in track_detections)

    # Find missing frames that are close to existing detections
    neighboring_missing_frames = []

    for detected_frame in detected_frames:
        # Check frames before and after this detection
        for offset in range(1, max_gap + 1):
            # Check frame before
            prev_frame = detected_frame - offset
            if (
                prev_frame >= 0
                and prev_frame not in detected_frames
                and prev_frame not in neighboring_missing_frames
            ):
                neighboring_missing_frames.append(prev_frame)

            # Check frame after
            next_frame = detected_frame + offset
            if (
                next_frame <= max_frame
                and next_frame not in detected_frames
                and next_frame not in neighboring_missing_frames
            ):
                neighboring_missing_frames.append(next_frame)

    return sorted(neighboring_missing_frames)


def calculate_gap_to_anchor(missing_frames: list[int], anchor: dict) -> dict[int, int]:
    """
    Calculate gap distance from each missing frame to the anchor segment.

    Args:
        missing_frames: List of missing frame numbers
        anchor: Anchor segment dictionary

    Returns:
        Dictionary mapping frame numbers to gap distances
    """
    gap_to_anchor = {}
    anchor_start = anchor["start"]
    anchor_end = anchor["end"]

    for frame in missing_frames:
        if frame < anchor_start:
            gap = anchor_start - frame
        elif frame > anchor_end:
            gap = frame - anchor_end
        else:
            gap = (
                0  # Frame is within anchor range (shouldn't happen for missing frames)
            )

        gap_to_anchor[frame] = gap

    return gap_to_anchor


def create_new_detections_from_fills(
    fills: list[dict], track_id: int, fps: float
) -> list[dict]:
    """
    Convert fill results to detection format compatible with detections.jsonl.

    Args:
        fills: List of fill results from redetect_in_rois
        track_id: Track ID for the detections
        fps: Video frame rate

    Returns:
        List of detection dictionaries
    """
    new_detections = []

    for fill in fills:
        if fill["status"] == "filled":
            frame = fill["frame"]
            bbox = fill["bbox_xyxy"]

            detection = {
                "video_id": "happy1.mp4",
                "frame": frame,
                "time": round(frame / fps, 3),
                "track_id": track_id,
                "det_idx": 0,  # Assuming single detection per frame
                "class_id": 2,  # Car class
                "class_name": "car",
                "conf": 0.15,  # Low confidence as per ROI detection
                "bbox_xyxy": bbox,
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
            }
            new_detections.append(detection)

    return new_detections


def propagate_bbox_with_optical_flow(
    video_path: str,
    prev_frame_num: int,
    curr_frame_num: int,
    prev_bbox: list[float],
    reference_bbox: list[float] | None = None,
    reference_frame: int | None = None,
    motion_params: dict | None = None,
) -> list[float] | None:
    """
    Propagate a bounding box from previous frame to current frame using optical flow.
    Now includes scale estimation to handle size changes.

    Args:
        video_path: Path to video file
        prev_frame_num: Previous frame number (where bbox is known)
        curr_frame_num: Current frame number (where we want to predict bbox)
        prev_bbox: Previous bounding box [x1, y1, x2, y2]
        reference_bbox: Optional reference bbox for size interpolation
        reference_frame: Optional reference frame number
        motion_params: Optional motion model parameters with dw, dh for size prediction

    Returns:
        Predicted bbox [x1, y1, x2, y2] or None if propagation fails
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        # Read previous frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_num)
        ret1, frame1 = cap.read()
        if not ret1:
            return None

        # Read current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_num)
        ret2, frame2 = cap.read()
        if not ret2:
            return None

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Extract points from previous bbox
        x1, y1, x2, y2 = prev_bbox

        # Create a denser grid of points, including corners and edges
        grid_size = 15
        points = []

        # Add corner points (important for scale estimation)
        corners = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],  # corners
            [x1, (y1 + y2) / 2],
            [x2, (y1 + y2) / 2],  # left/right midpoints
            [(x1 + x2) / 2, y1],
            [(x1 + x2) / 2, y2],  # top/bottom midpoints
        ]
        for corner in corners:
            points.append([corner])

        # Add grid points
        for i in np.linspace(x1, x2, grid_size):
            for j in np.linspace(y1, y2, grid_size):
                points.append([[i, j]])

        if not points:
            return None

        p0 = np.array(points, dtype=np.float32)

        # Calculate optical flow using Lucas-Kanade with larger window
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            p0,
            None,  # type: ignore
            winSize=(21, 21),  # Increased from 15
            maxLevel=3,  # Increased pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        if p1 is None or status is None:
            return None

        # Filter good points
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        if len(good_new) < 4:  # Need at least 4 points
            return None

        # Calculate displacement for translation
        displacement = good_new - good_old
        median_dx = np.median(displacement[:, 0])
        median_dy = np.median(displacement[:, 1])

        # Estimate scale change from optical flow
        # Calculate spread of points before and after
        old_center = np.mean(good_old, axis=0)
        new_center = np.mean(good_new, axis=0)

        old_distances = np.sqrt(np.sum((good_old - old_center) ** 2, axis=1))
        new_distances = np.sqrt(np.sum((good_new - new_center) ** 2, axis=1))

        # Filter outliers for scale estimation
        old_dist_median = np.median(old_distances)
        new_dist_median = np.median(new_distances)

        # Calculate scale ratio, but be conservative (don't let it shrink too much)
        if old_dist_median > 1:
            scale_ratio = new_dist_median / old_dist_median
            # Clamp scale ratio to reasonable range [0.8, 1.4]
            scale_ratio = np.clip(scale_ratio, 0.8, 1.4)
        else:
            scale_ratio = 1.0

        # Determine bbox size using multiple strategies
        prev_w = x2 - x1
        prev_h = y2 - y1

        # Strategy 1: Use reference frame interpolation if available
        if reference_bbox is not None and reference_frame is not None:
            ref_w = reference_bbox[2] - reference_bbox[0]
            ref_h = reference_bbox[3] - reference_bbox[1]

            # Calculate interpolation factor
            total_gap = abs(reference_frame - prev_frame_num)
            current_gap = abs(curr_frame_num - prev_frame_num)

            if total_gap > 0:
                alpha = current_gap / total_gap
                # Interpolate dimensions
                w = prev_w + alpha * (ref_w - prev_w)
                h = prev_h + alpha * (ref_h - prev_h)
            else:
                w = prev_w * scale_ratio
                h = prev_h * scale_ratio
        # Strategy 2: Use optical flow scale estimation with enhancement
        else:
            # Use optical flow scale but enhance it slightly
            # Since optical flow often underestimates box size, we apply a minimum scale
            enhanced_scale = max(scale_ratio, 1.1)  # At least 10% growth
            w = prev_w * enhanced_scale
            h = prev_h * enhanced_scale

        # Calculate new center
        center_x = (x1 + x2) / 2 + median_dx
        center_y = (y1 + y2) / 2 + median_dy

        # Create new bbox from center and scaled dimensions
        new_x1 = center_x - w / 2
        new_y1 = center_y - h / 2
        new_x2 = center_x + w / 2
        new_y2 = center_y + h / 2

        return [float(new_x1), float(new_y1), float(new_x2), float(new_y2)]

    finally:
        cap.release()


def try_detect_single_frame(
    video_path: str,
    frame_num: int,
    pred_bbox: list[float],
    detector_handle: Any,
    detect_params: dict[str, Any],
    class_id: int,
    img_size: dict[str, int],
    roi_scale: float = 2.0,
    aoi_polygon: list[list[float]] | None = None,
) -> dict | None:
    """
    Try to detect object in a single frame using ROI-based detection.

    Returns:
        Detection dict with bbox_xyxy, conf, score if successful, None otherwise
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return None

        # Create ROI from predicted bbox
        x1, y1, x2, y2 = pred_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Expand by scale
        new_w = w * roi_scale
        new_h = h * roi_scale

        roi_x1 = max(0, int(center_x - new_w / 2))
        roi_y1 = max(0, int(center_y - new_h / 2))
        roi_x2 = min(img_size["width"], int(center_x + new_w / 2))
        roi_y2 = min(img_size["height"], int(center_y + new_h / 2))

        roi_xyxy = [roi_x1, roi_y1, roi_x2, roi_y2]

        # Crop ROI
        roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi_crop.size == 0:
            return None

        # Run detection
        results = detector_handle(
            roi_crop,
            conf=detect_params["conf"],
            iou=detect_params["iou"],
            classes=[class_id],
            verbose=False,
        )[0]

        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)

        # Find candidates
        candidates = []
        for i in range(len(detections)):
            if detections.class_id is not None and detections.class_id[i] == class_id:
                crop_bbox = detections.xyxy[i].tolist()
                full_bbox = map_crop_to_fullframe(crop_bbox, roi_xyxy)
                conf = (
                    detections.confidence[i]
                    if detections.confidence is not None
                    else 0.0
                )
                candidates.append(
                    {"bbox_xyxy": full_bbox, "class_id": class_id, "conf": conf}
                )

        # Filter by AOI
        if aoi_polygon and candidates:
            filtered = []
            for det in candidates:
                cx = (det["bbox_xyxy"][0] + det["bbox_xyxy"][2]) / 2
                cy = (det["bbox_xyxy"][1] + det["bbox_xyxy"][3]) / 2
                if point_in_polygon((cx, cy), aoi_polygon):
                    filtered.append(det)
            candidates = filtered

        if not candidates:
            return None

        # Score candidates
        best_candidate = None
        best_score = -1

        for candidate in candidates:
            det_bbox = candidate["bbox_xyxy"]
            iou = calculate_iou(det_bbox, pred_bbox)
            cd = calculate_center_distance(det_bbox, pred_bbox)
            sr = calculate_size_ratio(det_bbox, pred_bbox)

            # Lenient thresholds
            score = 1.2 * iou + 0.6 * sr - 0.0005 * cd + 0.1 * candidate["conf"]

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # Check threshold
        if best_candidate and best_score >= detect_params.get("score_min", 0.01):
            return {
                "bbox_xyxy": best_candidate["bbox_xyxy"],
                "conf": best_candidate["conf"],
                "score": best_score,
            }

        return None

    finally:
        cap.release()


def backfill_track_sequential(
    track_id: int,
    detections: list[dict[str, Any]],
    video_path: str,
    max_frame: int,
    detector_handle: Any = None,
    img_size: dict[str, int] = {"width": 1280, "height": 720},
    detect_params: dict[str, Any] | None = None,
    class_id: int = 2,
    aoi_polygon: list[list[float]] | None = None,
    optical_flow_gap_threshold: int = 5,
    roi_scale: float = 2.0,
) -> dict[str, Any]:
    """
    Backfill missing detections for a track sequentially from anchor frame.

    This is the main workflow function that:
    1. Finds anchor segment (longest, most confident segment)
    2. Fits motion model on anchor
    3. Walks backward and forward from anchor, frame by frame
    4. For each missing frame:
       - Try ROI-based detection with motion model prediction
       - If detection fails and gap ≤ threshold, use optical flow propagation
    5. Returns new detections to merge with original

    Args:
        track_id: Track ID to backfill
        detections: List of all detections
        video_path: Path to video file
        max_frame: Maximum frame number in video
        detector_handle: YOLO detector instance (if None, will load default)
        img_size: Video frame dimensions
        detect_params: Detection parameters (conf, iou, score_min)
        class_id: Class ID for detections
        aoi_polygon: Optional area of interest polygon
        optical_flow_gap_threshold: Max gap for optical flow fallback
        roi_scale: Scale factor for ROI expansion

    Returns:
        Dictionary with:
            - new_detections: List of newly created detections
            - summary: Statistics about fills
            - fills: Detailed fill information per frame
    """
    # Set defaults
    if detect_params is None:
        detect_params = {"conf": 0.05, "iou": 0.25, "score_min": 0.01}

    if detector_handle is None:
        detector_handle = YOLO("yolov8s.pt")

    # Get track detections
    track_detections = get_track_to_detections(detections, [track_id]).get(track_id, [])

    if not track_detections:
        return {
            "new_detections": [],
            "summary": {"attempted": 0, "filled": 0, "missed": 0},
            "fills": [],
        }

    print(f"Track {track_id} has {len(track_detections)} existing detections")

    # Build segments and pick anchor
    segments = build_segments(track_detections)
    anchor = pick_anchor(segments)

    if not anchor:
        print("No valid anchor found")
        return {
            "new_detections": [],
            "summary": {"attempted": 0, "filled": 0, "missed": 0},
            "fills": [],
        }

    print(f"Anchor: frames {anchor['start']}-{anchor['end']} (length {anchor['len']})")

    # Fit motion model
    motion_model = fit_motion_model(track_detections, anchor)
    print(
        f"Motion model: vx={motion_model['params']['vx']:.2f}, vy={motion_model['params']['vy']:.2f}"
    )

    # Build frame-to-detection map
    detected_frames = set(d["frame"] for d in track_detections)
    frame_to_bbox = {d["frame"]: d["bbox_xyxy"] for d in track_detections}

    # Prepare to walk from anchor
    fills = []
    attempted = 0
    filled = 0
    missed = 0

    # Walk backward from anchor
    print(f"\nWalking backward from frame {anchor['start']}")
    current_frame = anchor["start"] - 1
    last_good_frame = anchor["start"]
    consecutive_misses = 0

    while current_frame >= 0:
        if current_frame in detected_frames:
            # Frame already has detection, update last good frame
            last_good_frame = current_frame
            consecutive_misses = 0
            current_frame -= 1
            continue

        attempted += 1
        gap_to_last_good = last_good_frame - current_frame

        # Try detection first using motion model prediction
        pred_bbox = predict_bbox(motion_model["params"], current_frame)
        detection_result = try_detect_single_frame(
            video_path,
            current_frame,
            pred_bbox,
            detector_handle,
            detect_params,
            class_id,
            img_size,
            roi_scale,
            aoi_polygon,
        )

        if detection_result:
            # Detection successful
            fills.append(
                {
                    "frame": current_frame,
                    "status": "filled",
                    "bbox_xyxy": detection_result["bbox_xyxy"],
                    "source": "detection",
                    "score": detection_result["score"],
                }
            )
            filled += 1
            frame_to_bbox[current_frame] = detection_result["bbox_xyxy"]
            last_good_frame = current_frame
            consecutive_misses = 0
        elif gap_to_last_good <= optical_flow_gap_threshold:
            # Try optical flow propagation
            prev_bbox = frame_to_bbox.get(last_good_frame)
            if prev_bbox:
                # When walking backward, find a reference frame before current_frame
                # to interpolate box sizes (vehicles typically get larger as they approach)
                reference_bbox = None
                reference_frame = None
                for ref_frame in sorted(detected_frames, reverse=True):
                    if ref_frame < current_frame:
                        reference_bbox = frame_to_bbox.get(ref_frame)
                        reference_frame = ref_frame
                        break

                flow_bbox = propagate_bbox_with_optical_flow(
                    video_path,
                    last_good_frame,
                    current_frame,
                    prev_bbox,
                    reference_bbox=reference_bbox,
                    reference_frame=reference_frame,
                    motion_params=motion_model["params"],
                )
                if flow_bbox:
                    fills.append(
                        {
                            "frame": current_frame,
                            "status": "filled",
                            "bbox_xyxy": flow_bbox,
                            "source": "optical_flow",
                            "score": 0.0,
                        }
                    )
                    filled += 1
                    frame_to_bbox[current_frame] = flow_bbox
                    last_good_frame = current_frame
                    consecutive_misses = 0
                else:
                    fills.append(
                        {
                            "frame": current_frame,
                            "status": "miss",
                            "reason": "optical_flow_failed",
                        }
                    )
                    missed += 1
                    consecutive_misses += 1
            else:
                fills.append(
                    {"frame": current_frame, "status": "miss", "reason": "no_prev_bbox"}
                )
                missed += 1
                consecutive_misses += 1
        else:
            fills.append(
                {"frame": current_frame, "status": "miss", "reason": "gap_too_large"}
            )
            missed += 1
            consecutive_misses += 1

        # Stop if too many consecutive misses
        if consecutive_misses > 10:
            print(f"Stopped at frame {current_frame} (10 consecutive misses)")
            break

        current_frame -= 1

    # Walk forward from anchor
    print(f"\nWalking forward from frame {anchor['end']}")
    current_frame = anchor["end"] + 1
    last_good_frame = anchor["end"]
    consecutive_misses = 0

    while current_frame <= max_frame:
        if current_frame in detected_frames:
            last_good_frame = current_frame
            consecutive_misses = 0
            current_frame += 1
            continue

        attempted += 1
        gap_to_last_good = current_frame - last_good_frame

        # Try detection first
        pred_bbox = predict_bbox(motion_model["params"], current_frame)
        detection_result = try_detect_single_frame(
            video_path,
            current_frame,
            pred_bbox,
            detector_handle,
            detect_params,
            class_id,
            img_size,
            roi_scale,
            aoi_polygon,
        )

        if detection_result:
            fills.append(
                {
                    "frame": current_frame,
                    "status": "filled",
                    "bbox_xyxy": detection_result["bbox_xyxy"],
                    "source": "detection",
                    "score": detection_result["score"],
                }
            )
            filled += 1
            frame_to_bbox[current_frame] = detection_result["bbox_xyxy"]
            last_good_frame = current_frame
            consecutive_misses = 0
        elif gap_to_last_good <= optical_flow_gap_threshold:
            prev_bbox = frame_to_bbox.get(last_good_frame)
            if prev_bbox:
                # Find next reference frame (frame with detection after current)
                reference_bbox = None
                reference_frame = None
                for ref_frame in sorted(detected_frames, reverse=True):
                    if ref_frame > current_frame:
                        reference_bbox = frame_to_bbox.get(ref_frame)
                        reference_frame = ref_frame
                        break

                flow_bbox = propagate_bbox_with_optical_flow(
                    video_path,
                    last_good_frame,
                    current_frame,
                    prev_bbox,
                    reference_bbox=reference_bbox,
                    reference_frame=reference_frame,
                    motion_params=motion_model["params"],
                )
                if flow_bbox:
                    fills.append(
                        {
                            "frame": current_frame,
                            "status": "filled",
                            "bbox_xyxy": flow_bbox,
                            "source": "optical_flow",
                            "score": 0.0,
                        }
                    )
                    filled += 1
                    frame_to_bbox[current_frame] = flow_bbox
                    last_good_frame = current_frame
                    consecutive_misses = 0
                else:
                    fills.append(
                        {
                            "frame": current_frame,
                            "status": "miss",
                            "reason": "optical_flow_failed",
                        }
                    )
                    missed += 1
                    consecutive_misses += 1
            else:
                fills.append(
                    {"frame": current_frame, "status": "miss", "reason": "no_prev_bbox"}
                )
                missed += 1
                consecutive_misses += 1
        else:
            fills.append(
                {"frame": current_frame, "status": "miss", "reason": "gap_too_large"}
            )
            missed += 1
            consecutive_misses += 1

        # Stop if too many consecutive misses
        if consecutive_misses > 10:
            print(f"Stopped at frame {current_frame} (10 consecutive misses)")
            break

        current_frame += 1

    # Create new detections from fills
    fps = 30.0  # Default FPS
    new_detections = []

    for fill in fills:
        if fill["status"] == "filled":
            frame = fill["frame"]
            bbox = fill["bbox_xyxy"]

            detection = {
                "video_id": Path(video_path).name,
                "frame": frame,
                "time": round(frame / fps, 3),
                "track_id": track_id,
                "det_idx": 0,
                "class_id": class_id,
                "class_name": "car",
                "conf": 0.15,
                "bbox_xyxy": bbox,
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "source": fill["source"],
            }
            new_detections.append(detection)

    summary = {
        "attempted": attempted,
        "filled": filled,
        "missed": missed,
        "detection_fills": sum(1 for f in fills if f.get("source") == "detection"),
        "optical_flow_fills": sum(
            1 for f in fills if f.get("source") == "optical_flow"
        ),
    }

    print(f"\nSummary: {summary}")

    return {"new_detections": new_detections, "summary": summary, "fills": fills}


if __name__ == "__main__":
    # Load real detections data
    detections_path = Path(
        "/Users/shuaima/code/accident_analysis/accident-analysis-hackathon/out/test_37/detections.jsonl"
    )
    detections = read_detections_from_jsonl(detections_path)

    # Configuration
    track_id = 17
    max_frame = 96
    video_path = "happy1.mp4"

    print(f"=== Backfilling Track {track_id} ===")
    print(f"Total detections loaded: {len(detections)}")
    print(f"Video: {video_path}")
    print(f"Max frame: {max_frame}\n")

    # Use the new unified workflow function
    result = backfill_track_sequential(
        track_id=track_id,
        detections=detections,
        video_path=video_path,
        max_frame=max_frame,
        detector_handle=None,  # Will load default YOLO model
        img_size={"width": 1280, "height": 720},
        detect_params={"conf": 0.05, "iou": 0.25, "score_min": 0.01},
        class_id=2,
        aoi_polygon=None,
        optical_flow_gap_threshold=5,
        roi_scale=2.0,
    )

    new_detections = result["new_detections"]
    summary = result["summary"]

    print("\n=== Results ===")
    print(f"Attempted: {summary['attempted']}")
    print(f"Filled: {summary['filled']}")
    print(f"  - Detection fills: {summary['detection_fills']}")
    print(f"  - Optical flow fills: {summary['optical_flow_fills']}")
    print(f"Missed: {summary['missed']}")

    # Save merged detections to file
    output_path = Path(
        "/Users/shuaima/code/accident_analysis/accident-analysis-hackathon/out/test_37/detections-redone.jsonl"
    )

    # Merge with original detections
    all_detections = detections + new_detections
    all_detections.sort(key=lambda x: x["frame"])

    with open(output_path, "w", encoding="utf-8") as f:
        for detection in all_detections:
            f.write(json.dumps(detection) + "\n")

    print("\n=== Output ===")
    print(f"Saved {len(all_detections)} total detections to {output_path}")
    print(f"Added {len(new_detections)} new detections for track {track_id}")

    # Print sample of new detections
    if new_detections:
        print("\nSample of new detections:")
        for det in new_detections[:10]:
            print(
                f"  Frame {det['frame']}: bbox {[round(x, 1) for x in det['bbox_xyxy']]} (source: {det.get('source', 'unknown')})"
            )
