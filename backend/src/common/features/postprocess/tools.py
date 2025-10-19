"""
Postprocessing tools for accident analysis.
These tools are designed to be called by LLM agents for analyzing vehicle collision data.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def calculate_compass_direction(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> str:
    """
    Calculate compass direction from lat/lon coordinates.

    Args:
        lat1, lon1: Starting coordinates
        lat2, lon2: Ending coordinates

    Returns:
        Compass direction (e.g., "North", "Northeast", "East", etc.)
    """
    # Calculate bearing in degrees
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)

    # Normalize to 0-360 degrees
    bearing_deg = (bearing_deg + 360) % 360

    # Convert to compass direction
    directions = [
        (0, 11.25, "North"),
        (11.25, 33.75, "Northeast"),
        (33.75, 56.25, "Northeast"),
        (56.25, 78.75, "East"),
        (78.75, 101.25, "East"),
        (101.25, 123.75, "Southeast"),
        (123.75, 146.25, "Southeast"),
        (146.25, 168.75, "South"),
        (168.75, 191.25, "South"),
        (191.25, 213.75, "Southwest"),
        (213.75, 236.25, "Southwest"),
        (236.25, 258.75, "West"),
        (258.75, 281.25, "West"),
        (281.25, 303.75, "Northwest"),
        (303.75, 326.25, "Northwest"),
        (326.25, 348.75, "North"),
        (348.75, 360, "North"),
    ]

    for start, end, direction in directions:
        if start <= bearing_deg < end:
            return direction

    return "North"  # Default fallback


def calculate_movement_description(
    lat1: float, lon1: float, lat2: float, lon2: float, speed_mph: float | None = None
) -> str:
    """
    Generate a human-readable description of vehicle movement.

    Args:
        lat1, lon1: Starting coordinates
        lat2, lon2: Ending coordinates
        speed_mph: Speed in miles per hour

    Returns:
        Description like "moving Northeast at 25 mph" or "traveling South"
    """
    direction = calculate_compass_direction(lat1, lon1, lat2, lon2)

    if speed_mph is not None and speed_mph > 0:
        return f"moving {direction} at {speed_mph:.1f} mph"
    else:
        return f"traveling {direction}"


@dataclass
class Detection:
    """Represents a single detection record."""

    video_id: str
    frame: int
    time: float
    track_id: int
    det_idx: int
    class_id: int
    class_name: str
    conf: float
    bbox_xyxy: list[float]
    center: list[float]
    speed_mph: float | None
    world_coords: list[float]
    tracking_point: str
    raw_bbox: list[float]
    event_time_real: str | None = None  # Optional field for timestamp data


@dataclass
class MetricRow:
    """Represents enriched detection data with computed metrics."""

    frame: int
    timestamp: float
    track_ids: list[int]
    bbox_xyxy: list[list[float]]
    world_coords: list[list[float]]
    speed_mph: list[float | None]
    iou: float
    pixel_center_distance_px: float
    world_distance_m: float | None
    relative_speed_mps: float | None
    heading_diff_deg: float | None
    iou_exceeds_threshold: bool
    distance_below_threshold: bool
    collision_candidate: bool
    metadata: dict[str, Any]
    # New direction fields
    vehicle_directions: list[str] | None = None
    movement_descriptions: list[str] | None = None


def load_detections(
    track_ids: list[int],
    frame_range: list[int] | None = None,
    fields: list[str] | None = None,
    require_pairing: bool = True,
    fps_hint: float | None = None,
    detections_file: str = "backend/src/common/features/process_video/detections.jsonl",
) -> dict[str, Any]:
    """
    Load detection data for specified track IDs.

    Args:
        track_ids: List of track IDs to load (e.g., [7, 14])
        frame_range: Optional [start_frame, end_frame] list to restrict loading
        fields: Optional list of fields to include (controls payload size)
        require_pairing: If True, only return frames where all track_ids appear
        fps_hint: Optional FPS hint for metadata
        detections_file: Path to the detections.jsonl file

    Returns:
        List of paired detection records sorted by frame with metadata
    """
    detections_path = Path(detections_file)
    if not detections_path.exists():
        raise FileNotFoundError(f"Detections file not found: {detections_file}")

    # Load all detections
    all_detections = []
    with open(detections_path) as f:
        for line in f:
            if line.strip():
                detection_data = json.loads(line.strip())
                all_detections.append(Detection(**detection_data))

    # Filter by track IDs
    filtered_detections = [d for d in all_detections if d.track_id in track_ids]

    # Filter by frame range if specified
    if frame_range:
        start_frame, end_frame = frame_range
        filtered_detections = [
            d for d in filtered_detections if start_frame <= d.frame <= end_frame
        ]

    # Group by frame
    frame_groups = {}
    for detection in filtered_detections:
        if detection.frame not in frame_groups:
            frame_groups[detection.frame] = []
        frame_groups[detection.frame].append(detection)

    # Create paired records
    paired_records = []
    for frame in sorted(frame_groups.keys()):
        frame_detections = frame_groups[frame]

        if require_pairing and len(frame_detections) != len(track_ids):
            continue  # Skip frames where not all tracks appear

        # Create a record for this frame
        record = {
            "frame": frame,
            "timestamp": frame_detections[0].time,
            "detections": {},
        }

        for detection in frame_detections:
            detection_dict = {
                "track_id": detection.track_id,
                "bbox_xyxy": detection.bbox_xyxy,
                "center": detection.center,
                "speed_mph": detection.speed_mph,
                "world_coords": detection.world_coords,
                "conf": detection.conf,
                "class_name": detection.class_name,
            }

            # Filter fields if specified
            if fields:
                detection_dict = {
                    k: v for k, v in detection_dict.items() if k in fields
                }

            record["detections"][detection.track_id] = detection_dict

        paired_records.append(record)

    # Add metadata
    if paired_records:
        frames = [r["frame"] for r in paired_records]
        timestamps = [r["timestamp"] for r in paired_records]

        metadata = {
            "total_frames": len(paired_records),
            "frame_range": [min(frames), max(frames)] if frames else None,
            "time_range": [min(timestamps), max(timestamps)] if timestamps else None,
            "track_ids": track_ids,
            "fps_estimated": fps_hint
            or (
                len(timestamps) / (max(timestamps) - min(timestamps))
                if len(timestamps) > 1
                else None
            ),
            "missing_frames": _find_missing_frames(frames),
            "data_gaps": _analyze_data_gaps(paired_records),
        }

        return {"records": paired_records, "metadata": metadata}

    return {"records": [], "metadata": {}}


def _find_missing_frames(frames: list[int]) -> list[int]:
    """Find missing frame numbers in a sequence."""
    if not frames:
        return []

    min_frame, max_frame = min(frames), max(frames)
    expected_frames = set(range(min_frame, max_frame + 1))
    actual_frames = set(frames)
    return sorted(list(expected_frames - actual_frames))


def _analyze_data_gaps(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze data quality issues."""
    gaps = {
        "missing_speeds": 0,
        "missing_world_coords": 0,
        "low_confidence": 0,
        "total_detections": 0,
    }

    for record in records:
        for track_id, detection in record["detections"].items():
            gaps["total_detections"] += 1
            if detection.get("speed_mph") is None:
                gaps["missing_speeds"] += 1
            if detection.get("world_coords") is None:
                gaps["missing_world_coords"] += 1
            if detection.get("conf", 1.0) < 0.5:
                gaps["low_confidence"] += 1

    return gaps


def compute_pair_metrics(
    pairs: list[dict[str, Any]],
    iou_threshold: float = 0.01,
    vehicle_length_m: float = 4.5,
    include_headings: bool = True,
) -> list[MetricRow]:
    """
    Compute metrics for paired detections.

    Args:
        pairs: List of paired detection records (from load_detections)
        iou_threshold: IoU threshold for collision detection
        vehicle_length_m: Assumed vehicle length for distance calculations
        include_headings: Whether to compute heading differences

    Returns:
        List of MetricRow objects with enriched data
    """
    metric_rows = []

    for record in pairs:
        detections = record["detections"]
        track_ids = list(detections.keys())

        if len(track_ids) != 2:
            continue  # Skip if not exactly 2 tracks

        track1_id, track2_id = track_ids
        det1 = detections[track1_id]
        det2 = detections[track2_id]

        # Compute IoU
        iou = _compute_iou(det1["bbox_xyxy"], det2["bbox_xyxy"])

        # Compute pixel distance between centers
        center1 = det1["center"]
        center2 = det2["center"]
        pixel_distance = math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

        # Compute world distance
        world_distance = None
        if det1.get("world_coords") and det2.get("world_coords"):
            world_distance = _compute_world_distance(
                det1["world_coords"], det2["world_coords"]
            )

        # Compute relative speed
        relative_speed = None
        if det1.get("speed_mph") is not None and det2.get("speed_mph") is not None:
            relative_speed = (
                abs(det1["speed_mph"] - det2["speed_mph"]) * 0.44704
            )  # Convert mph to m/s

        # Compute heading difference (if requested)
        heading_diff = None
        if include_headings:
            # This would require heading data which isn't in the current format
            # For now, we'll skip this calculation
            pass

        # Calculate vehicle directions and movement descriptions
        vehicle_directions = []
        movement_descriptions = []

        if include_headings:
            for det in [det1, det2]:
                if det.get("world_coords") and len(det["world_coords"]) >= 2:
                    # For now, we'll use a simple approach - in a real implementation,
                    # you'd want to calculate direction from previous frame
                    # This is a placeholder that shows the concept
                    lat, lon = det["world_coords"]
                    # Use a small offset to simulate movement direction
                    # In practice, you'd compare with previous frame
                    direction = calculate_compass_direction(
                        lat, lon, lat + 0.0001, lon + 0.0001
                    )
                    vehicle_directions.append(direction)

                    speed = det.get("speed_mph")
                    movement_desc = calculate_movement_description(
                        lat, lon, lat + 0.0001, lon + 0.0001, speed
                    )
                    movement_descriptions.append(movement_desc)
                else:
                    vehicle_directions.append("Unknown")
                    movement_descriptions.append("Unknown direction")
        else:
            vehicle_directions = None
            movement_descriptions = None

        # Determine flags
        iou_exceeds_threshold = iou > iou_threshold
        distance_below_threshold = world_distance is not None and world_distance < 5.0
        collision_candidate = iou_exceeds_threshold or distance_below_threshold

        metric_row = MetricRow(
            frame=record["frame"],
            timestamp=record["timestamp"],
            track_ids=[track1_id, track2_id],
            bbox_xyxy=[det1["bbox_xyxy"], det2["bbox_xyxy"]],
            world_coords=[det1.get("world_coords", []), det2.get("world_coords", [])],
            speed_mph=[det1.get("speed_mph"), det2.get("speed_mph")],
            iou=iou,
            pixel_center_distance_px=pixel_distance,
            world_distance_m=world_distance,
            relative_speed_mps=relative_speed,
            heading_diff_deg=heading_diff,
            iou_exceeds_threshold=iou_exceeds_threshold,
            distance_below_threshold=distance_below_threshold,
            collision_candidate=collision_candidate,
            metadata={
                "confidences": [det1.get("conf", 0), det2.get("conf", 0)],
                "class_names": [det1.get("class_name", ""), det2.get("class_name", "")],
            },
            vehicle_directions=vehicle_directions,
            movement_descriptions=movement_descriptions,
        )

        metric_rows.append(metric_row)

    return metric_rows


def _compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _compute_world_distance(coord1: list[float], coord2: list[float]) -> float:
    """Compute great-circle distance between two world coordinates."""
    # Using Haversine formula for great-circle distance
    lat1, lon1 = math.radians(coord1[1]), math.radians(coord1[0])
    lat2, lon2 = math.radians(coord2[1]), math.radians(coord2[0])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Earth's radius in meters
    r = 6371000
    return r * c


def trace_impact_window(
    metric_rows: list[MetricRow],
    iou_threshold: float = 0.01,
    distance_threshold_m: float = 5.0,
    persistence_frames: int = 3,
) -> dict[str, Any]:
    """
    Trace the impact window from metric rows.

    Args:
        metric_rows: List of MetricRow objects
        iou_threshold: IoU threshold for collision detection
        distance_threshold_m: Distance threshold in meters
        persistence_frames: Number of frames to persist overlap

    Returns:
        Dictionary with impact analysis results
    """
    if not metric_rows:
        return {
            "collision_detected": False,
            "impact_frames": [],
            "closest_approach": None,
            "overlap_duration_frames": 0,
            "diagnostic_notes": ["No metric data available"],
        }

    # Find first qualifying frame
    first_contact_frame = None
    last_overlap_frame = None
    min_distance_frame = None
    min_distance = float("inf")

    qualifying_frames = []

    for i, row in enumerate(metric_rows):
        qualifies = False
        reasons = []

        if row.iou > iou_threshold:
            qualifies = True
            reasons.append(f"IoU {row.iou:.3f} > {iou_threshold}")

        if (
            row.world_distance_m is not None
            and row.world_distance_m < distance_threshold_m
        ):
            qualifies = True
            reasons.append(
                f"Distance {row.world_distance_m:.1f}m < {distance_threshold_m}m"
            )

        if qualifies:
            qualifying_frames.append(
                {
                    "frame": row.frame,
                    "timestamp": row.timestamp,
                    "iou": row.iou,
                    "world_distance_m": row.world_distance_m,
                    "reasons": reasons,
                }
            )

            if first_contact_frame is None:
                first_contact_frame = row.frame

            last_overlap_frame = row.frame

        # Track minimum distance
        if row.world_distance_m is not None and row.world_distance_m < min_distance:
            min_distance = row.world_distance_m
            min_distance_frame = row.frame

    # Determine collision detection
    collision_detected = len(qualifying_frames) >= persistence_frames

    # Calculate overlap duration
    overlap_duration_frames = 0
    if first_contact_frame is not None and last_overlap_frame is not None:
        overlap_duration_frames = last_overlap_frame - first_contact_frame + 1

    # Generate diagnostic notes
    diagnostic_notes = []
    if not collision_detected:
        if len(qualifying_frames) == 0:
            diagnostic_notes.append("No frames met collision criteria")
        else:
            diagnostic_notes.append(
                f"Only {len(qualifying_frames)} frames met criteria (need {persistence_frames})"
            )

    if qualifying_frames:
        iou_max = max(f["iou"] for f in qualifying_frames)
        diagnostic_notes.append(f"Maximum IoU: {iou_max:.3f}")

        distances = [
            f["world_distance_m"]
            for f in qualifying_frames
            if f["world_distance_m"] is not None
        ]
        if distances:
            min_dist = min(distances)
            diagnostic_notes.append(f"Minimum distance: {min_dist:.1f}m")

    return {
        "collision_detected": collision_detected,
        "impact_frames": qualifying_frames,
        "closest_approach": {
            "frame": min_distance_frame,
            "distance_m": min_distance if min_distance != float("inf") else None,
        },
        "overlap_duration_frames": overlap_duration_frames,
        "diagnostic_notes": diagnostic_notes,
        "first_contact_frame": first_contact_frame,
        "last_overlap_frame": last_overlap_frame,
    }


def build_timeline(
    metric_rows: list[MetricRow],
    impact_summary: dict[str, Any],
    padding_frames: int = 10,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a structured timeline from metric rows and impact summary.

    Args:
        metric_rows: List of MetricRow objects
        impact_summary: Output from trace_impact_window
        padding_frames: Number of frames to include before/after impact
        stages: Custom stage labels (defaults to approach, first_contact, peak_overlap, separation)

    Returns:
        Dictionary with timeline entries and summary table
    """
    if not metric_rows:
        return {
            "timeline": [],
            "summary_table": [],
            "stages": stages
            or ["approach", "first_contact", "peak_overlap", "separation"],
        }

    default_stages = ["approach", "first_contact", "peak_overlap", "separation"]
    stage_labels = stages or default_stages

    # Determine frame range
    all_frames = [row.frame for row in metric_rows]
    min_frame = min(all_frames)
    max_frame = max(all_frames)

    # Add padding
    start_frame = max(min_frame - padding_frames, min_frame)
    end_frame = min(max_frame + padding_frames, max_frame)

    # Create timeline entries
    timeline = []
    summary_data = []

    # Find key frames
    first_contact_frame = impact_summary.get("first_contact_frame")
    last_overlap_frame = impact_summary.get("last_overlap_frame")
    closest_approach_frame = impact_summary.get("closest_approach", {}).get("frame")

    # Find peak overlap (highest IoU)
    peak_iou_frame = None
    peak_iou = 0
    for row in metric_rows:
        if row.iou > peak_iou:
            peak_iou = row.iou
            peak_iou_frame = row.frame

    # Create timeline entries for each stage
    stage_frames = {
        "approach": start_frame,
        "first_contact": first_contact_frame,
        "peak_overlap": peak_iou_frame,
        "separation": last_overlap_frame,
    }

    for stage in stage_labels:
        frame = stage_frames.get(stage)
        if frame is not None:
            # Find the metric row for this frame
            row = next((r for r in metric_rows if r.frame == frame), None)
            if row:
                entry = {
                    "stage": stage,
                    "frame": frame,
                    "timestamp": row.timestamp,
                    "metrics": {
                        "iou": row.iou,
                        "world_distance_m": row.world_distance_m,
                        "pixel_distance_px": row.pixel_center_distance_px,
                        "relative_speed_mps": row.relative_speed_mps,
                    },
                    "directions": {
                        "vehicle_directions": row.vehicle_directions,
                        "movement_descriptions": row.movement_descriptions,
                    },
                    "narrative": _generate_narrative(stage, row, impact_summary),
                }
                timeline.append(entry)

                # Add to summary table
                summary_data.append(
                    {
                        "Stage": stage,
                        "Frame": frame,
                        "Time (s)": f"{row.timestamp:.3f}",
                        "IoU": f"{row.iou:.3f}",
                        "Distance (miles)": f"{row.world_distance_m * 0.000621371:.3f}"
                        if row.world_distance_m
                        else "N/A",
                        "Speed Diff (mph)": f"{row.relative_speed_mps * 2.237:.1f}"
                        if row.relative_speed_mps
                        else "N/A",
                        "Directions": f"{row.vehicle_directions[0] if row.vehicle_directions else 'Unknown'} vs {row.vehicle_directions[1] if row.vehicle_directions and len(row.vehicle_directions) > 1 else 'Unknown'}",
                    }
                )

    return {
        "timeline": timeline,
        "summary_table": summary_data,
        "stages": stage_labels,
        "frame_range": [start_frame, end_frame],
        "total_frames": len(metric_rows),
    }


def _generate_narrative(
    stage: str, row: MetricRow, impact_summary: dict[str, Any]
) -> str:
    """Generate narrative text for a timeline stage with everyday language and directions."""
    timestamp_str = f"{row.timestamp:.3f}s"

    # Get vehicle information
    track_ids = row.track_ids
    directions = row.vehicle_directions or ["Unknown", "Unknown"]
    movement_descs = row.movement_descriptions or ["Unknown", "Unknown"]

    if stage == "approach":
        distance_desc = (
            f"{row.world_distance_m:.1f} miles apart"
            if row.world_distance_m
            else "approaching"
        )
        vehicle1_desc = (
            f"Vehicle {track_ids[0]} ({movement_descs[0]})"
            if movement_descs[0] != "Unknown"
            else f"Vehicle {track_ids[0]}"
        )
        vehicle2_desc = (
            f"Vehicle {track_ids[1]} ({movement_descs[1]})"
            if movement_descs[1] != "Unknown"
            else f"Vehicle {track_ids[1]}"
        )
        return f"At {timestamp_str}: {vehicle1_desc} and {vehicle2_desc} are {distance_desc}"

    elif stage == "first_contact":
        vehicle1_desc = (
            f"Vehicle {track_ids[0]} ({directions[0]})"
            if directions[0] != "Unknown"
            else f"Vehicle {track_ids[0]}"
        )
        vehicle2_desc = (
            f"Vehicle {track_ids[1]} ({directions[1]})"
            if directions[1] != "Unknown"
            else f"Vehicle {track_ids[1]}"
        )
        return f"At {timestamp_str}: First contact detected between {vehicle1_desc} and {vehicle2_desc}"

    elif stage == "peak_overlap":
        vehicle1_desc = (
            f"Vehicle {track_ids[0]} ({directions[0]})"
            if directions[0] != "Unknown"
            else f"Vehicle {track_ids[0]}"
        )
        vehicle2_desc = (
            f"Vehicle {track_ids[1]} ({directions[1]})"
            if directions[1] != "Unknown"
            else f"Vehicle {track_ids[1]}"
        )
        return f"At {timestamp_str}: Maximum overlap between {vehicle1_desc} and {vehicle2_desc}"

    elif stage == "separation":
        vehicle1_desc = (
            f"Vehicle {track_ids[0]} ({movement_descs[0]})"
            if movement_descs[0] != "Unknown"
            else f"Vehicle {track_ids[0]}"
        )
        vehicle2_desc = (
            f"Vehicle {track_ids[1]} ({movement_descs[1]})"
            if movement_descs[1] != "Unknown"
            else f"Vehicle {track_ids[1]}"
        )
        return f"At {timestamp_str}: Vehicles separating - {vehicle1_desc} and {vehicle2_desc}"

    else:
        return f"At {timestamp_str}: Stage {stage} at frame {row.frame}"


def report_assumptions(
    metric_rows: list[MetricRow],
    metadata: dict[str, Any],
    warn_if_missing: list[str] = ["world_coords", "speed_mph"],
) -> list[str]:
    """
    Report assumptions and data quality issues.

    Args:
        metric_rows: List of MetricRow objects
        metadata: Metadata from load_detections
        warn_if_missing: Fields to warn about if missing

    Returns:
        List of assumption/gap descriptions
    """
    assumptions = []

    if not metric_rows:
        assumptions.append("No metric data available for analysis")
        return assumptions

    # Check for missing data
    missing_speeds = sum(
        1 for row in metric_rows if row.speed_mph[0] is None or row.speed_mph[1] is None
    )
    missing_coords = sum(
        1 for row in metric_rows if not row.world_coords[0] or not row.world_coords[1]
    )

    if "speed_mph" in warn_if_missing and missing_speeds > 0:
        assumptions.append(
            f"Speed data missing for {missing_speeds}/{len(metric_rows)} frames"
        )

    if "world_coords" in warn_if_missing and missing_coords > 0:
        assumptions.append(
            f"World coordinates missing for {missing_coords}/{len(metric_rows)} frames"
        )

    # Check data consistency
    if metadata.get("data_gaps", {}).get("low_confidence", 0) > 0:
        low_conf_count = metadata["data_gaps"]["low_confidence"]
        total_count = metadata["data_gaps"]["total_detections"]
        assumptions.append(
            f"{low_conf_count}/{total_count} detections have low confidence (<0.5)"
        )

    # Check frame continuity
    missing_frames = metadata.get("missing_frames", [])
    if missing_frames:
        assumptions.append(
            f"Missing frames detected: {len(missing_frames)} gaps in sequence"
        )

    # Check collision detection assumptions
    collision_candidates = sum(1 for row in metric_rows if row.collision_candidate)
    if collision_candidates > 0:
        assumptions.append(
            f"{collision_candidates}/{len(metric_rows)} frames flagged as collision candidates"
        )

    # Check IoU distribution
    ious = [row.iou for row in metric_rows]
    if ious:
        max_iou = max(ious)
        avg_iou = sum(ious) / len(ious)
        assumptions.append(
            f"IoU range: {min(ious):.3f} - {max_iou:.3f} (avg: {avg_iou:.3f})"
        )

    return assumptions
