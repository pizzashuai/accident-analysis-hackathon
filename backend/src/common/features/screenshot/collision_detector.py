"""Collision detection helper that reuses existing postprocess tools."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, List

from ..postprocess.tools import (
    load_detections,
    compute_pair_metrics,
    trace_impact_window,
    Detection
)

logger = logging.getLogger(__name__)


def detect_collision_from_jsonl(
    jsonl_path: Union[str, Path],
    track_ids: List[int] = None,
    iou_threshold: float = 0.01,
    distance_threshold_m: float = 5.0,
    persistence_frames: int = 3
) -> Dict[str, Any]:
    """
    Detect collision from JSONL detection data using existing postprocess tools.
    
    Args:
        jsonl_path: Path to the JSONL detection file
        track_ids: List of track IDs to analyze (if None, auto-detect from data)
        iou_threshold: IoU threshold for collision detection
        distance_threshold_m: Distance threshold in meters
        persistence_frames: Minimum frames for collision persistence
        
    Returns:
        Dictionary with collision detection results:
        - success: bool
        - collision_detected: bool
        - collision_timestamp: float (if collision found)
        - collision_frame: int (if collision found)
        - track_ids: List[int]
        - impact_summary: dict (from trace_impact_window)
        - error: str (if failed)
    """
    jsonl_path = Path(jsonl_path)
    
    try:
        # Load detections from JSONL
        detections_data = _load_jsonl_detections(jsonl_path)
        
        if not detections_data:
            return {
                "success": False,
                "collision_detected": False,
                "collision_timestamp": None,
                "collision_frame": None,
                "track_ids": [],
                "impact_summary": {},
                "error": "No detections found in file"
            }
        
        # Auto-detect track IDs if not provided
        if track_ids is None:
            track_ids = _extract_track_ids(detections_data)
            if len(track_ids) < 2:
                return {
                    "success": False,
                    "collision_detected": False,
                    "collision_timestamp": None,
                    "collision_frame": None,
                    "track_ids": track_ids,
                    "impact_summary": {},
                    "error": f"Need at least 2 track IDs, found: {track_ids}"
                }
        
        logger.info(f"Analyzing collision for track IDs: {track_ids}")
        
        # Convert to Detection objects and save to temporary file for load_detections
        temp_file = _create_temp_detections_file(detections_data)
        
        try:
            # Load paired detections using existing tool
            detections_result = load_detections(
                track_ids=track_ids,
                detections_file=str(temp_file)
            )
            
            if not detections_result.get("records"):
                return {
                    "success": False,
                    "collision_detected": False,
                    "collision_timestamp": None,
                    "collision_frame": None,
                    "track_ids": track_ids,
                    "impact_summary": {},
                    "error": "No paired detections found for specified track IDs"
                }
            
            # Compute pair metrics using existing tool
            metric_rows = compute_pair_metrics(
                pairs=detections_result["records"],
                iou_threshold=iou_threshold
            )
            
            if not metric_rows:
                return {
                    "success": False,
                    "collision_detected": False,
                    "collision_timestamp": None,
                    "collision_frame": None,
                    "track_ids": track_ids,
                    "impact_summary": {},
                    "error": "No metric rows computed"
                }
            
            # Trace impact window using existing tool
            impact_summary = trace_impact_window(
                metric_rows=metric_rows,
                iou_threshold=iou_threshold,
                distance_threshold_m=distance_threshold_m,
                persistence_frames=persistence_frames
            )
            
            collision_detected = impact_summary.get("collision_detected", False)
            collision_frame = None
            collision_timestamp = None
            
            if collision_detected:
                collision_frame = impact_summary.get("first_contact_frame")
                # Find timestamp for collision frame
                for record in detections_result["records"]:
                    if record["frame"] == collision_frame:
                        collision_timestamp = record["timestamp"]
                        break
            
            logger.info(f"Collision analysis complete. Detected: {collision_detected}")
            if collision_detected:
                logger.info(f"Collision at frame {collision_frame}, timestamp {collision_timestamp}s")
            
            return {
                "success": True,
                "collision_detected": collision_detected,
                "collision_timestamp": collision_timestamp,
                "collision_frame": collision_frame,
                "track_ids": track_ids,
                "impact_summary": impact_summary,
                "error": None
            }
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
        
    except Exception as e:
        logger.error(f"Error detecting collision: {e}")
        return {
            "success": False,
            "collision_detected": False,
            "collision_timestamp": None,
            "collision_frame": None,
            "track_ids": [],
            "impact_summary": {},
            "error": str(e)
        }


def _load_jsonl_detections(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load detections from JSONL file."""
    detections = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    detection = json.loads(line.strip())
                    detections.append(detection)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
    
    return detections


def _extract_track_ids(detections: List[Dict[str, Any]]) -> List[int]:
    """Extract unique track IDs from detections."""
    track_ids = set()
    
    for detection in detections:
        track_id = detection.get('track_id')
        if track_id is not None:
            track_ids.add(track_id)
    
    return sorted(list(track_ids))


def _create_temp_detections_file(detections: List[Dict[str, Any]]) -> Path:
    """Create temporary detections file for load_detections function."""
    import tempfile
    
    temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
    
    with open(temp_file, 'w') as f:
        for detection in detections:
            f.write(json.dumps(detection) + '\n')
    
    return temp_file


def get_collision_point_from_detections(
    detections: List[Dict[str, Any]], 
    collision_frame: int
) -> tuple[float, float]:
    """
    Get the collision point (average of vehicle positions) at collision frame.
    
    Args:
        detections: List of detection dictionaries
        collision_frame: Frame number where collision occurred
        
    Returns:
        Tuple of (lat, lng) for collision point
    """
    collision_detections = [
        d for d in detections 
        if d.get('frame') == collision_frame and d.get('world_coords')
    ]
    
    if not collision_detections:
        return None, None
    
    # Calculate average position
    total_lat = 0
    total_lng = 0
    count = 0
    
    for detection in collision_detections:
        world_coords = detection.get('world_coords')
        if world_coords and len(world_coords) == 2:
            total_lng += world_coords[0]  # longitude (first element)
            total_lat += world_coords[1]  # latitude (second element)
            count += 1
    
    if count == 0:
        return None, None
    
    return total_lat / count, total_lng / count


def analyze_detection_data_quality(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality of detection data.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with data quality metrics
    """
    if not detections:
        return {"error": "No detections provided"}
    
    total_detections = len(detections)
    missing_world_coords = 0
    missing_speeds = 0
    low_confidence = 0
    track_ids = set()
    frame_range = [float('inf'), -float('inf')]
    
    for detection in detections:
        # Track IDs
        track_id = detection.get('track_id')
        if track_id is not None:
            track_ids.add(track_id)
        
        # Frame range
        frame = detection.get('frame')
        if frame is not None:
            frame_range[0] = min(frame_range[0], frame)
            frame_range[1] = max(frame_range[1], frame)
        
        # World coordinates
        if not detection.get('world_coords'):
            missing_world_coords += 1
        
        # Speed
        if detection.get('speed_mph') is None:
            missing_speeds += 1
        
        # Confidence
        conf = detection.get('conf', 1.0)
        if conf < 0.5:
            low_confidence += 1
    
    return {
        "total_detections": total_detections,
        "unique_track_ids": sorted(list(track_ids)),
        "frame_range": frame_range if frame_range[0] != float('inf') else None,
        "missing_world_coords": missing_world_coords,
        "missing_world_coords_pct": (missing_world_coords / total_detections) * 100,
        "missing_speeds": missing_speeds,
        "missing_speeds_pct": (missing_speeds / total_detections) * 100,
        "low_confidence": low_confidence,
        "low_confidence_pct": (low_confidence / total_detections) * 100,
        "data_quality_score": _calculate_quality_score(
            missing_world_coords, missing_speeds, low_confidence, total_detections
        )
    }


def _calculate_quality_score(
    missing_world_coords: int, 
    missing_speeds: int, 
    low_confidence: int, 
    total: int
) -> float:
    """Calculate a data quality score (0-100)."""
    if total == 0:
        return 0.0
    
    world_coords_score = ((total - missing_world_coords) / total) * 40
    speeds_score = ((total - missing_speeds) / total) * 30
    confidence_score = ((total - low_confidence) / total) * 30
    
    return world_coords_score + speeds_score + confidence_score
