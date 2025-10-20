"""
Screenshot Generation Module

Generates collision screenshots and map overlays for PDF reports.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import cv2
import numpy as np
import requests

from src.common.features.screenshot.collision_detector import detect_collision_from_jsonl
from src.common.features.screenshot.map_overlay import generate_map_overlay
from src.common.features.screenshot.main import extract_trajectories_from_detections, get_collision_point_from_detections, calculate_map_center_and_zoom
from src.common.config import settings

logger = logging.getLogger(__name__)


def generate_collision_screenshots(
    video_path: str,
    detections_jsonl_path: str,
    analysis_result: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate collision screenshots and map overlay for PDF report.
    
    Args:
        video_path: Path to the video file
        detections_jsonl_path: Path to the detections JSONL file
        analysis_result: LLM analysis result containing timeline info
        output_dir: Optional output directory for screenshots
        
    Returns:
        Dictionary with generation results:
        - success: bool
        - video_screenshot_path: str
        - map_overlay_path: str
        - collision_frame: int
        - collision_timestamp: float
        - collision_point: tuple
        - error: str (if failed)
    """
    try:
        # Create output directory
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="collision_screenshots_")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract collision information from analysis result
        collision_info = _extract_collision_info(analysis_result)
        
        if not collision_info:
            return {
                "success": False,
                "error": "Could not extract collision information from analysis result",
                "video_screenshot_path": "",
                "map_overlay_path": "",
                "collision_frame": 0,
                "collision_timestamp": 0.0,
                "collision_point": (None, None)
            }
        
        collision_frame = collision_info["frame"]
        collision_timestamp = collision_info["timestamp"]
        
        logger.info(f"Generating screenshots for collision at frame {collision_frame}, timestamp {collision_timestamp}s")
        
        # Generate video screenshot
        video_screenshot_path = _generate_video_screenshot(
            video_path=video_path,
            frame_number=collision_frame,
            output_dir=output_dir
        )
        
        if not video_screenshot_path:
            return {
                "success": False,
                "error": "Failed to generate video screenshot",
                "video_screenshot_path": "",
                "map_overlay_path": "",
                "collision_frame": collision_frame,
                "collision_timestamp": collision_timestamp,
                "collision_point": (None, None)
            }
        
        # Generate map overlay
        map_result = _generate_map_overlay(
            detections_jsonl_path=detections_jsonl_path,
            collision_frame=collision_frame,
            output_dir=output_dir
        )
        
        if not map_result["success"]:
            logger.warning(f"Map overlay generation failed: {map_result['error']}")
            map_overlay_path = ""
            collision_point = (None, None)
        else:
            map_overlay_path = map_result["output_path"]
            collision_point = map_result["collision_point"]
        
        logger.info(f"Successfully generated collision screenshots")
        
        return {
            "success": True,
            "video_screenshot_path": video_screenshot_path,
            "map_overlay_path": map_overlay_path,
            "collision_frame": collision_frame,
            "collision_timestamp": collision_timestamp,
            "collision_point": collision_point,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Failed to generate collision screenshots: {e}")
        return {
            "success": False,
            "error": str(e),
            "video_screenshot_path": "",
            "map_overlay_path": "",
            "collision_frame": 0,
            "collision_timestamp": 0.0,
            "collision_point": (None, None)
        }


def _extract_collision_info(analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract collision frame and timestamp from LLM analysis result."""
    try:
        # Look for collision information in the analysis result
        # This could be in different formats depending on the LLM output
        
        # Method 1: Look for explicit collision data
        if "collision_data" in analysis_result:
            collision_data = analysis_result["collision_data"]
            if "frame" in collision_data and "timestamp" in collision_data:
                return {
                    "frame": collision_data["frame"],
                    "timestamp": collision_data["timestamp"]
                }
        
        # Method 2: Parse from timeline events
        if "timeline" in analysis_result:
            timeline = analysis_result["timeline"]
            for event in timeline:
                if isinstance(event, dict) and event.get("type") == "collision":
                    return {
                        "frame": event.get("frame", 0),
                        "timestamp": event.get("timestamp", 0.0)
                    }
        
        # Method 3: Parse from analysis text (fallback)
        analysis_text = analysis_result.get("analysis", "")
        if analysis_text:
            # Look for frame and timestamp patterns in the text
            import re
            
            # Look for patterns like "Frame: 49" or "frame 49"
            frame_match = re.search(r'(?:frame|Frame)[:\s]+(\d+)', analysis_text, re.IGNORECASE)
            # Look for patterns like "Timestamp: 1.234s" or "1.234 seconds"
            timestamp_match = re.search(r'(?:timestamp|time)[:\s]+(\d+\.?\d*)\s*(?:s|seconds?)?', analysis_text, re.IGNORECASE)
            
            if frame_match and timestamp_match:
                return {
                    "frame": int(frame_match.group(1)),
                    "timestamp": float(timestamp_match.group(1))
                }
        
        # Method 4: Use collision detection as fallback
        logger.warning("Could not extract collision info from analysis result, using fallback")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting collision info: {e}")
        return None


def _generate_video_screenshot(
    video_path: str,
    frame_number: int,
    output_dir: Path
) -> Optional[str]:
    """Generate screenshot from video at specific frame."""
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Clamp frame number to valid range
        frame_number = max(0, min(frame_number, total_frames - 1))
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Could not read frame {frame_number}")
            return None
        
        # Save screenshot
        screenshot_path = output_dir / f"collision_screenshot_frame_{frame_number}.png"
        success = cv2.imwrite(str(screenshot_path), frame)
        
        if not success:
            logger.error(f"Could not save screenshot to {screenshot_path}")
            return None
        
        logger.info(f"Generated video screenshot: {screenshot_path}")
        return str(screenshot_path)
        
    except Exception as e:
        logger.error(f"Error generating video screenshot: {e}")
        return None


def _generate_map_overlay(
    detections_jsonl_path: str,
    collision_frame: int,
    output_dir: Path
) -> Dict[str, Any]:
    """Generate map overlay with trajectories."""
    try:
        # Load detections
        detections = []
        with open(detections_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    detections.append(json.loads(line.strip()))
        
        if not detections:
            return {
                "success": False,
                "error": "No detections found",
                "output_path": "",
                "collision_point": (None, None)
            }
        
        # Detect collision to get track IDs
        collision_result = detect_collision_from_jsonl(
            jsonl_path=detections_jsonl_path,
            track_ids=None,  # Auto-detect
            iou_threshold=0.01,
            distance_threshold_m=5.0,
            persistence_frames=3
        )
        
        if not collision_result["collision_detected"]:
            logger.warning("No collision detected, using closest approach")
        
        # Get track IDs involved
        track_ids = collision_result.get("track_ids", [])
        if not track_ids:
            # Fallback: use first two unique track IDs
            unique_tracks = list(set(d["track_id"] for d in detections))
            track_ids = unique_tracks[:2]
        
        logger.info(f"Using track IDs for map overlay: {track_ids}")
        
        # Extract trajectories
        trajectories = extract_trajectories_from_detections(detections, track_ids)
        
        if not trajectories:
            return {
                "success": False,
                "error": "No trajectories extracted",
                "output_path": "",
                "collision_point": (None, None)
            }
        
        # Get collision point
        collision_point = get_collision_point_from_detections(detections, collision_frame)
        
        if collision_point[0] is None:
            return {
                "success": False,
                "error": "Could not determine collision point coordinates",
                "output_path": "",
                "collision_point": (None, None)
            }
        
        # Calculate map center and zoom
        center_lat, center_lng, zoom = calculate_map_center_and_zoom(trajectories, collision_point)
        
        # Generate map overlay
        map_path = output_dir / f"collision_map_frame_{collision_frame}.png"
        
        # Get Google Maps API key
        api_key = getattr(settings, "GOOGLE_MAP_API_KEY", None)
        if not api_key:
            return {
                "success": False,
                "error": "Google Maps API key not configured",
                "output_path": "",
                "collision_point": collision_point
            }
        
        map_result = generate_map_overlay(
            center_lat=center_lat,
            center_lng=center_lng,
            trajectories=trajectories,
            collision_point=collision_point,
            api_key=api_key,
            output_path=map_path,
            zoom=zoom,
            map_size="640x640"
        )
        
        if not map_result["success"]:
            return {
                "success": False,
                "error": map_result["error"],
                "output_path": "",
                "collision_point": collision_point
            }
        
        logger.info(f"Generated map overlay: {map_result['output_path']}")
        
        return {
            "success": True,
            "output_path": map_result["output_path"],
            "collision_point": collision_point,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error generating map overlay: {e}")
        return {
            "success": False,
            "error": str(e),
            "output_path": "",
            "collision_point": (None, None)
        }
