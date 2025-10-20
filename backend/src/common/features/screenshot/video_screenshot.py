"""Video screenshot extraction using OpenCV."""

import logging
from pathlib import Path
from typing import Dict, Any, Union

import cv2

logger = logging.getLogger(__name__)


def extract_screenshot(
    video_path: Union[str, Path], 
    timestamp: float, 
    output_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Extract a screenshot from video at the specified timestamp.
    
    Args:
        video_path: Path to the video file
        timestamp: Timestamp in seconds to extract frame
        output_path: Path to save the screenshot (PNG format)
        
    Returns:
        Dictionary with extraction results:
        - success: bool
        - output_path: str
        - frame_number: int
        - timestamp: float
        - error: str (if failed)
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {
                "success": False,
                "output_path": str(output_path),
                "frame_number": -1,
                "timestamp": timestamp,
                "error": f"Could not open video file: {video_path}"
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            cap.release()
            return {
                "success": False,
                "output_path": str(output_path),
                "frame_number": -1,
                "timestamp": timestamp,
                "error": "Invalid FPS detected"
            }
        
        # Calculate frame number from timestamp
        frame_number = int(timestamp * fps)
        
        # Clamp frame number to valid range
        frame_number = max(0, min(frame_number, total_frames - 1))
        
        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return {
                "success": False,
                "output_path": str(output_path),
                "frame_number": frame_number,
                "timestamp": timestamp,
                "error": f"Could not read frame {frame_number}"
            }
        
        # Save frame as PNG
        success = cv2.imwrite(str(output_path), frame)
        
        cap.release()
        
        if not success:
            return {
                "success": False,
                "output_path": str(output_path),
                "frame_number": frame_number,
                "timestamp": timestamp,
                "error": f"Could not save frame to {output_path}"
            }
        
        logger.info(f"Successfully extracted screenshot at {timestamp}s (frame {frame_number}) to {output_path}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "frame_number": frame_number,
            "timestamp": timestamp,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error extracting screenshot: {e}")
        return {
            "success": False,
            "output_path": str(output_path),
            "frame_number": -1,
            "timestamp": timestamp,
            "error": str(e)
        }


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get video metadata information.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video info:
        - fps: float
        - total_frames: int
        - duration: float (seconds)
        - width: int
        - height: int
    """
    video_path = Path(video_path)
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {"error": f"Could not open video file: {video_path}"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {"error": str(e)}
