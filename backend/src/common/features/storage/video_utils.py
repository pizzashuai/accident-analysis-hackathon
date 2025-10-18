"""Video processing utilities using OpenCV."""

import logging
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_video_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract video metadata using OpenCV.
    
    Args:
        file_path: Path to video file
        
    Returns:
        dict: Video metadata including fps, duration, width, height, frame_count
        
    Raises:
        ValueError: If video file cannot be opened or is invalid
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(str(file_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {file_path}")
        
        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Release video capture
        cap.release()
        
        metadata = {
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
        
        logger.info(f"Extracted video metadata: {metadata}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract video metadata from {file_path}: {e}")
        raise ValueError(f"Failed to extract video metadata: {e}")


def extract_first_frame(video_path: Path, output_path: Path) -> bool:
    """
    Extract the first frame (frame 0) from video and save as PNG.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save extracted frame
        
    Returns:
        bool: True if extraction successful
        
    Raises:
        ValueError: If video file cannot be opened or frame extraction fails
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Read first frame
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise ValueError(f"Cannot read first frame from video: {video_path}")
        
        # Release video capture
        cap.release()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save frame as PNG
        success = cv2.imwrite(str(output_path), frame)
        
        if not success:
            raise ValueError(f"Failed to save frame to {output_path}")
        
        logger.info(f"Successfully extracted first frame to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract first frame from {video_path}: {e}")
        raise ValueError(f"Failed to extract first frame: {e}")


def validate_video_file(file_path: Path) -> bool:
    """
    Validate that file is a valid video file.
    
    Args:
        file_path: Path to video file
        
    Returns:
        bool: True if valid video file
    """
    try:
        cap = cv2.VideoCapture(str(file_path))
        
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, _ = cap.read()
        cap.release()
        
        return ret
        
    except Exception:
        return False
