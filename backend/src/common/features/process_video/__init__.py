"""
Process Video Module

This module provides video processing capabilities including:
- YOLO detection and ByteTrack tracking
- Speed calculation using homography
- Video annotation with bounding boxes and trails
- Detection persistence and retrieval
"""

from .main import process_video_with_supervision
from .src.annotate_video import VideoAnnotator
from .src.persist_detections import (
    write_detections_to_jsonl,
    read_detections_from_jsonl,
    get_detections_by_frame,
    convert_jsonl_to_supervision_detections,
)
from .src.enhance_frame import VideoEnhancer
from .src.estimate_distance import DistanceEstimator

__all__ = [
    "process_video_with_supervision",
    "VideoAnnotator", 
    "write_detections_to_jsonl",
    "read_detections_from_jsonl",
    "get_detections_by_frame",
    "convert_jsonl_to_supervision_detections",
    "VideoEnhancer",
    "DistanceEstimator",
]
