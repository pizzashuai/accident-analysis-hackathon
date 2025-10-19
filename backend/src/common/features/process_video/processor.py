"""
Unified Video Processing Interface

This module provides clean, simple interfaces for video processing tasks,
eliminating duplication between tasks.py and the process_video modules.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .src.enhance_frame import VideoEnhancer
from .src.persist_detections import write_detections_to_jsonl, read_detections_from_jsonl
from .src.annotate_video import VideoAnnotator
from .src.estimate_distance import DistanceEstimator
from .src.bbox_smoothing.bbox_smoother import BboxSmoother
from .src.speed_smoothing.speed_smoother import SpeedSmoother

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Unified video processing interface that handles detection, tracking, and speed calculation.
    
    This class consolidates all video processing logic into clean, reusable methods.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.3,
        classes: Optional[List[int]] = None,
        trail_length: int = 10,
        # Tracking parameters
        minimum_consecutive_frames: int = 2,
        track_activation_threshold: float = 0.1,
        lost_track_buffer: int = 100,
        minimum_matching_threshold: float = 0.95,
        # Smoothing parameters (optimal settings)
        bbox_smoothing_method: str = "kalman",
        bbox_smoothing_window: int = 5,
        speed_smoothing_method: str = "kalman_with_outlier_rejection", 
        speed_smoothing_window: int = 5,
        tracking_point: str = "bottom_center",
        # Speed limits for outlier rejection
        max_reasonable_speed: float = 100.0,  # mph
        min_reasonable_speed: float = 0.0,    # mph
    ):
        """
        Initialize video processor with optimal settings.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (None for default vehicle classes)
            trail_length: Length of tracking trails
            minimum_consecutive_frames: Minimum frames for track activation
            track_activation_threshold: Threshold for track activation
            lost_track_buffer: Buffer for lost tracks
            minimum_matching_threshold: Minimum threshold for track matching
            bbox_smoothing_method: Bbox smoothing method (optimal: "kalman")
            bbox_smoothing_window: Window size for bbox smoothing
            speed_smoothing_method: Speed smoothing method (optimal: "kalman_with_outlier_rejection")
            speed_smoothing_window: Window size for speed smoothing
            tracking_point: Point to track on vehicle (optimal: "bottom_center")
            max_reasonable_speed: Maximum reasonable speed for outlier rejection (mph)
            min_reasonable_speed: Minimum reasonable speed for outlier rejection (mph)
        """
        # Default vehicle classes (COCO): car, motorcycle, bus, truck
        if classes is None:
            classes = [2, 3, 5, 7, 9]
            
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.trail_length = trail_length
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.bbox_smoothing_method = bbox_smoothing_method
        self.bbox_smoothing_window = bbox_smoothing_window
        self.speed_smoothing_method = speed_smoothing_method
        self.speed_smoothing_window = speed_smoothing_window
        self.tracking_point = tracking_point
        self.max_reasonable_speed = max_reasonable_speed
        self.min_reasonable_speed = min_reasonable_speed
        
        # Initialize components
        self._model = None
        self._byte_tracker = None
        self._enhancer = None
        self._bbox_smoother = None
        self._speed_smoother = None
        self._distance_estimator = None
        
    def _initialize_components(self, fps: float, homography_file: Optional[str] = None):
        """Initialize processing components."""
        if self._model is None:
            from ultralytics import YOLO
            import supervision as sv
            
            self._model = YOLO(self.model_path)
            self._byte_tracker = sv.ByteTrack(
                minimum_consecutive_frames=self.minimum_consecutive_frames,
                frame_rate=fps,
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
            )
            self._enhancer = VideoEnhancer()
            
        if homography_file and self._distance_estimator is None:
            self._distance_estimator = DistanceEstimator(homography_file)
            
        if self._bbox_smoother is None:
            self._bbox_smoother = BboxSmoother(
                method=self.bbox_smoothing_method,
                window_size=self.bbox_smoothing_window,
                kalman_q=0.5,
                kalman_r=3.0,
            )
            
        if self._speed_smoother is None:
            self._speed_smoother = SpeedSmoother(
                method=self.speed_smoothing_method,
                window_size=self.speed_smoothing_window,
            )
    
    def extract_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract video information including properties needed for processing.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        cap.release()

        if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
            raise RuntimeError("Video probe failed (width/height/fps/frames invalid).")

        return {
            "video": str(video_path.resolve()),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": total_frames,
            "duration_sec": duration_sec,
            "ok": True,
        }
    
    def process_video_detections(
        self,
        video_path: Path,
        output_dir: Path,
        homography_file: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process video for detection and tracking, saving results to JSONL.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output files
            homography_file: Optional path to homography file for speed calculation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        import supervision as sv
        
        # Extract video info
        video_info = self.extract_video_info(video_path)
        fps = video_info["fps"]
        width = video_info["width"]
        height = video_info["height"]
        total_frames = video_info["frames"]
        
        # Initialize components
        self._initialize_components(fps, homography_file)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "detections.jsonl"
        
        # Clear any existing JSONL file
        if jsonl_path.exists():
            jsonl_path.unlink()
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        logger.info(f"Using optimal smoothing: {self.bbox_smoothing_method} bbox + {self.speed_smoothing_method} speed + {self.tracking_point} tracking")
        
        # Process video frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        frame_count = 0
        processed_frames = 0
        
        # Tracking state for speed calculation
        vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
        vehicle_speeds = {}    # {track_id: current_speed_mph}
        vehicle_bboxes = {}    # {track_id: [bbox_history] for smoothing}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance frame
            enhanced_frame = self._enhancer.enhance(frame)
            
            # Run detection
            results = self._model(
                enhanced_frame,
                conf=self.conf_threshold,
                classes=self.classes,
                verbose=False,
                iou=self.iou_threshold,
            )[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            detections = self._byte_tracker.update_with_detections(detections)
            
            # Process detections for speed calculation if homography is available
            if self._distance_estimator and detections.tracker_id is not None:
                self._process_detections_for_speed(
                    detections, frame_count, fps, width, height,
                    vehicle_positions, vehicle_speeds, vehicle_bboxes
                )
            
            # Write detections to JSONL
            write_detections_to_jsonl(
                detections, self._model, frame_count, fps, video_path.name, jsonl_path
            )
            
            processed_frames += 1
            
            # Progress callback
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames, f"Processed {frame_count}/{total_frames} frames")
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Detection and tracking complete! Processed {processed_frames} frames")
        logger.info(f"Detections saved to: {jsonl_path}")
        
        return {
            "jsonl_path": jsonl_path,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "video_info": video_info,
            "vehicle_speeds": vehicle_speeds,
        }
    
    def _process_detections_for_speed(
        self,
        detections,
        frame_count: int,
        fps: float,
        width: int,
        height: int,
        vehicle_positions: Dict,
        vehicle_speeds: Dict,
        vehicle_bboxes: Dict,
    ):
        """Process detections for speed calculation with optimal smoothing."""
        timestamp = frame_count / fps
        
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            if track_id is None:
                continue
                
            bbox = detections.xyxy[i]
            
            # Apply bbox smoothing for more stable tracking
            smoothed_bbox = bbox.copy()
            if track_id not in vehicle_bboxes:
                vehicle_bboxes[track_id] = []
            vehicle_bboxes[track_id].append(bbox)
            
            # Keep only last 10 frames of bbox history
            if len(vehicle_bboxes[track_id]) > 10:
                vehicle_bboxes[track_id] = vehicle_bboxes[track_id][-10:]
            
            # Apply Kalman smoothing to bbox
            if len(vehicle_bboxes[track_id]) >= 2:
                smoothed_bbox = self._bbox_smoother._kalman_smooth(track_id, bbox)
            
            # Use bottom center for more stable tracking
            center_x = smoothed_bbox[0] + (smoothed_bbox[2] - smoothed_bbox[0]) / 2
            center_y = smoothed_bbox[3]  # Bottom center instead of center
            
            # Convert to normalized coordinates (0-1 range)
            x_norm = center_x / width
            y_norm = center_y / height
            
            # Initialize tracking history for this vehicle
            if track_id not in vehicle_positions:
                vehicle_positions[track_id] = []
            
            # Add current position (ensure all values are Python types)
            vehicle_positions[track_id].append((frame_count, float(x_norm), float(y_norm), float(timestamp)))
            
            # Keep only last 30 frames of history
            if len(vehicle_positions[track_id]) > 30:
                vehicle_positions[track_id] = vehicle_positions[track_id][-30:]
            
            # Calculate speed if we have enough history (use 5 frames for smoothing)
            history = vehicle_positions[track_id]
            if len(history) >= 5:
                old_frame, old_x, old_y, old_time = history[-5]
                new_frame, new_x, new_y, new_time = history[-1]
                
                # Calculate distance using homography
                try:
                    distance_meters = self._distance_estimator.estimate_distance(
                        (old_x, old_y), (new_x, new_y)
                    )
                    
                    # Calculate time difference
                    time_diff = new_time - old_time
                    
                    if time_diff > 0:
                        # Calculate speed in meters per second
                        speed_mps = distance_meters / time_diff
                        
                        # Convert to miles per hour (1 m/s = 2.23694 mph)
                        raw_speed_mph = speed_mps * 2.23694
                        
                        # Apply outlier rejection before smoothing
                        max_reasonable_speed = self.max_reasonable_speed
                        min_reasonable_speed = self.min_reasonable_speed
                        
                        # Reject obvious outliers
                        if raw_speed_mph < min_reasonable_speed or raw_speed_mph > max_reasonable_speed:
                            # Use previous speed if available, otherwise skip
                            if track_id in vehicle_speeds:
                                raw_speed_mph = vehicle_speeds[track_id]
                            else:
                                continue  # Skip this calculation
                        
                        # Apply speed smoothing (now using Kalman with outlier rejection)
                        speed_mph = self._speed_smoother.smooth_speed(track_id, raw_speed_mph)
                        
                        # Final bounds check after smoothing
                        if speed_mph < min_reasonable_speed or speed_mph > max_reasonable_speed:
                            if track_id in vehicle_speeds:
                                speed_mph = vehicle_speeds[track_id]  # Use previous speed
                            else:
                                speed_mph = min(max(speed_mph, min_reasonable_speed), max_reasonable_speed)
                        
                        # Store current speed for this vehicle
                        vehicle_speeds[track_id] = speed_mph
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
                    # Use previous speed if available
                    if track_id in vehicle_speeds:
                        pass  # Keep existing speed
    
    def create_homography_data(
        self,
        homography_session,
        homography_model,
    ) -> Dict[str, Any]:
        """
        Create homography data in the format expected by DistanceEstimator.
        
        Args:
            homography_session: Database homography session object
            homography_model: Database homography model object
            
        Returns:
            Dictionary with homography data
        """
        pairs_data = []
        logger.info(f"Found {len(homography_session.pairs)} homography pairs")
        
        for idx, pair in enumerate(homography_session.pairs):
            logger.info(f"Pair {idx}: image({pair.image_x_norm:.4f}, {pair.image_y_norm:.4f}) -> geo({pair.map_lat:.6f}, {pair.map_lng:.6f})")
            pairs_data.append({
                "id": idx,
                "a": {
                    "xNorm": float(pair.image_x_norm),  # Ensure Python float
                    "yNorm": float(pair.image_y_norm)   # Ensure Python float
                },
                "b": {
                    "lat": float(pair.map_lat),  # Ensure Python float
                    "lng": float(pair.map_lng)   # Ensure Python float
                }
            })
        
        homography_data = {
            "pairs": pairs_data,
            "imagesMeta": homography_model.meta.get("imagesMeta", {}) if homography_model.meta else {},
            "mapMeta": homography_model.meta.get("mapMeta", {}) if homography_model.meta else {},
        }
        
        logger.info(f"Created homography data with {len(pairs_data)} pairs")
        return homography_data
    
    def save_homography_file(self, homography_data: Dict[str, Any]) -> Path:
        """
        Save homography data to a temporary file.
        
        Args:
            homography_data: Homography data dictionary
            
        Returns:
            Path to temporary homography file
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as homography_file:
            json.dump(homography_data, homography_file)
            return Path(homography_file.name)
    
    def process_video_detections_from_objects(
        self,
        video_frames: List[np.ndarray],
        fps: float,
        homography_data: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process video frames for detection and tracking using Python objects instead of files.
        
        Args:
            video_frames: List of video frames as numpy arrays
            fps: Video frame rate
            homography_data: Optional homography data dictionary for speed calculation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        import supervision as sv
        
        total_frames = len(video_frames)
        if total_frames == 0:
            raise ValueError("No frames provided")
        
        # Get frame dimensions from first frame
        height, width = video_frames[0].shape[:2]
        
        # Initialize components
        homography_file = None
        if homography_data:
            homography_file = str(self.save_homography_file(homography_data))
        self._initialize_components(fps, homography_file)
        
        logger.info(f"Processing {total_frames} frames: {width}x{height} @ {fps}fps")
        logger.info(f"Using optimal smoothing: {self.bbox_smoothing_method} bbox + {self.speed_smoothing_method} speed + {self.tracking_point} tracking")
        
        # Process video frames
        processed_frames = 0
        
        # Tracking state for speed calculation
        vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
        vehicle_speeds = {}    # {track_id: current_speed_mph}
        vehicle_bboxes = {}    # {track_id: [bbox_history] for smoothing}
        
        # Store detections in memory instead of writing to file
        detections_list = []
        
        for frame_count, frame in enumerate(video_frames):
            # Enhance frame
            enhanced_frame = self._enhancer.enhance(frame)
            
            # Run detection
            results = self._model(
                enhanced_frame,
                conf=self.conf_threshold,
                classes=self.classes,
                verbose=False,
                iou=self.iou_threshold,
            )[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            detections = self._byte_tracker.update_with_detections(detections)
            
            # Process detections for speed calculation if homography is available
            if self._distance_estimator and detections.tracker_id is not None:
                self._process_detections_for_speed(
                    detections, frame_count, fps, width, height,
                    vehicle_positions, vehicle_speeds, vehicle_bboxes
                )
            
            # Convert detections to dictionary format
            time_sec = frame_count / fps
            class_names = getattr(getattr(self._model, "model", None), "names", {}) or {}
            
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

                detection_record = {
                    "video_id": "video.mp4",  # Default video ID
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
                detections_list.append(detection_record)
            
            processed_frames += 1
            
            # Progress callback
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames, f"Processed {frame_count}/{total_frames} frames")
        
        logger.info(f"Detection and tracking complete! Processed {processed_frames} frames")
        
        return {
            "detections": detections_list,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "frames": total_frames,
                "duration_sec": total_frames / fps if fps > 0 else 0,
                "ok": True,
            },
            "vehicle_speeds": vehicle_speeds,
        }
    
    def convert_detections_to_database_format_from_objects(
        self,
        detections_list: List[Dict[str, Any]],
        project_uuid,
        video_width: int,
        video_height: int,
        homography_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert detection objects to database format with speed calculation.
        
        Args:
            detections_list: List of detection dictionaries
            project_uuid: Project UUID
            video_width: Video width in pixels
            video_height: Video height in pixels
            homography_data: Optional homography data dictionary
            
        Returns:
            List of detection records in database format
        """
        detections_data = []
        
        # Initialize distance estimator if homography is available
        distance_estimator = None
        if homography_data:
            # Create temporary file for DistanceEstimator
            homography_path = self.save_homography_file(homography_data)
            distance_estimator = DistanceEstimator(str(homography_path))
            logger.info(f"Initialized DistanceEstimator with {len(distance_estimator.homography_data.pairs)} point pairs")
        
        # Initialize optimal smoothing components
        bbox_smoother = BboxSmoother(
            method=self.bbox_smoothing_method,
            window_size=self.bbox_smoothing_window,
            kalman_q=0.5,
            kalman_r=3.0,
        )
        speed_smoother = SpeedSmoother(
            method=self.speed_smoothing_method,
            window_size=self.speed_smoothing_window,
        )
        
        # Track vehicle positions for speed calculation with smoothing
        vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
        vehicle_speeds = {}    # {track_id: current_speed_mph}
        vehicle_bboxes = {}    # {track_id: [bbox_history] for smoothing}
        
        for detection in detections_list:
            # Convert to database format
            bbox = detection["bbox_xyxy"]
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Apply bbox smoothing for more stable tracking
            track_id = detection.get("track_id")
            smoothed_bbox = bbox.copy()
            if track_id is not None:
                # Store bbox history for smoothing
                if track_id not in vehicle_bboxes:
                    vehicle_bboxes[track_id] = []
                vehicle_bboxes[track_id].append(bbox)
                
                # Keep only last 10 frames of bbox history
                if len(vehicle_bboxes[track_id]) > 10:
                    vehicle_bboxes[track_id] = vehicle_bboxes[track_id][-10:]
                
                # Apply Kalman smoothing to bbox
                if len(vehicle_bboxes[track_id]) >= 2:
                    smoothed_bbox = bbox_smoother._kalman_smooth(track_id, np.array(bbox))
            
            # Use bottom center for more stable tracking
            center_x = smoothed_bbox[0] + (smoothed_bbox[2] - smoothed_bbox[0]) / 2
            center_y = smoothed_bbox[3]  # Bottom center instead of center
            
            # Convert to normalized coordinates (0-1 range)
            x_norm = center_x / video_width
            y_norm = center_y / video_height
            
            # Transform to world coordinates using homography
            wx, wy = None, None
            if distance_estimator:
                try:
                    geo_point = distance_estimator.image_to_geo(x_norm, y_norm)
                    wx, wy = geo_point.lng, geo_point.lat
                except Exception as e:
                    logger.warning(f"Failed to transform coordinates: {e}")
            
            # Calculate speed if we have tracking data
            speed_mph = None
            if track_id is not None and distance_estimator:
                # Initialize tracking history for this vehicle
                if track_id not in vehicle_positions:
                    vehicle_positions[track_id] = []
                
                # Add current position (ensure all values are Python types)
                timestamp = detection["time"]
                vehicle_positions[track_id].append((detection["frame"], float(x_norm), float(y_norm), float(timestamp)))
                
                # Keep only last 30 frames of history
                if len(vehicle_positions[track_id]) > 30:
                    vehicle_positions[track_id] = vehicle_positions[track_id][-30:]
                
                # Calculate speed if we have enough history (use 5 frames for smoothing)
                history = vehicle_positions[track_id]
                if len(history) >= 5:
                    old_frame, old_x, old_y, old_time = history[-5]
                    new_frame, new_x, new_y, new_time = history[-1]
                    
                    # Calculate distance using homography
                    try:
                        distance_meters = distance_estimator.estimate_distance(
                            (old_x, old_y), (new_x, new_y)
                        )
                        
                        # Calculate time difference
                        time_diff = new_time - old_time
                        
                        if time_diff > 0:
                            # Calculate speed in meters per second
                            speed_mps = distance_meters / time_diff
                            
                            # Convert to miles per hour (1 m/s = 2.23694 mph)
                            raw_speed_mph = speed_mps * 2.23694
                            
                            # Apply outlier rejection before smoothing
                            max_reasonable_speed = self.max_reasonable_speed
                            min_reasonable_speed = self.min_reasonable_speed
                            
                            # Reject obvious outliers
                            if raw_speed_mph < min_reasonable_speed or raw_speed_mph > max_reasonable_speed:
                                # Use previous speed if available, otherwise skip
                                if track_id in vehicle_speeds:
                                    raw_speed_mph = vehicle_speeds[track_id]
                                else:
                                    continue  # Skip this calculation
                            
                            # Apply speed smoothing (now using Kalman with outlier rejection)
                            speed_mph = speed_smoother.smooth_speed(track_id, raw_speed_mph)
                            
                            # Final bounds check after smoothing
                            if speed_mph < min_reasonable_speed or speed_mph > max_reasonable_speed:
                                if track_id in vehicle_speeds:
                                    speed_mph = vehicle_speeds[track_id]  # Use previous speed
                                else:
                                    speed_mph = min(max(speed_mph, min_reasonable_speed), max_reasonable_speed)
                            
                            # Store current speed for this vehicle
                            vehicle_speeds[track_id] = speed_mph
                            
                    except Exception as e:
                        logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
                        speed_mph = vehicle_speeds.get(track_id)  # Use previous speed if available
            
            # Use smoothed bbox coordinates for database storage (ensure Python native types)
            smoothed_x = float(smoothed_bbox[0])
            smoothed_y = float(smoothed_bbox[1])
            smoothed_w = float(smoothed_bbox[2] - smoothed_bbox[0])
            smoothed_h = float(smoothed_bbox[3] - smoothed_bbox[1])
            
            detection_data = {
                "project_id": project_uuid,
                "frame_idx": detection["frame"],
                "t_ms": int(detection["time"] * 1000),
                "track_id": track_id,
                "cls": detection["class_name"],
                "conf": float(detection["conf"]) if detection["conf"] is not None else 0.0,
                "x": smoothed_x,
                "y": smoothed_y,
                "w": smoothed_w,
                "h": smoothed_h,
                "wx": float(wx) if wx is not None else None,
                "wy": float(wy) if wy is not None else None,
                "extra": {
                    "speed_mph": float(speed_mph) if speed_mph is not None else None,
                    "class_id": int(detection["class_id"]) if detection["class_id"] is not None else 0,
                    "center": [float(center_x), float(center_y)],  # Convert to Python floats
                    "raw_bbox": [float(x) for x in bbox],  # Convert numpy array to Python list
                    "smoothed_bbox": [float(x) for x in smoothed_bbox],  # Convert numpy array to Python list
                    "tracking_point": self.tracking_point,  # Indicate tracking method used
                }
            }
            detections_data.append(detection_data)
        
        # Debug logging for detections data
        logger.info(f"Processed {len(detections_data)} detections from objects")
        speed_count = sum(1 for d in detections_data if d["extra"].get("speed_mph") is not None)
        logger.info(f"Detections with speed data: {speed_count}/{len(detections_data)}")
        
        return detections_data
    
    def convert_detections_to_database_format(
        self,
        jsonl_path: Path,
        project_uuid,
        video_width: int,
        video_height: int,
        homography_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert JSONL detections to database format with speed calculation.
        
        Args:
            jsonl_path: Path to JSONL detections file
            project_uuid: Project UUID
            video_width: Video width in pixels
            video_height: Video height in pixels
            homography_file: Optional path to homography file
            
        Returns:
            List of detection records in database format
        """
        detections_data = []
        
        # Initialize distance estimator if homography is available
        distance_estimator = None
        if homography_file:
            distance_estimator = DistanceEstimator(homography_file)
            logger.info(f"Initialized DistanceEstimator with {len(distance_estimator.homography_data.pairs)} point pairs")
        
        # Initialize optimal smoothing components
        bbox_smoother = BboxSmoother(
            method=self.bbox_smoothing_method,
            window_size=self.bbox_smoothing_window,
            kalman_q=0.5,
            kalman_r=3.0,
        )
        speed_smoother = SpeedSmoother(
            method=self.speed_smoothing_method,
            window_size=self.speed_smoothing_window,
        )
        
        # Track vehicle positions for speed calculation with smoothing
        vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
        vehicle_speeds = {}    # {track_id: current_speed_mph}
        vehicle_bboxes = {}    # {track_id: [bbox_history] for smoothing}
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                detection = json.loads(line.strip())
                
                # Convert to database format
                bbox = detection["bbox_xyxy"]
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Apply bbox smoothing for more stable tracking
                track_id = detection.get("track_id")
                smoothed_bbox = bbox.copy()
                if track_id is not None:
                    # Store bbox history for smoothing
                    if track_id not in vehicle_bboxes:
                        vehicle_bboxes[track_id] = []
                    vehicle_bboxes[track_id].append(bbox)
                    
                    # Keep only last 10 frames of bbox history
                    if len(vehicle_bboxes[track_id]) > 10:
                        vehicle_bboxes[track_id] = vehicle_bboxes[track_id][-10:]
                    
                    # Apply Kalman smoothing to bbox
                    if len(vehicle_bboxes[track_id]) >= 2:
                        smoothed_bbox = bbox_smoother._kalman_smooth(track_id, bbox)
                
                # Use bottom center for more stable tracking
                center_x = smoothed_bbox[0] + (smoothed_bbox[2] - smoothed_bbox[0]) / 2
                center_y = smoothed_bbox[3]  # Bottom center instead of center
                
                # Convert to normalized coordinates (0-1 range)
                x_norm = center_x / video_width
                y_norm = center_y / video_height
                
                # Transform to world coordinates using homography
                wx, wy = None, None
                if distance_estimator:
                    try:
                        geo_point = distance_estimator.image_to_geo(x_norm, y_norm)
                        wx, wy = geo_point.lng, geo_point.lat
                    except Exception as e:
                        logger.warning(f"Failed to transform coordinates: {e}")
                
                # Calculate speed if we have tracking data
                speed_mph = None
                if track_id is not None and distance_estimator:
                    # Initialize tracking history for this vehicle
                    if track_id not in vehicle_positions:
                        vehicle_positions[track_id] = []
                    
                    # Add current position (ensure all values are Python types)
                    timestamp = detection["time"]
                    vehicle_positions[track_id].append((detection["frame"], float(x_norm), float(y_norm), float(timestamp)))
                    
                    # Keep only last 30 frames of history
                    if len(vehicle_positions[track_id]) > 30:
                        vehicle_positions[track_id] = vehicle_positions[track_id][-30:]
                    
                    # Calculate speed if we have enough history (use 5 frames for smoothing)
                    history = vehicle_positions[track_id]
                    if len(history) >= 5:
                        old_frame, old_x, old_y, old_time = history[-5]
                        new_frame, new_x, new_y, new_time = history[-1]
                        
                        # Calculate distance using homography
                        try:
                            distance_meters = distance_estimator.estimate_distance(
                                (old_x, old_y), (new_x, new_y)
                            )
                            
                            # Calculate time difference
                            time_diff = new_time - old_time
                            
                            if time_diff > 0:
                                # Calculate speed in meters per second
                                speed_mps = distance_meters / time_diff
                                
                                # Convert to miles per hour (1 m/s = 2.23694 mph)
                                raw_speed_mph = speed_mps * 2.23694
                                
                                # Apply outlier rejection before smoothing
                                max_reasonable_speed = self.max_reasonable_speed
                                min_reasonable_speed = self.min_reasonable_speed
                                
                                # Reject obvious outliers
                                if raw_speed_mph < min_reasonable_speed or raw_speed_mph > max_reasonable_speed:
                                    # Use previous speed if available, otherwise skip
                                    if track_id in vehicle_speeds:
                                        raw_speed_mph = vehicle_speeds[track_id]
                                    else:
                                        continue  # Skip this calculation
                                
                                # Apply speed smoothing (now using Kalman with outlier rejection)
                                speed_mph = speed_smoother.smooth_speed(track_id, raw_speed_mph)
                                
                                # Final bounds check after smoothing
                                if speed_mph < min_reasonable_speed or speed_mph > max_reasonable_speed:
                                    if track_id in vehicle_speeds:
                                        speed_mph = vehicle_speeds[track_id]  # Use previous speed
                                    else:
                                        speed_mph = min(max(speed_mph, min_reasonable_speed), max_reasonable_speed)
                                
                                # Store current speed for this vehicle
                                vehicle_speeds[track_id] = speed_mph
                                
                        except Exception as e:
                            logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
                            speed_mph = vehicle_speeds.get(track_id)  # Use previous speed if available
                
                # Use smoothed bbox coordinates for database storage (ensure Python native types)
                smoothed_x = float(smoothed_bbox[0])
                smoothed_y = float(smoothed_bbox[1])
                smoothed_w = float(smoothed_bbox[2] - smoothed_bbox[0])
                smoothed_h = float(smoothed_bbox[3] - smoothed_bbox[1])
                
                detection_data = {
                    "project_id": project_uuid,
                    "frame_idx": detection["frame"],
                    "t_ms": int(detection["time"] * 1000),
                    "track_id": track_id,
                    "cls": detection["class_name"],
                    "conf": float(detection["conf"]) if detection["conf"] is not None else 0.0,
                    "x": smoothed_x,
                    "y": smoothed_y,
                    "w": smoothed_w,
                    "h": smoothed_h,
                    "wx": float(wx) if wx is not None else None,
                    "wy": float(wy) if wy is not None else None,
                    "extra": {
                        "speed_mph": float(speed_mph) if speed_mph is not None else None,
                        "class_id": int(detection["class_id"]) if detection["class_id"] is not None else 0,
                        "center": [float(center_x), float(center_y)],  # Convert to Python floats
                        "raw_bbox": [float(x) for x in bbox],  # Convert numpy array to Python list
                        "smoothed_bbox": [float(x) for x in smoothed_bbox],  # Convert numpy array to Python list
                        "tracking_point": self.tracking_point,  # Indicate tracking method used
                    }
                }
                detections_data.append(detection_data)
        
        # Debug logging for detections data
        logger.info(f"Processed {len(detections_data)} detections from JSONL")
        speed_count = sum(1 for d in detections_data if d["extra"].get("speed_mph") is not None)
        logger.info(f"Detections with speed data: {speed_count}/{len(detections_data)}")
        
        return detections_data
    
    def update_jsonl_with_speed_data(
        self,
        jsonl_path: Path,
        detections_data: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Tuple[Path, int]:
        """
        Update JSONL file with speed data from processed detections.
        
        Args:
            jsonl_path: Path to original JSONL file
            detections_data: Processed detections with speed data
            output_dir: Directory to save updated JSONL
            
        Returns:
            Tuple of (updated_jsonl_path, speed_updates_count)
        """
        # Update JSONL file with speed data
        updated_jsonl_path = output_dir / "detections_with_speed.jsonl"
        speed_updates_count = 0
        total_detections = 0
        
        with open(jsonl_path, 'r') as input_file, open(updated_jsonl_path, 'w') as output_file:
            for line in input_file:
                detection = json.loads(line.strip())
                total_detections += 1
                
                # Find matching detection in our processed data
                matching_detection = None
                for det_data in detections_data:
                    if (det_data["frame_idx"] == detection["frame"] and 
                        det_data["track_id"] == detection.get("track_id")):
                        matching_detection = det_data
                        break
                
                # Add speed data if found
                if matching_detection and matching_detection["extra"].get("speed_mph") is not None:
                    detection["speed_mph"] = matching_detection["extra"]["speed_mph"]
                    speed_updates_count += 1
                    
                    # Debug logging for first few speed updates
                    if speed_updates_count <= 5:
                        logger.info(f"Updated detection frame {detection['frame']} track {detection.get('track_id')} with speed {detection['speed_mph']:.2f} mph")
                
                output_file.write(json.dumps(detection) + "\n")
        
        logger.info(f"JSONL update complete: {speed_updates_count}/{total_detections} detections updated with speed data")
        return updated_jsonl_path, speed_updates_count
    
    def create_annotated_video(
        self,
        original_video_path: Path,
        jsonl_path: Path,
        output_path: Path,
        homography_file: Optional[str] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> None:
        """
        Create annotated video using VideoAnnotator with optimal settings.
        
        Args:
            original_video_path: Path to original video
            jsonl_path: Path to JSONL detections file
            output_path: Path for output annotated video
            homography_file: Optional path to homography file for speed calculation
            show_trails: Whether to show tracking trails
            show_labels: Whether to show detection labels
            show_boxes: Whether to show bounding boxes
        """
        logger.info(f"Using optimal smoothing configuration: Kalman filter for bbox (22.2% improvement) + Kalman with outlier rejection for speed (max speed: {self.max_reasonable_speed} mph) + Bottom center tracking")
        
        annotator = VideoAnnotator(
            trail_length=self.trail_length,
            homography_file=homography_file,  # Enable speed calculation
            bbox_smoothing=self.bbox_smoothing_method,  # Use Kalman filter for best speed stability
            bbox_smoothing_window=self.bbox_smoothing_window,
            speed_smoothing=self.speed_smoothing_method,  # Use moving average for best speed smoothing
            smoothing_window=self.speed_smoothing_window,
            tracking_point=self.tracking_point,  # Use bottom center for more stable tracking
            debug_speed=False,  # Disable debug output for production
        )
        
        # Render annotated video
        annotator.annotate_video_from_jsonl(
            original_video_path=original_video_path,
            jsonl_path=jsonl_path,
            output_path=output_path,
            show_trails=show_trails,
            show_labels=show_labels,
            show_boxes=show_boxes,
        )
        
        logger.info(f"Annotated video saved to: {output_path}")


class VideoProcessingResult:
    """Container for video processing results."""
    
    def __init__(
        self,
        success: bool,
        detection_count: int = 0,
        duration_sec: float = 0.0,
        fps: float = 0.0,
        frame_count: int = 0,
        video_uri: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self.success = success
        self.detection_count = detection_count
        self.duration_sec = duration_sec
        self.fps = fps
        self.frame_count = frame_count
        self.video_uri = video_uri
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "detection_count": self.detection_count,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "video_uri": self.video_uri,
            "error_message": self.error_message,
        }


def analyze_speeds_from_detections(
    jsonl_path: Path,
    homography_file: Optional[str] = None,
    max_reasonable_speed: float = 150.0,
) -> Dict[str, Any]:
    """
    Analyze speeds from detections JSONL file.
    
    Args:
        jsonl_path: Path to detections JSONL file
        homography_file: Path to homography file
        max_reasonable_speed: Maximum reasonable speed in mph for outlier detection
        
    Returns:
        Dictionary with speed analysis results
    """
    if not homography_file:
        logger.warning("No homography file provided, cannot calculate speeds")
        return {}
    
    # Read detections
    detections = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            det = json.loads(line.strip())
            detections.append(det)
    
    # Group by track_id
    tracks = {}
    for det in detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(det)
    
    # Sort each track by frame
    for track_id in tracks:
        tracks[track_id].sort(key=lambda x: x["frame"])
    
    # Initialize distance estimator
    distance_estimator = DistanceEstimator(homography_file)
    logger.info(f"Initialized DistanceEstimator with homography data")
    
    # Analyze speeds for each track
    all_raw_speeds = []
    track_stats = {}
    
    for track_id, track_detections in tracks.items():
        if len(track_detections) < 2:
            continue
            
        raw_speeds = []
        
        # Calculate raw speeds between consecutive frames
        for i in range(1, len(track_detections)):
            prev_det = track_detections[i-1]
            curr_det = track_detections[i]
            
            # Get normalized coordinates
            prev_center = prev_det["center"]
            curr_center = curr_det["center"]
            
            # Assume video dimensions (should be passed in ideally)
            # Get from bbox
            video_width = 1280  # Default, will be overridden if we can infer
            video_height = 720
            
            prev_x_norm = prev_center[0] / video_width
            prev_y_norm = prev_center[1] / video_height
            curr_x_norm = curr_center[0] / video_width
            curr_y_norm = curr_center[1] / video_height
            
            # Calculate distance in meters
            try:
                distance_meters = distance_estimator.estimate_distance(
                    (prev_x_norm, prev_y_norm), (curr_x_norm, curr_y_norm)
                )
                
                # Calculate time difference
                time_diff = curr_det["time"] - prev_det["time"]
                
                if time_diff > 0:
                    # Calculate speed in mph
                    speed_mps = distance_meters / time_diff
                    speed_mph = speed_mps * 2.23694
                    
                    raw_speeds.append(speed_mph)
                    all_raw_speeds.append(speed_mph)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
        
        if raw_speeds:
            track_stats[track_id] = {
                "raw_speeds": raw_speeds,
                "avg_speed": np.mean(raw_speeds),
                "median_speed": np.median(raw_speeds),
                "std_speed": np.std(raw_speeds),
                "max_speed": np.max(raw_speeds),
                "min_speed": np.min(raw_speeds),
                "outliers": sum(1 for s in raw_speeds if s > max_reasonable_speed),
            }
    
    # Overall statistics
    if all_raw_speeds:
        stats = {
            "total_tracks": len(tracks),
            "tracks_with_speed": len(track_stats),
            "total_speed_measurements": len(all_raw_speeds),
            "avg_speed": float(np.mean(all_raw_speeds)),
            "median_speed": float(np.median(all_raw_speeds)),
            "std_speed": float(np.std(all_raw_speeds)),
            "max_speed": float(np.max(all_raw_speeds)),
            "min_speed": float(np.min(all_raw_speeds)),
            "outliers_count": sum(1 for s in all_raw_speeds if s > max_reasonable_speed),
            "outliers_percent": 100 * sum(1 for s in all_raw_speeds if s > max_reasonable_speed) / len(all_raw_speeds),
            "track_stats": track_stats,
        }
        
        logger.info(f"Speed Analysis Results:")
        logger.info(f"  Total tracks: {stats['total_tracks']}")
        logger.info(f"  Tracks with speed: {stats['tracks_with_speed']}")
        logger.info(f"  Avg speed: {stats['avg_speed']:.2f} mph")
        logger.info(f"  Median speed: {stats['median_speed']:.2f} mph")
        logger.info(f"  Max speed: {stats['max_speed']:.2f} mph")
        logger.info(f"  Outliers (>{max_reasonable_speed} mph): {stats['outliers_count']} ({stats['outliers_percent']:.1f}%)")
        
        return stats
    
    return {}


def calculate_speeds_with_smoothing(
    jsonl_path: Path,
    homography_file: str,
    output_path: Path,
    video_width: int = 1280,
    video_height: int = 720,
    smoothing_method: str = "median_moving_average",
    smoothing_window: int = 5,
    max_reasonable_speed: float = 100.0,
    min_reasonable_speed: float = 0.0,
    lookback_frames: int = 5,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Calculate speeds from detections with advanced smoothing and outlier rejection.
    
    Args:
        jsonl_path: Path to detections JSONL file
        homography_file: Path to homography file
        output_path: Path for output JSONL with speeds
        video_width: Video width in pixels
        video_height: Video height in pixels
        smoothing_method: Smoothing method to use
        smoothing_window: Window size for smoothing
        max_reasonable_speed: Maximum reasonable speed for outlier rejection
        min_reasonable_speed: Minimum reasonable speed for outlier rejection
        lookback_frames: Number of frames to look back for speed calculation
        
    Returns:
        Tuple of (output_path, statistics)
    """
    logger.info(f"Calculating speeds with {smoothing_method} smoothing...")
    logger.info(f"  Window size: {smoothing_window}")
    logger.info(f"  Speed range: {min_reasonable_speed}-{max_reasonable_speed} mph")
    logger.info(f"  Lookback frames: {lookback_frames}")
    
    # Read detections
    detections = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            det = json.loads(line.strip())
            detections.append(det)
    
    # Group by track_id
    tracks = {}
    for det in detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(det)
    
    # Sort each track by frame
    for track_id in tracks:
        tracks[track_id].sort(key=lambda x: x["frame"])
    
    # Initialize distance estimator
    distance_estimator = DistanceEstimator(homography_file)
    
    # Initialize smoothers for different methods
    from collections import deque
    
    if smoothing_method == "median_moving_average":
        # Median filter + moving average
        speed_history = {}  # {track_id: deque of speeds}
        median_window = smoothing_window
    elif smoothing_method == "kalman_with_outlier_rejection":
        # Kalman filter with outlier rejection
        speed_smoother = SpeedSmoother(
            method="kalman",
            window_size=smoothing_window,
            kalman_q=0.5,
            kalman_r=5.0,  # Higher R = trust measurements less
        )
    elif smoothing_method == "exponential_with_outlier_rejection":
        # EMA with outlier rejection
        speed_smoother = SpeedSmoother(
            method="exponential",
            window_size=smoothing_window,
            ema_alpha=0.3,
        )
    elif smoothing_method == "moving_average":
        # Simple moving average
        speed_smoother = SpeedSmoother(
            method="moving_average",
            window_size=smoothing_window,
        )
    else:
        # No smoothing
        speed_smoother = SpeedSmoother(method="none")
    
    # Track statistics
    stats = {
        "total_detections": len(detections),
        "total_tracks": len(tracks),
        "speed_calculations": 0,
        "outliers_rejected": 0,
        "speeds_assigned": 0,
        "raw_speeds": [],
        "smoothed_speeds": [],
    }
    
    # Calculate speeds for each track
    vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
    vehicle_speeds = {}     # {track_id: current_speed_mph}
    
    if smoothing_method == "median_moving_average":
        speed_history = {}
    
    # Process detections in order
    detections_with_speed = []
    
    for det in detections:
        track_id = det.get("track_id")
        
        # Copy detection
        det_with_speed = det.copy()
        det_with_speed["speed_mph"] = None
        
        if track_id is None:
            detections_with_speed.append(det_with_speed)
            continue
        
        # Get normalized coordinates
        center = det["center"]
        x_norm = center[0] / video_width
        y_norm = center[1] / video_height
        timestamp = det["time"]
        frame = det["frame"]
        
        # Initialize tracking history
        if track_id not in vehicle_positions:
            vehicle_positions[track_id] = []
        
        vehicle_positions[track_id].append((frame, x_norm, y_norm, timestamp))
        
        # Keep only necessary history
        if len(vehicle_positions[track_id]) > lookback_frames * 2:
            vehicle_positions[track_id] = vehicle_positions[track_id][-lookback_frames * 2:]
        
        # Calculate speed if we have enough history
        history = vehicle_positions[track_id]
        if len(history) >= lookback_frames:
            old_frame, old_x, old_y, old_time = history[-lookback_frames]
            new_frame, new_x, new_y, new_time = history[-1]
            
            try:
                # Calculate distance
                distance_meters = distance_estimator.estimate_distance(
                    (old_x, old_y), (new_x, new_y)
                )
                
                # Calculate time difference
                time_diff = new_time - old_time
                
                if time_diff > 0:
                    # Calculate raw speed
                    speed_mps = distance_meters / time_diff
                    raw_speed_mph = speed_mps * 2.23694
                    
                    stats["speed_calculations"] += 1
                    stats["raw_speeds"].append(raw_speed_mph)
                    
                    # Apply outlier rejection and smoothing
                    if smoothing_method == "median_moving_average":
                        # Use median filter to remove outliers
                        if track_id not in speed_history:
                            speed_history[track_id] = deque(maxlen=median_window)
                        
                        # Add to history
                        speed_history[track_id].append(raw_speed_mph)
                        
                        # Calculate median of recent speeds
                        recent_speeds = list(speed_history[track_id])
                        if len(recent_speeds) >= 3:
                            # Use median to reduce outlier impact
                            median_speed = np.median(recent_speeds)
                            
                            # Check if current speed is an outlier
                            if abs(raw_speed_mph - median_speed) > max_reasonable_speed * 0.5:
                                # Reject outlier, use median instead
                                speed_mph = median_speed
                                stats["outliers_rejected"] += 1
                            else:
                                # Take moving average of recent speeds
                                speed_mph = np.mean(recent_speeds)
                        else:
                            speed_mph = raw_speed_mph
                        
                        # Final bounds check
                        if speed_mph < min_reasonable_speed or speed_mph > max_reasonable_speed:
                            if track_id in vehicle_speeds:
                                speed_mph = vehicle_speeds[track_id]  # Use previous
                            else:
                                speed_mph = np.clip(speed_mph, min_reasonable_speed, max_reasonable_speed)
                                stats["outliers_rejected"] += 1
                        
                    elif smoothing_method in ["kalman_with_outlier_rejection", "exponential_with_outlier_rejection"]:
                        # Reject outliers before smoothing
                        if raw_speed_mph < min_reasonable_speed or raw_speed_mph > max_reasonable_speed:
                            # Use previous speed or clip
                            if track_id in vehicle_speeds:
                                raw_speed_mph = vehicle_speeds[track_id]
                            else:
                                raw_speed_mph = np.clip(raw_speed_mph, min_reasonable_speed, max_reasonable_speed)
                            stats["outliers_rejected"] += 1
                        
                        # Apply smoothing
                        speed_mph = speed_smoother.smooth_speed(track_id, raw_speed_mph)
                    
                    else:
                        # Simple smoothing with bounds check
                        speed_mph = speed_smoother.smooth_speed(track_id, raw_speed_mph)
                        
                        if speed_mph < min_reasonable_speed or speed_mph > max_reasonable_speed:
                            speed_mph = np.clip(speed_mph, min_reasonable_speed, max_reasonable_speed)
                            stats["outliers_rejected"] += 1
                    
                    # Store speed
                    vehicle_speeds[track_id] = speed_mph
                    det_with_speed["speed_mph"] = float(speed_mph)
                    stats["speeds_assigned"] += 1
                    stats["smoothed_speeds"].append(speed_mph)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
        
        detections_with_speed.append(det_with_speed)
    
    # Write output
    with open(output_path, 'w') as f:
        for det in detections_with_speed:
            f.write(json.dumps(det) + "\n")
    
    # Calculate final statistics
    if stats["smoothed_speeds"]:
        stats["avg_raw_speed"] = float(np.mean(stats["raw_speeds"]))
        stats["avg_smoothed_speed"] = float(np.mean(stats["smoothed_speeds"]))
        stats["median_smoothed_speed"] = float(np.median(stats["smoothed_speeds"]))
        stats["max_smoothed_speed"] = float(np.max(stats["smoothed_speeds"]))
        stats["min_smoothed_speed"] = float(np.min(stats["smoothed_speeds"]))
        stats["outlier_rejection_rate"] = 100 * stats["outliers_rejected"] / stats["speed_calculations"]
    
    # Remove large arrays for cleaner output
    del stats["raw_speeds"]
    del stats["smoothed_speeds"]
    
    logger.info(f"Speed calculation complete:")
    logger.info(f"  Speeds assigned: {stats['speeds_assigned']}/{stats['total_detections']}")
    logger.info(f"  Outliers rejected: {stats['outliers_rejected']} ({stats.get('outlier_rejection_rate', 0):.1f}%)")
    logger.info(f"  Avg smoothed speed: {stats.get('avg_smoothed_speed', 0):.2f} mph")
    logger.info(f"  Median smoothed speed: {stats.get('median_smoothed_speed', 0):.2f} mph")
    logger.info(f"  Speed range: {stats.get('min_smoothed_speed', 0):.2f}-{stats.get('max_smoothed_speed', 0):.2f} mph")
    
    return output_path, stats


def main():
    """
    Main function to test speed calculation with different smoothing techniques.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video detections and calculate speeds")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to detections JSONL file")
    parser.add_argument("--homography", type=str, required=True, help="Path to homography JSON file")
    parser.add_argument("--output-dir", type=str, default="./speed_test_output", help="Output directory")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--video-height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--max-speed", type=float, default=100.0, help="Maximum reasonable speed in mph")
    parser.add_argument("--min-speed", type=float, default=0.0, help="Minimum reasonable speed in mph")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback frames for speed calculation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = Path(args.jsonl)
    homography_file = args.homography
    
    logger.info("="*80)
    logger.info("SPEED CALCULATION AND SMOOTHING ANALYSIS")
    logger.info("="*80)
    
    # Test different smoothing methods
    methods = [
        ("none", "no_smoothing"),
        ("moving_average", "moving_average"),
        ("exponential_with_outlier_rejection", "exponential_outlier"),
        ("kalman_with_outlier_rejection", "kalman_outlier"),
        ("median_moving_average", "median_ma"),  # Recommended
    ]
    
    results = {}
    
    for method, filename_suffix in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing method: {method}")
        logger.info(f"{'='*80}")
        
        output_path = output_dir / f"detections_{filename_suffix}.jsonl"
        
        try:
            _, stats = calculate_speeds_with_smoothing(
                jsonl_path=jsonl_path,
                homography_file=homography_file,
                output_path=output_path,
                video_width=args.video_width,
                video_height=args.video_height,
                smoothing_method=method,
                smoothing_window=5,
                max_reasonable_speed=args.max_speed,
                min_reasonable_speed=args.min_speed,
                lookback_frames=args.lookback,
            )
            
            results[method] = {
                "output_file": str(output_path),
                "stats": stats,
            }
            
        except Exception as e:
            logger.error(f"Failed to process with method {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON OF SMOOTHING METHODS")
    logger.info(f"{'='*80}")
    
    logger.info(f"\n{'Method':<35} {'Avg Speed':>12} {'Median':>12} {'Max':>12} {'Outliers':>12}")
    logger.info("-" * 85)
    
    for method, result in results.items():
        stats = result["stats"]
        avg_speed = stats.get("avg_smoothed_speed", 0)
        median_speed = stats.get("median_smoothed_speed", 0)
        max_speed = stats.get("max_smoothed_speed", 0)
        outliers = stats.get("outliers_rejected", 0)
        
        logger.info(f"{method:<35} {avg_speed:>12.2f} {median_speed:>12.2f} {max_speed:>12.2f} {outliers:>12}")
    
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATION: Use 'median_moving_average' for best outlier rejection")
    logger.info(f"{'='*80}")
    
    # Save results summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
