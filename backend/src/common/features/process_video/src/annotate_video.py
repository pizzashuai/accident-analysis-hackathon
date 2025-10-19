import cv2
import supervision as sv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json
from collections import deque

try:
    # Try relative imports first (when running as a module)
    from .persist_detections import (
        read_detections_from_jsonl,
        get_detections_by_frame,
        convert_jsonl_to_supervision_detections,
    )
    from .estimate_distance import DistanceEstimator
except ImportError:
    # Fall back to direct imports (when running directly or from sys.path)
    from persist_detections import (
        read_detections_from_jsonl,
        get_detections_by_frame,
        convert_jsonl_to_supervision_detections,
    )
    from estimate_distance import DistanceEstimator


class VideoAnnotator:
    """Handles video annotation with detections from various sources."""

    def __init__(
        self,
        trail_length: int = 10,
        box_color: Optional[sv.Color] = None,
        text_color: Optional[sv.Color] = None,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        box_thickness: int = 2,
        homography_file: Optional[str] = None,
        speed_smoothing: str = "moving_average",  # Options: "none", "moving_average", "exponential", "kalman"
        smoothing_window: int = 5,  # Window size for moving average
        tracking_point: str = "center",  # Options: "center", "bottom_center"
        debug_speed: bool = False,  # Enable detailed speed debug output
        debug_jsonl_path: Optional[Path] = None,  # Path to write debug data
        bbox_smoothing: str = "exponential",  # Options: "none", "moving_average", "exponential", "kalman", "iou_weighted"
        bbox_smoothing_window: int = 5,  # Window size for bbox smoothing
    ):
        """
        Initialize video annotator.

        Args:
            trail_length: Length of tracking trails
            box_color: Color for bounding boxes
            text_color: Color for text labels
            text_scale: Scale for text labels
            text_thickness: Thickness for text labels
            box_thickness: Thickness for bounding boxes
            homography_file: Path to homography file for speed calculation
            speed_smoothing: Smoothing algorithm for speed ("none", "moving_average", "exponential", "kalman")
            smoothing_window: Window size for smoothing algorithms
            tracking_point: Point to track on vehicle ("center", "bottom_center")
            debug_speed: Enable detailed speed calculation debug output
            debug_jsonl_path: Path to write detailed debug data
            bbox_smoothing: Smoothing algorithm for bounding boxes ("none", "moving_average", "exponential", "kalman", "iou_weighted")
            bbox_smoothing_window: Window size for bbox smoothing algorithms
        """
        self.trail_length = trail_length
        self.box_color = box_color or sv.Color.WHITE
        self.text_color = text_color or sv.Color.BLACK
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.box_thickness = box_thickness
        self.speed_smoothing = speed_smoothing
        self.smoothing_window = smoothing_window
        self.tracking_point = tracking_point
        self.debug_speed = debug_speed
        self.debug_jsonl_path = debug_jsonl_path
        self.bbox_smoothing = bbox_smoothing
        self.bbox_smoothing_window = bbox_smoothing_window

        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(
            color=self.box_color, thickness=self.box_thickness
        )
        self.label_annotator = sv.LabelAnnotator(
            color=self.text_color,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
        )

        # Tracking history for trails
        self.tracking_history = {}

        # Speed tracking
        self.speed_tracker = {}  # {tracker_id: [(frame_num, x_norm, y_norm, timestamp), ...]}
        self.current_speeds = {}  # {tracker_id: speed_mph}
        self.raw_speeds = {}  # {tracker_id: deque of raw speed values for smoothing}

        # Exponential smoothing state
        self.ema_speeds = {}  # {tracker_id: exponential moving average}
        self.ema_alpha = 0.3  # Smoothing factor for exponential moving average

        # Kalman filter state (simplified 1D for speed)
        self.kalman_states = {}  # {tracker_id: {"x": speed, "P": variance}}

        # Bounding box smoothing state
        self.bbox_history = {}  # {tracker_id: deque of bboxes for moving average}
        self.ema_bboxes = {}  # {tracker_id: exponential moving average bbox}
        self.bbox_ema_alpha = (
            0.3  # Smoothing factor for bbox exponential moving average
        )
        self.kalman_bbox_states = {}  # {tracker_id: {"x": [x1,y1,x2,y2], "P": covariance matrix}}

        # Debug data collection
        self.debug_data = []  # List to collect all debug records

        # Distance estimator for speed calculation
        self.distance_estimator = None
        if homography_file:
            try:
                self.distance_estimator = DistanceEstimator(homography_file)
            except Exception as e:
                print(f"Warning: Could not initialize distance estimator: {e}")
                print("Speed calculation will be disabled.")

        # Video properties (set during annotation)
        self.video_width = None
        self.video_height = None
        self.video_fps = None

    def annotate_video_from_detections(
        self,
        video_path: Union[str, Path],
        detections: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> Path:
        """
        Annotate video using detection data from JSONL format.

        Args:
            video_path: Path to input video
            detections: List of detection dictionaries
            output_path: Path for output video (default: input_video_annotated.mp4)
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Path to the annotated video
        """
        video_path = Path(video_path)

        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
        else:
            output_path = Path(output_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Store video properties for speed calculation
        self.video_width = width
        self.video_height = height
        self.video_fps = fps

        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Annotating video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Total frames: {total_frames}")

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame
            frame_detections = get_detections_by_frame(detections, frame_count)

            if frame_detections:
                # Convert to supervision format
                sv_detections = convert_jsonl_to_supervision_detections(
                    frame_detections
                )

                # Annotate frame
                annotated_frame = self._get_annotated_frame(
                    frame,
                    sv_detections,
                    show_trails,
                    show_labels,
                    show_boxes,
                    frame_count,
                )
            else:
                annotated_frame = frame

            # Write frame
            out.write(annotated_frame)
            processed_frames += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

            frame_count += 1

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Annotation complete! Processed {processed_frames} frames")
        print(f"Annotated video saved to: {output_path}")

        # Write debug data if enabled
        self._write_debug_data()

        return output_path

    def annotate_video_from_jsonl(
        self,
        original_video_path: Union[str, Path],
        jsonl_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> Path:
        """
        Annotate video using detection data from JSONL file.

        Args:
            video_path: Path to input video
            jsonl_path: Path to JSONL file with detections
            output_path: Path for output video (default: input_video_annotated.mp4)
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Path to the annotated video
        """
        # Read detections from JSONL
        detections = read_detections_from_jsonl(Path(jsonl_path))

        return self.annotate_video_from_detections(
            original_video_path,
            detections,
            output_path,
            show_trails,
            show_labels,
            show_boxes,
        )

    def annotate_video_from_supervision_detections(
        self,
        original_video_path: Union[str, Path],
        detections_list: List[sv.Detections],
        output_path: Optional[Union[str, Path]] = None,
        show_trails: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> Path:
        """
        Annotate video using supervision Detections objects.

        Args:
            original_video_path: Path to input video
            detections_list: List of supervision Detections objects (one per frame)
            output_path: Path for output video (default: input_video_annotated.mp4)
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes

        Returns:
            Path to the annotated video
        """
        video_path = Path(original_video_path)

        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
        else:
            output_path = Path(output_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Store video properties for speed calculation
        self.video_width = width
        self.video_height = height
        self.video_fps = fps

        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Annotating video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Total frames: {total_frames}")

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame
            if frame_count < len(detections_list):
                detections = detections_list[frame_count]

                # Annotate frame
                annotated_frame = self._get_annotated_frame(
                    frame, detections, show_trails, show_labels, show_boxes, frame_count
                )
            else:
                annotated_frame = frame

            # Write frame
            out.write(annotated_frame)
            processed_frames += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

            frame_count += 1

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Annotation complete! Processed {processed_frames} frames")
        print(f"Annotated video saved to: {output_path}")

        # Write debug data if enabled
        self._write_debug_data()

        return output_path

    def _get_annotated_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        show_trails: bool,
        show_labels: bool,
        show_boxes: bool,
        frame_count: int = 0,
    ) -> np.ndarray:
        """
        Annotate a single frame with detections.

        Args:
            frame: Input frame
            detections: Supervision detections
            show_trails: Whether to show tracking trails
            show_labels: Whether to show labels
            show_boxes: Whether to show bounding boxes
            frame_count: Current frame number for speed calculation

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        if len(detections) == 0:
            return annotated_frame

        # Apply bbox smoothing if enabled
        if detections.tracker_id is not None and self.bbox_smoothing != "none":
            detections = self._apply_bbox_smoothing(detections)

        # Update tracking history for trails
        if show_trails and detections.tracker_id is not None:
            self._update_tracking_history(detections)

        # Update speed tracking
        if detections.tracker_id is not None and self.distance_estimator is not None:
            self._update_speed_tracking(detections, frame_count)

        # Draw bounding boxes
        if show_boxes:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        # Draw labels
        if show_labels:
            labels = self._create_labels(detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

        # Draw trails
        if show_trails:
            annotated_frame = self._draw_trails(annotated_frame)

        return annotated_frame

    def _update_tracking_history(self, detections: sv.Detections) -> None:
        """Update tracking history for trail drawing."""
        if detections.tracker_id is None:
            return

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is not None:
                if tracker_id not in self.tracking_history:
                    self.tracking_history[tracker_id] = []

                # Get center point of bounding box
                x1, y1, x2, y2 = detections.xyxy[detection_idx]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                self.tracking_history[tracker_id].append((center_x, center_y))

                # Keep only the last trail_length points
                if len(self.tracking_history[tracker_id]) > self.trail_length:
                    self.tracking_history[tracker_id] = self.tracking_history[
                        tracker_id
                    ][-self.trail_length :]

    def _get_tracking_point(self, bbox: Union[np.ndarray, List]) -> tuple:
        """
        Get the tracking point from bounding box based on configuration.

        Args:
            bbox: Bounding box as [x1, y1, x2, y2]

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        x1, y1, x2, y2 = bbox

        if self.tracking_point == "bottom_center":
            # Use bottom center (contact point with ground)
            return ((x1 + x2) / 2, y2)
        else:  # "center" (default)
            # Use center of bounding box
            return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _apply_bbox_smoothing(self, detections: sv.Detections) -> sv.Detections:
        """
        Apply smoothing to bounding boxes to reduce jitter.

        Args:
            detections: Supervision detections with tracker_id

        Returns:
            Detections with smoothed bounding boxes
        """
        if detections.tracker_id is None or len(detections) == 0:
            return detections

        # Create a copy of the xyxy array to avoid modifying the original
        smoothed_xyxy = detections.xyxy.copy()

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is not None:
                raw_bbox = detections.xyxy[detection_idx]  # [x1, y1, x2, y2]

                if self.bbox_smoothing == "moving_average":
                    # Moving average smoothing
                    if tracker_id not in self.bbox_history:
                        self.bbox_history[tracker_id] = deque(
                            maxlen=self.bbox_smoothing_window
                        )

                    self.bbox_history[tracker_id].append(raw_bbox)

                    # Average all bboxes in history
                    smoothed_bbox = np.mean(
                        np.array(self.bbox_history[tracker_id]), axis=0
                    )
                    smoothed_xyxy[detection_idx] = smoothed_bbox

                elif self.bbox_smoothing == "exponential":
                    # Exponential moving average
                    if tracker_id not in self.ema_bboxes:
                        self.ema_bboxes[tracker_id] = raw_bbox
                    else:
                        self.ema_bboxes[tracker_id] = (
                            self.bbox_ema_alpha * raw_bbox
                            + (1 - self.bbox_ema_alpha) * self.ema_bboxes[tracker_id]
                        )
                    smoothed_xyxy[detection_idx] = self.ema_bboxes[tracker_id]

                elif self.bbox_smoothing == "kalman":
                    # Simplified Kalman filter for 4D bbox (x1, y1, x2, y2)
                    Q = 0.5  # Process noise (lower = smoother, higher = more responsive)
                    R = 3.0  # Measurement noise (higher = trust measurements less)

                    # Ensure raw_bbox is a numpy array
                    raw_bbox = np.array(raw_bbox, dtype=np.float64)

                    if tracker_id not in self.kalman_bbox_states:
                        # Initialize
                        self.kalman_bbox_states[tracker_id] = {
                            "x": raw_bbox.copy(),
                            "P": np.eye(4) * 1.0,  # Initial covariance
                        }

                    state = self.kalman_bbox_states[tracker_id]

                    # Prediction step (assuming constant bbox)
                    x_pred = state["x"]
                    P_pred = state["P"] + np.eye(4) * Q

                    # Update step
                    identity_matrix = np.eye(4)
                    K = P_pred @ np.linalg.inv(P_pred + np.eye(4) * R)  # Kalman gain
                    x_new = x_pred + K @ (raw_bbox - x_pred)
                    P_new = (identity_matrix - K) @ P_pred

                    # Store updated state
                    self.kalman_bbox_states[tracker_id] = {"x": x_new, "P": P_new}
                    smoothed_xyxy[detection_idx] = x_new

                elif self.bbox_smoothing == "iou_weighted":
                    # IOU-weighted smoothing (gives more weight to similar-sized boxes)
                    if tracker_id not in self.bbox_history:
                        self.bbox_history[tracker_id] = deque(
                            maxlen=self.bbox_smoothing_window
                        )

                    self.bbox_history[tracker_id].append(raw_bbox)

                    # Calculate IOU weights
                    if len(self.bbox_history[tracker_id]) > 1:
                        weights = []
                        bboxes = np.array(self.bbox_history[tracker_id])

                        # Use most recent bbox as reference
                        ref_bbox = bboxes[-1]

                        for bbox in bboxes:
                            # Simple IOU-like metric based on size similarity
                            ref_area = (ref_bbox[2] - ref_bbox[0]) * (
                                ref_bbox[3] - ref_bbox[1]
                            )
                            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            size_ratio = min(ref_area, bbox_area) / max(
                                ref_area, bbox_area
                            )
                            weights.append(size_ratio)

                        # Normalize weights
                        weights = np.array(weights)
                        weights = weights / weights.sum()

                        # Weighted average
                        smoothed_bbox = np.average(bboxes, axis=0, weights=weights)
                        smoothed_xyxy[detection_idx] = smoothed_bbox
                    else:
                        smoothed_xyxy[detection_idx] = raw_bbox

        # Create new detections object with smoothed bboxes
        smoothed_detections = sv.Detections(
            xyxy=smoothed_xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            tracker_id=detections.tracker_id,
        )
        # Copy any additional data
        if hasattr(detections, "data") and detections.data is not None:
            smoothed_detections.data = detections.data

        return smoothed_detections

    def _apply_smoothing(self, tracker_id: int, raw_speed: float) -> float:
        """
        Apply smoothing algorithm to raw speed value.

        Args:
            tracker_id: Tracker ID
            raw_speed: Raw calculated speed in mph

        Returns:
            Smoothed speed in mph
        """
        if self.speed_smoothing == "none":
            return raw_speed

        elif self.speed_smoothing == "moving_average":
            # Initialize deque for this tracker if needed
            if tracker_id not in self.raw_speeds:
                self.raw_speeds[tracker_id] = deque(maxlen=self.smoothing_window)

            # Add new speed
            self.raw_speeds[tracker_id].append(raw_speed)

            # Return average
            return sum(self.raw_speeds[tracker_id]) / len(self.raw_speeds[tracker_id])

        elif self.speed_smoothing == "exponential":
            # Exponential moving average
            if tracker_id not in self.ema_speeds:
                self.ema_speeds[tracker_id] = raw_speed
            else:
                self.ema_speeds[tracker_id] = (
                    self.ema_alpha * raw_speed
                    + (1 - self.ema_alpha) * self.ema_speeds[tracker_id]
                )
            return self.ema_speeds[tracker_id]

        elif self.speed_smoothing == "kalman":
            # Simplified 1D Kalman filter
            # Process noise and measurement noise
            Q = 0.1  # Process noise covariance
            R = 2.0  # Measurement noise covariance

            if tracker_id not in self.kalman_states:
                # Initialize
                self.kalman_states[tracker_id] = {"x": raw_speed, "P": 1.0}

            state = self.kalman_states[tracker_id]

            # Prediction step (assuming constant speed)
            x_pred = state["x"]
            P_pred = state["P"] + Q

            # Update step
            K = P_pred / (P_pred + R)  # Kalman gain
            x_new = x_pred + K * (raw_speed - x_pred)
            P_new = (1 - K) * P_pred

            # Store updated state
            self.kalman_states[tracker_id] = {"x": x_new, "P": P_new}

            return x_new

        return raw_speed

    def _update_speed_tracking(
        self, detections: sv.Detections, frame_count: int
    ) -> None:
        """Update speed tracking and calculate speeds with detailed debug info."""
        if (
            detections.tracker_id is None
            or self.video_width is None
            or self.video_fps is None
        ):
            return

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is not None:
                # Get bounding box
                x1, y1, x2, y2 = detections.xyxy[detection_idx]

                # Get tracking point based on configuration
                track_x, track_y = self._get_tracking_point([x1, y1, x2, y2])

                # Convert to normalized coordinates
                x_norm = track_x / self.video_width
                y_norm = track_y / self.video_height

                # Calculate timestamp
                timestamp = frame_count / self.video_fps

                # Initialize tracker history if needed
                if tracker_id not in self.speed_tracker:
                    self.speed_tracker[tracker_id] = []

                # Add current position
                self.speed_tracker[tracker_id].append(
                    (frame_count, x_norm, y_norm, timestamp)
                )

                # Calculate speed if we have enough history
                # Use last 5 frames for smoother speed calculation
                history = self.speed_tracker[tracker_id]
                if len(history) >= 5 and self.distance_estimator is not None:
                    # Keep only last 30 frames of history
                    if len(history) > 30:
                        self.speed_tracker[tracker_id] = history[-30:]
                        history = self.speed_tracker[tracker_id]

                    # Calculate speed using last 5 frames
                    old_frame, old_x, old_y, old_time = history[-5]
                    new_frame, new_x, new_y, new_time = history[-1]

                    # Calculate distance using homography
                    distance_meters = self.distance_estimator.estimate_distance(
                        (old_x, old_y), (new_x, new_y)
                    )

                    # Transform to geo coordinates for debugging
                    old_geo = self.distance_estimator.image_to_geo(old_x, old_y)
                    new_geo = self.distance_estimator.image_to_geo(new_x, new_y)

                    # Calculate time difference
                    time_diff = new_time - old_time

                    if time_diff > 0:
                        # Calculate speed in meters per second
                        speed_mps = distance_meters / time_diff

                        # Convert to miles per hour (1 m/s = 2.23694 mph)
                        raw_speed_mph = speed_mps * 2.23694

                        # Apply smoothing
                        smoothed_speed_mph = self._apply_smoothing(
                            tracker_id, raw_speed_mph
                        )

                        # Store the smoothed speed
                        self.current_speeds[tracker_id] = smoothed_speed_mph

                        # Collect debug data
                        if self.debug_speed:
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1

                            debug_record = {
                                "frame": frame_count,
                                "time": round(timestamp, 3),
                                "track_id": int(tracker_id),
                                "bbox_xyxy": [
                                    float(x1),
                                    float(y1),
                                    float(x2),
                                    float(y2),
                                ],
                                "bbox_size": {
                                    "width": float(bbox_width),
                                    "height": float(bbox_height),
                                },
                                "tracking_point": self.tracking_point,
                                "track_point_pixel": {
                                    "x": float(track_x),
                                    "y": float(track_y),
                                },
                                "track_point_norm": {
                                    "x": float(x_norm),
                                    "y": float(y_norm),
                                },
                                "speed_calc": {
                                    "frames_used": 5,
                                    "old_frame": int(old_frame),
                                    "new_frame": int(new_frame),
                                    "old_point_norm": {
                                        "x": float(old_x),
                                        "y": float(old_y),
                                    },
                                    "new_point_norm": {
                                        "x": float(new_x),
                                        "y": float(new_y),
                                    },
                                    "old_geo": {
                                        "lat": float(old_geo.lat),
                                        "lng": float(old_geo.lng),
                                    },
                                    "new_geo": {
                                        "lat": float(new_geo.lat),
                                        "lng": float(new_geo.lng),
                                    },
                                    "distance_meters": float(distance_meters),
                                    "time_diff": float(time_diff),
                                    "speed_mps": float(speed_mps),
                                    "raw_speed_mph": float(raw_speed_mph),
                                    "smoothed_speed_mph": float(smoothed_speed_mph),
                                    "smoothing_method": self.speed_smoothing,
                                },
                            }
                            self.debug_data.append(debug_record)

    def _create_labels(self, detections: sv.Detections) -> List[str]:
        """Create labels for detections."""
        labels = []
        if detections.tracker_id is not None:
            for detection_idx in range(len(detections)):
                tracker_id = detections.tracker_id[detection_idx]
                if tracker_id is not None:
                    label = f"ID:{tracker_id}"

                    # Add speed if available
                    if tracker_id in self.current_speeds:
                        speed_mph = self.current_speeds[tracker_id]
                        label += f" | {speed_mph:.1f} mph"

                    labels.append(label)
                else:
                    labels.append("")
        else:
            labels = [""] * len(detections)
        return labels

    def _draw_trails(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking trails on frame."""
        for tracker_id, trace_points in self.tracking_history.items():
            if len(trace_points) > 1:
                # Get color for this tracker ID
                color = sv.ColorPalette.DEFAULT.by_idx(tracker_id % 20)
                color_bgr = (int(color.b), int(color.g), int(color.r))

                # Draw lines connecting the trail points
                for i in range(1, len(trace_points)):
                    pt1 = trace_points[i - 1]
                    pt2 = trace_points[i]
                    cv2.line(frame, pt1, pt2, color_bgr, 2)

        return frame

    def _write_debug_data(self) -> None:
        """Write collected debug data to JSONL file."""
        if self.debug_jsonl_path and self.debug_data:
            print(
                f"Writing {len(self.debug_data)} debug records to {self.debug_jsonl_path}"
            )
            with open(self.debug_jsonl_path, "w", encoding="utf-8") as f:
                for record in self.debug_data:
                    f.write(json.dumps(record) + "\n")
            print("Debug data written successfully!")


def annotate_video(
    video_path: Union[str, Path],
    detections_source: Union[str, Path, List[Dict[str, Any]], List[sv.Detections]],
    output_path: Optional[Union[str, Path]] = None,
    trail_length: int = 10,
    show_trails: bool = True,
    show_labels: bool = True,
    show_boxes: bool = True,
    homography_file: Optional[str] = None,
    speed_smoothing: str = "moving_average",
    smoothing_window: int = 5,
    tracking_point: str = "center",
    debug_speed: bool = False,
    debug_jsonl_path: Optional[Path] = None,
    bbox_smoothing: str = "exponential",
    bbox_smoothing_window: int = 5,
) -> Path:
    """
    Convenience function to annotate video with detections.

    Args:
        video_path: Path to input video
        detections_source: Source of detections (JSONL path, detection list, or supervision detections)
        output_path: Path for output video (default: input_video_annotated.mp4)
        trail_length: Length of tracking trails
        show_trails: Whether to show tracking trails
        show_labels: Whether to show labels
        show_boxes: Whether to show bounding boxes
        homography_file: Path to homography file for speed calculation
        speed_smoothing: Smoothing algorithm for speed ("none", "moving_average", "exponential", "kalman")
        smoothing_window: Window size for smoothing algorithms
        tracking_point: Point to track on vehicle ("center", "bottom_center")
        debug_speed: Enable detailed speed calculation debug output
        debug_jsonl_path: Path to write detailed debug data
        bbox_smoothing: Smoothing algorithm for bounding boxes ("none", "moving_average", "exponential", "kalman", "iou_weighted")
        bbox_smoothing_window: Window size for bbox smoothing algorithms

    Returns:
        Path to the annotated video
    """
    annotator = VideoAnnotator(
        trail_length=trail_length,
        homography_file=homography_file,
        speed_smoothing=speed_smoothing,
        smoothing_window=smoothing_window,
        tracking_point=tracking_point,
        debug_speed=debug_speed,
        debug_jsonl_path=debug_jsonl_path,
        bbox_smoothing=bbox_smoothing,
        bbox_smoothing_window=bbox_smoothing_window,
    )

    # Determine detections source type
    if isinstance(detections_source, (str, Path)):
        # Assume it's a JSONL file path
        return annotator.annotate_video_from_jsonl(
            video_path,
            detections_source,
            output_path,
            show_trails,
            show_labels,
            show_boxes,
        )
    elif (
        isinstance(detections_source, list)
        and len(detections_source) > 0
        and detections_source[0] is not None
    ):
        if isinstance(detections_source[0], dict):
            # List of detection dictionaries
            from typing import cast

            detections_dict_list = cast(List[Dict[str, Any]], detections_source)
            return annotator.annotate_video_from_detections(
                video_path,
                detections_dict_list,
                output_path,
                show_trails,
                show_labels,
                show_boxes,
            )
        elif isinstance(detections_source[0], sv.Detections):
            # List of supervision Detections
            from typing import cast

            detections_sv_list = cast(List[sv.Detections], detections_source)
            return annotator.annotate_video_from_supervision_detections(
                video_path,
                detections_sv_list,
                output_path,
                show_trails,
                show_labels,
                show_boxes,
            )

    raise ValueError("Invalid detections source type")


if __name__ == "__main__":
    video_path = Path(
        "/Users/shuaima/code/accident_analysis/accident-analysis-hackathon/happy1.mp4"
    )
    detections_path = Path(
        "/Users/shuaima/code/accident_analysis/accident-analysis-hackathon/out/test_37/detections-redone.jsonl"
    )
    output_path = Path(
        "/Users/shuaima/code/accident_analysis/accident-analysis-hackathon/out/test_37/detections-redone_with-speed_annotated.mp4"
    )
    homography_file = "/Users/shuaima/code/accident_analysis/accident-analysis-hackathon/homography-points.json"

    print("=" * 70)
    print("Vehicle Speed Annotation")
    print("=" * 70)
    print(f"Video: {video_path.name}")
    print(f"Detections: {detections_path.name}")
    print("Homography: enabled (speed calculation active)")
    print(f"Output: {output_path.name}")
    print("=" * 70)
    print()

    annotate_video(
        video_path, detections_path, output_path, homography_file=homography_file
    )
