"""
Bounding Box Smoother Implementation

Provides various smoothing algorithms for bounding box detections to reduce jitter
and improve tracking stability for speed estimation.
"""

import numpy as np
from collections import deque
from typing import Dict
import supervision as sv


class BboxSmoother:
    """
    Handles smoothing of bounding box detections using various algorithms.

    This class maintains state for each tracked object and applies smoothing
    algorithms to reduce jitter in bounding box coordinates.
    """

    def __init__(
        self,
        method: str = "kalman",
        window_size: int = 5,
        ema_alpha: float = 0.3,
        kalman_q: float = 0.5,
        kalman_r: float = 3.0,
    ):
        """
        Initialize the bbox smoother.

        Args:
            method: Smoothing method ('none', 'moving_average', 'exponential', 'kalman', 'iou_weighted')
            window_size: Window size for moving_average and iou_weighted methods
            ema_alpha: Alpha parameter for exponential moving average (0.0-1.0)
            kalman_q: Process noise for Kalman filter (lower = smoother)
            kalman_r: Measurement noise for Kalman filter (higher = trust measurements less)
        """
        self.method = method
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r

        # State storage for different methods
        self.bbox_history: Dict[int, deque] = {}  # For moving_average and iou_weighted
        self.ema_bboxes: Dict[int, np.ndarray] = {}  # For exponential
        self.kalman_bbox_states: Dict[int, Dict] = {}  # For kalman

    def smooth_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Apply smoothing to bounding boxes in detections.

        Args:
            detections: Supervision detections with tracker_id

        Returns:
            Detections with smoothed bounding boxes
        """
        if detections.tracker_id is None or len(detections) == 0:
            return detections

        if self.method == "none":
            return detections

        # Create a copy of the xyxy array to avoid modifying the original
        smoothed_xyxy = detections.xyxy.copy()

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is not None:
                raw_bbox = detections.xyxy[detection_idx]  # [x1, y1, x2, y2]
                smoothed_bbox = self._smooth_bbox(tracker_id, raw_bbox)
                smoothed_xyxy[detection_idx] = smoothed_bbox

        # Create new detections object with smoothed bboxes
        smoothed_detections = sv.Detections(
            xyxy=smoothed_xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=detections.tracker_id,
        )

        # Copy any additional data
        if hasattr(detections, "data") and detections.data is not None:
            smoothed_detections.data = detections.data

        return smoothed_detections

    def _smooth_bbox(self, tracker_id: int, raw_bbox: np.ndarray) -> np.ndarray:
        """
        Apply smoothing algorithm to a single bounding box.

        Args:
            tracker_id: Unique tracker ID for this object
            raw_bbox: Raw bounding box [x1, y1, x2, y2]

        Returns:
            Smoothed bounding box [x1, y1, x2, y2]
        """
        if self.method == "moving_average":
            return self._moving_average_smooth(tracker_id, raw_bbox)
        elif self.method == "exponential":
            return self._exponential_smooth(tracker_id, raw_bbox)
        elif self.method == "kalman":
            return self._kalman_smooth(tracker_id, raw_bbox)
        elif self.method == "iou_weighted":
            return self._iou_weighted_smooth(tracker_id, raw_bbox)
        else:
            return raw_bbox

    def _moving_average_smooth(
        self, tracker_id: int, raw_bbox: np.ndarray
    ) -> np.ndarray:
        """Apply moving average smoothing."""
        # Ensure raw_bbox is a numpy array
        raw_bbox = np.array(raw_bbox, dtype=np.float64)
        
        if tracker_id not in self.bbox_history:
            self.bbox_history[tracker_id] = deque(maxlen=self.window_size)

        self.bbox_history[tracker_id].append(raw_bbox)

        # Average all bboxes in history
        return np.mean(np.array(self.bbox_history[tracker_id]), axis=0)

    def _exponential_smooth(self, tracker_id: int, raw_bbox: np.ndarray) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        # Ensure raw_bbox is a numpy array
        raw_bbox = np.array(raw_bbox, dtype=np.float64)
        
        if tracker_id not in self.ema_bboxes:
            self.ema_bboxes[tracker_id] = raw_bbox
        else:
            self.ema_bboxes[tracker_id] = (
                self.ema_alpha * raw_bbox
                + (1 - self.ema_alpha) * self.ema_bboxes[tracker_id]
            )
        return self.ema_bboxes[tracker_id]

    def _kalman_smooth(self, tracker_id: int, raw_bbox: np.ndarray) -> np.ndarray:
        """Apply Kalman filter smoothing."""
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
        P_pred = state["P"] + np.eye(4) * self.kalman_q

        # Update step
        identity_matrix = np.eye(4)
        K = P_pred @ np.linalg.inv(P_pred + np.eye(4) * self.kalman_r)  # Kalman gain
        x_new = x_pred + K @ (raw_bbox - x_pred)
        P_new = (identity_matrix - K) @ P_pred

        # Store updated state
        self.kalman_bbox_states[tracker_id] = {"x": x_new, "P": P_new}

        return x_new

    def _iou_weighted_smooth(self, tracker_id: int, raw_bbox: np.ndarray) -> np.ndarray:
        """Apply IOU-weighted smoothing."""
        # Ensure raw_bbox is a numpy array
        raw_bbox = np.array(raw_bbox, dtype=np.float64)
        
        if tracker_id not in self.bbox_history:
            self.bbox_history[tracker_id] = deque(maxlen=self.window_size)

        self.bbox_history[tracker_id].append(raw_bbox)

        # Calculate IOU weights
        if len(self.bbox_history[tracker_id]) > 1:
            weights = []
            bboxes = np.array(self.bbox_history[tracker_id])

            # Use most recent bbox as reference
            ref_bbox = bboxes[-1]

            for bbox in bboxes:
                iou = self._calculate_iou(ref_bbox, bbox)
                weights.append(iou)

            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize

            # Weighted average
            return np.average(bboxes, axis=0, weights=weights)
        else:
            return raw_bbox

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
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

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def reset(self):
        """Reset all internal state."""
        self.bbox_history.clear()
        self.ema_bboxes.clear()
        self.kalman_bbox_states.clear()

    def get_method_info(self) -> Dict[str, str]:
        """Get information about the current smoothing method."""
        info = {
            "method": self.method,
            "description": self._get_method_description(),
            "parameters": self._get_method_parameters(),
        }
        return info

    def _get_method_description(self) -> str:
        """Get description of current method."""
        descriptions = {
            "none": "No smoothing applied (baseline)",
            "moving_average": f"Simple moving average over {self.window_size} frames",
            "exponential": f"Exponential moving average with alpha={self.ema_alpha}",
            "kalman": f"Kalman filter with Q={self.kalman_q}, R={self.kalman_r}",
            "iou_weighted": f"IOU-weighted average over {self.window_size} frames",
        }
        return descriptions.get(self.method, "Unknown method")

    def _get_method_parameters(self) -> str:
        """Get current method parameters."""
        if self.method == "moving_average" or self.method == "iou_weighted":
            return f"window_size={self.window_size}"
        elif self.method == "exponential":
            return f"ema_alpha={self.ema_alpha}"
        elif self.method == "kalman":
            return f"kalman_q={self.kalman_q}, kalman_r={self.kalman_r}"
        else:
            return "No parameters"
