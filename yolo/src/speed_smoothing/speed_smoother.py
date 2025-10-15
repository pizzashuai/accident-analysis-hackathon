"""
Speed Smoother Implementation

Provides various smoothing algorithms for speed estimates to reduce noise
and improve accuracy in vehicle speed tracking.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class SpeedSmoother:
    """
    Handles smoothing of speed estimates using various algorithms.

    This class maintains state for each tracked object and applies smoothing
    algorithms to reduce noise in speed calculations.
    """

    def __init__(
        self,
        method: str = "kalman",
        window_size: int = 5,
        ema_alpha: float = 0.3,
        kalman_q: float = 0.1,
        kalman_r: float = 2.0,
    ):
        """
        Initialize the speed smoother.

        Args:
            method: Smoothing method ('none', 'moving_average', 'exponential', 'kalman')
            window_size: Window size for moving_average method
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
        self.raw_speeds: Dict[int, deque] = {}  # For moving_average
        self.ema_speeds: Dict[int, float] = {}  # For exponential
        self.kalman_states: Dict[int, Dict] = {}  # For kalman

    def smooth_speed(self, tracker_id: int, raw_speed: float) -> float:
        """
        Apply smoothing algorithm to raw speed value.

        Args:
            tracker_id: Unique tracker ID for this object
            raw_speed: Raw calculated speed in mph

        Returns:
            Smoothed speed in mph
        """
        if self.method == "none":
            return raw_speed
        elif self.method == "moving_average":
            return self._moving_average_smooth(tracker_id, raw_speed)
        elif self.method == "exponential":
            return self._exponential_smooth(tracker_id, raw_speed)
        elif self.method == "kalman":
            return self._kalman_smooth(tracker_id, raw_speed)
        else:
            return raw_speed

    def _moving_average_smooth(self, tracker_id: int, raw_speed: float) -> float:
        """Apply moving average smoothing."""
        # Initialize deque for this tracker if needed
        if tracker_id not in self.raw_speeds:
            self.raw_speeds[tracker_id] = deque(maxlen=self.window_size)

        # Add new speed
        self.raw_speeds[tracker_id].append(raw_speed)

        # Return average
        return sum(self.raw_speeds[tracker_id]) / len(self.raw_speeds[tracker_id])

    def _exponential_smooth(self, tracker_id: int, raw_speed: float) -> float:
        """Apply exponential moving average smoothing."""
        if tracker_id not in self.ema_speeds:
            self.ema_speeds[tracker_id] = raw_speed
        else:
            self.ema_speeds[tracker_id] = (
                self.ema_alpha * raw_speed
                + (1 - self.ema_alpha) * self.ema_speeds[tracker_id]
            )
        return self.ema_speeds[tracker_id]

    def _kalman_smooth(self, tracker_id: int, raw_speed: float) -> float:
        """Apply Kalman filter smoothing."""
        if tracker_id not in self.kalman_states:
            # Initialize
            self.kalman_states[tracker_id] = {"x": raw_speed, "P": 1.0}

        state = self.kalman_states[tracker_id]

        # Prediction step (assuming constant speed)
        x_pred = state["x"]
        P_pred = state["P"] + self.kalman_q

        # Update step
        K = P_pred / (P_pred + self.kalman_r)  # Kalman gain
        x_new = x_pred + K * (raw_speed - x_pred)
        P_new = (1 - K) * P_pred

        # Store updated state
        self.kalman_states[tracker_id] = {"x": x_new, "P": P_new}

        return x_new

    def reset(self):
        """Reset all internal state."""
        self.raw_speeds.clear()
        self.ema_speeds.clear()
        self.kalman_states.clear()

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
            "none": "No smoothing applied (raw speeds for debugging)",
            "moving_average": f"Simple moving average over {self.window_size} measurements",
            "exponential": f"Exponential moving average with alpha={self.ema_alpha}",
            "kalman": f"Kalman filter with Q={self.kalman_q}, R={self.kalman_r}",
        }
        return descriptions.get(self.method, "Unknown method")

    def _get_method_parameters(self) -> str:
        """Get current method parameters."""
        if self.method == "moving_average":
            return f"window_size={self.window_size}"
        elif self.method == "exponential":
            return f"ema_alpha={self.ema_alpha}"
        elif self.method == "kalman":
            return f"kalman_q={self.kalman_q}, kalman_r={self.kalman_r}"
        else:
            return "No parameters"

    def get_tracker_stats(self, tracker_id: int) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific tracker.

        Args:
            tracker_id: Tracker ID to get stats for

        Returns:
            Dictionary with statistics or None if tracker not found
        """
        stats = {}

        if self.method == "moving_average" and tracker_id in self.raw_speeds:
            speeds = list(self.raw_speeds[tracker_id])
            if speeds:
                stats["current_speed"] = speeds[-1]
                stats["average_speed"] = sum(speeds) / len(speeds)
                stats["speed_std"] = np.std(speeds) if len(speeds) > 1 else 0.0
                stats["measurements_count"] = len(speeds)

        elif self.method == "exponential" and tracker_id in self.ema_speeds:
            stats["smoothed_speed"] = self.ema_speeds[tracker_id]

        elif self.method == "kalman" and tracker_id in self.kalman_states:
            state = self.kalman_states[tracker_id]
            stats["smoothed_speed"] = state["x"]
            stats["uncertainty"] = state["P"]

        return stats if stats else None
