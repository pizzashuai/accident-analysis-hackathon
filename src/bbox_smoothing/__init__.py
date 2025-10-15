"""
Bounding Box Smoothing Module

This module provides various algorithms for smoothing bounding box detections
to reduce jitter and improve tracking stability, especially for speed estimation.

Available methods:
- none: No smoothing (baseline)
- moving_average: Simple moving average over sliding window
- exponential: Exponential moving average with configurable alpha
- kalman: Kalman filter for optimal smoothing under Gaussian noise
- iou_weighted: IOU-weighted average for adaptive smoothing

Recommended: Use 'kalman' for best speed estimation accuracy.
"""

from .bbox_smoother import BboxSmoother

__all__ = ["BboxSmoother"]
