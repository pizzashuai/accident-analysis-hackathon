"""
Speed Smoothing Module

This module provides various algorithms for smoothing speed estimates
to reduce noise and improve accuracy in vehicle speed tracking.

Available methods:
- none: No smoothing (raw speeds for debugging)
- moving_average: Simple moving average over sliding window
- exponential: Exponential moving average with configurable alpha
- kalman: Kalman filter for optimal smoothing under Gaussian noise

Recommended: Use 'kalman' for best speed estimation accuracy.
"""

from .speed_smoother import SpeedSmoother

__all__ = ["SpeedSmoother"]
