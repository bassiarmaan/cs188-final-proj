"""Tiny PID on xyz — numpy only, nothing clever."""

from __future__ import annotations

import numpy as np


class PIDPosition:
    def __init__(self, kp: float, ki: float, kd: float, target: np.ndarray):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = np.array(target, dtype=np.float64)
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.last_error_norm = 0.0

    def set_target(self, target: np.ndarray):
        self.target = np.array(target, dtype=np.float64)

    def reset(self, target: np.ndarray | None = None):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.last_error_norm = 0.0
        if target is not None:
            self.target = np.array(target, dtype=np.float64)

    def error_norm(self) -> float:
        return float(self.last_error_norm)

    def update(self, current_pos: np.ndarray, dt: float) -> np.ndarray:
        err = self.target - np.array(current_pos, dtype=np.float64)
        self.last_error_norm = float(np.linalg.norm(err))
        self.integral += err * dt
        deriv = (err - self.prev_error) / dt if dt > 0 else np.zeros(3)
        self.prev_error = err.copy()
        return self.kp * err + self.ki * self.integral + self.kd * deriv
