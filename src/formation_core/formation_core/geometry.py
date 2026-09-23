"""SE(2) helpers and the state record shared by every module.

Angles are always radians. Anything that crosses a module boundary as an
observation uses sin/cos instead of a raw angle (see :mod:`contract`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def rotation(theta):
    """2x2 rotation matrix for ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def to_body_frame(point_world, origin_xy, origin_theta):
    """Express ``point_world`` in the body frame at (``origin_xy``, ``origin_theta``)."""
    delta = np.asarray(point_world, dtype=float) - np.asarray(origin_xy, dtype=float)
    return rotation(-origin_theta) @ delta


@dataclass
class RobotState:
    """Planar unicycle state: pose plus body-frame velocities.

    ``v`` is forward speed (m/s) and ``w`` is yaw rate (rad/s). This is the one
    state record used by the fast twin, the predictor, the comm interface and
    the Gazebo adapter, so all of them stay interchangeable.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    w: float = 0.0

    @property
    def xy(self):
        return np.array([self.x, self.y], dtype=float)

    def to_array(self):
        return np.array([self.x, self.y, self.theta, self.v, self.w], dtype=float)

    @classmethod
    def from_array(cls, arr):
        arr = np.asarray(arr, dtype=float)
        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]))

    def copy(self):
        return RobotState(self.x, self.y, self.theta, self.v, self.w)


def state_error_norm(a, b, heading_weight=0.0):
    """Distance between two states, used by event-triggered transmission.

    Position error in metres by default. ``heading_weight`` (metres per radian)
    optionally folds in heading error; it is 0.0 by default so the trigger
    threshold delta reads directly as a distance.
    """
    position = math.hypot(a.x - b.x, a.y - b.y)
    if heading_weight == 0.0:
        return position
    heading = abs(float(wrap_angle(a.theta - b.theta)))
    return math.hypot(position, heading_weight * heading)
