"""Closed reference paths for the leader.

The task needs information demand to VARY over time, so the default path mixes
segments a constant-velocity predictor handles perfectly (straights, zero
curvature) with segments where it degrades fast (constant-curvature caps).
That contrast is what makes event-triggered transmission beat periodic
transmission; a circle-only path would not show it.

Paths are stored as a dense closed polyline, which makes nearest-point and
lookahead queries exact enough for pure pursuit and keeps every path type
behind one interface.
"""

from __future__ import annotations

import numpy as np


class ClosedPath:
    """A closed polyline with arc-length queries.

    Args:
        points: (N, 2) vertices, in order, WITHOUT repeating the first point.
        name: identifier recorded in results.
    """

    def __init__(self, points, name='path'):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            raise ValueError('points must be an (N, 2) array with N >= 4')
        self.points = pts
        self.name = name
        # Segment i joins points[i] -> points[(i + 1) % N]; cumulative[i] is the
        # arc length at points[i].
        deltas = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        self.segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        self.cumulative = np.concatenate([[0.0], np.cumsum(self.segment_lengths)])
        self.length = float(self.cumulative[-1])

    def __len__(self):
        return len(self.points)

    def closest(self, xy):
        """Nearest vertex to ``xy``.

        Returns ``(index, point, distance, arc_length)``. The polyline is dense
        (roughly 1 cm spacing), so vertex resolution is enough here and keeps
        this cheap inside the control loop.
        """
        xy = np.asarray(xy, dtype=float)
        d2 = np.sum((self.points - xy) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx, self.points[idx], float(np.sqrt(d2[idx])), float(self.cumulative[idx])

    def point_at(self, arc_length):
        """Interpolated point at ``arc_length``, wrapping around the loop."""
        s = float(np.mod(arc_length, self.length))
        idx = int(np.searchsorted(self.cumulative, s, side='right') - 1)
        idx = min(max(idx, 0), len(self.points) - 1)
        seg_len = self.segment_lengths[idx]
        t = 0.0 if seg_len <= 0.0 else (s - self.cumulative[idx]) / seg_len
        start = self.points[idx]
        end = self.points[(idx + 1) % len(self.points)]
        return start + t * (end - start)

    def lookahead(self, xy, distance):
        """Point ``distance`` metres ahead (along the path) of the nearest vertex."""
        _, _, _, s = self.closest(xy)
        return self.point_at(s + distance)

    def lookahead_pose(self, xy, distance):
        """Lookahead point ``distance`` ahead of ``xy``, and the tangent there.

        One call for what the centralized observation needs, so the expensive
        nearest-vertex search runs once per control step instead of twice.
        """
        _, _, _, s = self.closest(xy)
        return self.point_at(s + distance), self.tangent_at(s + distance)

    def tracking_error(self, xy):
        """Distance from ``xy`` to the path -- the leader's path-tracking error."""
        return self.closest(xy)[2]

    def tangent_at(self, arc_length):
        """Unit tangent (heading direction) at ``arc_length``."""
        delta = self.point_at(arc_length + 0.05) - self.point_at(arc_length - 0.05)
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0 else np.array([1.0, 0.0])

    def start_pose(self):
        """Pose (x, y, theta) on the path, used for deterministic resets."""
        point = self.points[0]
        tangent = self.tangent_at(0.0)
        return float(point[0]), float(point[1]), float(np.arctan2(tangent[1], tangent[0]))

    def curvature(self):
        """Discrete curvature at each vertex (1/m), for diagnostics and plots."""
        pts = self.points
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        v1 = pts - prev
        v2 = nxt - pts
        a1 = np.arctan2(v1[:, 1], v1[:, 0])
        a2 = np.arctan2(v2[:, 1], v2[:, 0])
        dtheta = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
        ds = 0.5 * (np.hypot(v1[:, 0], v1[:, 1]) + np.hypot(v2[:, 0], v2[:, 1]))
        return np.where(ds > 0, dtheta / np.maximum(ds, 1e-9), 0.0)


def oval_path(straight_length=4.0, radius=1.5, resolution=0.01):
    """Oval: two straights joined by semicircular caps (the DEFAULT path).

    Curvature is exactly 0 on the straights and 1/``radius`` on the caps, which
    gives the sharpest possible contrast in predictor difficulty.
    """
    if straight_length <= 0 or radius <= 0:
        raise ValueError('straight_length and radius must be positive')
    half = straight_length / 2.0
    n_straight = max(int(straight_length / resolution), 2)
    n_arc = max(int(np.pi * radius / resolution), 2)

    bottom = np.stack([
        np.linspace(-half, half, n_straight, endpoint=False),
        np.full(n_straight, -radius),
    ], axis=1)
    angles_right = np.linspace(-np.pi / 2, np.pi / 2, n_arc, endpoint=False)
    right = np.stack([
        half + radius * np.cos(angles_right),
        radius * np.sin(angles_right),
    ], axis=1)
    top = np.stack([
        np.linspace(half, -half, n_straight, endpoint=False),
        np.full(n_straight, radius),
    ], axis=1)
    angles_left = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc, endpoint=False)
    left = np.stack([
        -half + radius * np.cos(angles_left),
        radius * np.sin(angles_left),
    ], axis=1)

    return ClosedPath(np.vstack([bottom, right, top, left]), name='oval')


def figure_eight_path(width=4.0, height=2.5, resolution=0.01):
    """Figure-8 (Gerono lemniscate): curvature varies continuously and flips sign."""
    if width <= 0 or height <= 0:
        raise ValueError('width and height must be positive')
    # Estimate the sample count from a coarse pass so spacing is ~resolution.
    t = np.linspace(0.0, 2 * np.pi, 2000, endpoint=False)
    coarse = np.stack([0.5 * width * np.cos(t), 0.5 * height * np.sin(2 * t)], axis=1)
    perimeter = float(np.sum(np.hypot(*np.diff(np.vstack([coarse, coarse[:1]]), axis=0).T)))
    n = max(int(perimeter / resolution), 500)
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    points = np.stack([0.5 * width * np.cos(t), 0.5 * height * np.sin(2 * t)], axis=1)
    return ClosedPath(points, name='figure8')


#: Registry so configs can name a path as a string.
PATH_BUILDERS = {
    'oval': oval_path,
    'figure8': figure_eight_path,
}


def make_path(name='oval', **kwargs):
    """Build a path by name, with keyword arguments from the config."""
    try:
        builder = PATH_BUILDERS[name]
    except KeyError:
        raise ValueError(
            f'unknown path {name!r}; available: {sorted(PATH_BUILDERS)}') from None
    return builder(**kwargs)
