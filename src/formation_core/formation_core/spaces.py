"""A minimal Box space that mirrors ``gymnasium.spaces.Box``.

formation_core deliberately has no RL dependency. This class implements the
subset of the Gymnasium API the environments need (``low``, ``high``,
``shape``, ``dtype``, ``sample``, ``contains``), so wrapping the env for
Gymnasium later is a one-line adapter that swaps this for the real Box.
"""

from __future__ import annotations

import numpy as np


class Box:
    """Bounded continuous space."""

    def __init__(self, low, high, shape=None, dtype=np.float32, names=None):
        self.dtype = np.dtype(dtype)
        if shape is None:
            low_arr = np.asarray(low, dtype=self.dtype)
            shape = low_arr.shape
        self.shape = tuple(shape)
        self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()
        if np.any(self.low > self.high):
            raise ValueError('low must not exceed high')
        #: Optional per-dimension names. Not part of the Gymnasium API, but it
        #: keeps CSV columns and debugging readable.
        self.names = tuple(names) if names is not None else None
        if self.names is not None and len(self.names) != int(np.prod(self.shape)):
            raise ValueError('names must have one entry per dimension')

    def sample(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return bool(
            x.shape == self.shape and np.all(x >= self.low - 1e-6) and np.all(x <= self.high + 1e-6))

    def __repr__(self):
        return f'Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype.name})'

    def __eq__(self, other):
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high))
