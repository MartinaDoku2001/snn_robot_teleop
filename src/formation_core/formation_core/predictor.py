"""Dead-reckoning predictors used between transmissions.

The predictor is deliberately GENERIC: it knows nothing about the reference
path, the leader's controller, or the noise process. That asymmetry is what
makes information demand vary -- it is near-exact on straights and degrades
through turns, so a good transmission policy spends messages on turns.

A learned predictor can be dropped in later by implementing :class:`Predictor`.
"""

from __future__ import annotations

import abc

import numpy as np

from .geometry import RobotState, wrap_angle


class Predictor(abc.ABC):
    """Propagates an estimate forward one timestep without new measurements."""

    @abc.abstractmethod
    def step(self, estimate, dt, last_command=None):
        """Return the estimate advanced by ``dt``.

        Args:
            estimate: current :class:`RobotState` estimate.
            dt: timestep in seconds.
            last_command: optional (v, w) command, for predictors that get to
                see what was commanded. Phase 1 predictors ignore it (the
                follower does not know the leader's commands); the hook exists
                so a model-based or learned predictor can use it unchanged.
        """

    def reset(self, state):
        """Optional hook for predictors carrying internal state."""
        del state

    @property
    def name(self):
        return type(self).__name__


class ConstantVelocityPredictor(Predictor):
    """Constant-velocity, constant-yaw-rate dead reckoning.

    Holds (v, w) from the last received message and integrates the unicycle
    model. Exact on constant-velocity motion (including steady turns), wrong
    wherever the leader accelerates, turns into or out of a curve, or is pushed
    by process noise.
    """

    def __init__(self, hold_yaw_rate=True):
        #: If False, the predictor assumes straight-line motion (w = 0), which
        #: is the weaker "constant heading" variant. Kept configurable because
        #: it is a useful ablation for the paper.
        self.hold_yaw_rate = hold_yaw_rate

    def step(self, estimate, dt, last_command=None):
        del last_command
        w = estimate.w if self.hold_yaw_rate else 0.0
        theta_mid = estimate.theta + 0.5 * w * dt
        return RobotState(
            x=estimate.x + estimate.v * np.cos(theta_mid) * dt,
            y=estimate.y + estimate.v * np.sin(theta_mid) * dt,
            theta=float(wrap_angle(estimate.theta + w * dt)),
            v=estimate.v,
            w=w,
        )

    @property
    def name(self):
        return 'constant_velocity' if self.hold_yaw_rate else 'constant_heading'


PREDICTORS = {
    'constant_velocity': lambda: ConstantVelocityPredictor(hold_yaw_rate=True),
    'constant_heading': lambda: ConstantVelocityPredictor(hold_yaw_rate=False),
}


def make_predictor(name='constant_velocity', **kwargs):
    try:
        builder = PREDICTORS[name]
    except KeyError:
        raise ValueError(
            f'unknown predictor {name!r}; available: {sorted(PREDICTORS)}') from None
    return builder(**kwargs)
