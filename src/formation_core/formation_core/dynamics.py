"""Unicycle kinematics and the leader's process noise.

The fast twin integrates this model; the Gazebo backend replaces it with the
real simulator. Both produce :class:`~formation_core.geometry.RobotState`, so
everything downstream is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RobotState, wrap_angle


@dataclass(frozen=True)
class MotionLimits:
    """Velocity and acceleration limits applied to every command."""

    v_max: float = 1.0
    w_max: float = 2.0
    a_max: float = 2.0      # m/s^2
    alpha_max: float = 6.0  # rad/s^2

    def clamp_command(self, v, w):
        return (
            float(np.clip(v, -self.v_max, self.v_max)),
            float(np.clip(w, -self.w_max, self.w_max)),
        )

    def apply_acceleration(self, state, v_cmd, w_cmd, dt):
        """Rate-limit a command toward what the robot can actually reach."""
        dv = np.clip(v_cmd - state.v, -self.a_max * dt, self.a_max * dt)
        dw = np.clip(w_cmd - state.w, -self.alpha_max * dt, self.alpha_max * dt)
        return float(state.v + dv), float(state.w + dw)


@dataclass(frozen=True)
class ProcessNoise:
    """Small BOUNDED process noise on the leader's realised velocities.

    Deliberately correlated (AR(1)) rather than white: white noise averages out
    and a constant-velocity predictor barely notices it, whereas correlated
    noise makes the leader drift off any generic model -- which is the point of
    the task. Both the bound and the correlation are configurable, and the
    noise is always clipped to +-``v_bound`` / +-``w_bound`` so the leader stays
    well behaved.

    Set ``v_bound = w_bound = 0`` for a noise-free (perfectly predictable) run.
    """

    v_bound: float = 0.05      # m/s
    w_bound: float = 0.15      # rad/s
    correlation: float = 0.9   # AR(1) coefficient in [0, 1)

    def __post_init__(self):
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError('correlation must be in [0, 1)')
        if self.v_bound < 0.0 or self.w_bound < 0.0:
            raise ValueError('noise bounds must be non-negative')

    @property
    def enabled(self):
        return self.v_bound > 0.0 or self.w_bound > 0.0


class NoiseProcess:
    """Stateful AR(1) noise generator driven by a seeded RNG."""

    def __init__(self, spec, rng):
        self.spec = spec
        self.rng = rng
        self.state = np.zeros(2, dtype=float)

    def reset(self):
        self.state = np.zeros(2, dtype=float)

    def sample(self):
        """Next (dv, dw) noise pair, bounded by the spec."""
        if not self.spec.enabled:
            return 0.0, 0.0
        rho = self.spec.correlation
        # Scale the innovation so the stationary spread fills the bound.
        innovation = self.rng.uniform(-1.0, 1.0, size=2) * (1.0 - rho)
        self.state = np.clip(rho * self.state + innovation, -1.0, 1.0)
        return float(self.state[0] * self.spec.v_bound), float(self.state[1] * self.spec.w_bound)


def unicycle_step(state, v_cmd, w_cmd, dt, limits, noise=(0.0, 0.0)):
    """Integrate one step of unicycle motion.

    Commands are acceleration- and velocity-limited, then ``noise`` is added to
    the realised velocities, then the pose is integrated (midpoint on heading,
    which is accurate enough at dt <= 0.05 s and exact on straights).
    """
    v_cmd, w_cmd = limits.clamp_command(v_cmd, w_cmd)
    v, w = limits.apply_acceleration(state, v_cmd, w_cmd, dt)
    v = float(np.clip(v + noise[0], -limits.v_max, limits.v_max))
    w = float(np.clip(w + noise[1], -limits.w_max, limits.w_max))

    theta_mid = state.theta + 0.5 * w * dt
    return RobotState(
        x=state.x + v * np.cos(theta_mid) * dt,
        y=state.y + v * np.sin(theta_mid) * dt,
        theta=float(wrap_angle(state.theta + w * dt)),
        v=v,
        w=w,
    )
