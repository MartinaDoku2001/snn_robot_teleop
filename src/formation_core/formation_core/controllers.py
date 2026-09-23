"""Controllers.

Two different jobs:

* :class:`Controller` -- the FOLLOWER interface, ``obs -> normalized action``.
  This is the slot an RL policy and later a spiking network drop into. It sees
  only the frozen contract observation, so it cannot cheat by reading ground
  truth, the path, or the leader's commands.
* :class:`PurePursuitLeader` -- the leader's scripted path tracker. It is part
  of the environment, not a learned component, so it reads the path directly.
"""

from __future__ import annotations

import abc

import numpy as np

from .contract import decode_observation, unscale_action
from .geometry import wrap_angle


class Controller(abc.ABC):
    """Maps a contract observation to a normalized action in [-1, 1]^2."""

    @abc.abstractmethod
    def act(self, obs):
        """Return a length-2 array of normalized (v, w)."""

    def reset(self):
        """Per-episode hook for stateful controllers."""

    @property
    def name(self):
        return type(self).__name__

    @property
    def params(self):
        return {}


class AnalyticFollower(Controller):
    """Proportional pure-pursuit-style law on the relative slot error.

    Exercises the whole harness without any learning, and is the baseline an RL
    or spiking controller has to beat.

    Law, all quantities in the follower's body frame (from the observation):

    * ``v`` tracks the estimated leader speed (feed-forward) plus a
      proportional term on the longitudinal slot error ``ex``.
    * ``w`` steers toward the slot, blending the bearing to the slot with the
      leader's heading. Far from the slot, driving at it is right; on top of
      the slot, bearing is ill-conditioned, so heading alignment takes over.
      ``align_radius`` sets where the blend happens.
    """

    def __init__(
        self,
        config,
        k_longitudinal=1.2,
        k_bearing=2.0,
        k_heading=1.2,
        align_radius=0.25,
        reverse_guard=True,
    ):
        self.config = config
        self.k_longitudinal = float(k_longitudinal)
        self.k_bearing = float(k_bearing)
        self.k_heading = float(k_heading)
        self.align_radius = float(align_radius)
        #: Do not drive backwards to correct a slot the follower has overshot;
        #: a differential-drive robot corrects it by slowing down instead.
        self.reverse_guard = bool(reverse_guard)

    def act(self, obs):
        decoded = decode_observation(obs, self.config)
        ex, ey = decoded['slot_rel']
        distance = float(np.hypot(ex, ey))

        # --- forward speed: match the leader, close the longitudinal gap
        v_cmd = decoded['v_leader_est'] + self.k_longitudinal * ex
        if self.reverse_guard:
            v_cmd = max(v_cmd, 0.0)

        # --- yaw rate: steer at the slot when far, align with the leader when on it
        bearing = float(np.arctan2(ey, ex)) if distance > 1e-6 else 0.0
        blend = float(np.clip(distance / max(self.align_radius, 1e-6), 0.0, 1.0))
        heading_term = float(wrap_angle(decoded['dtheta']))
        w_cmd = (
            blend * self.k_bearing * bearing
            + (1.0 - blend) * self.k_heading * heading_term
            + decoded['w_leader_est'] * (1.0 - blend))

        return unscale_action(v_cmd, w_cmd, self.config)

    @property
    def name(self):
        return 'analytic'

    @property
    def params(self):
        return {
            'k_longitudinal': self.k_longitudinal,
            'k_bearing': self.k_bearing,
            'k_heading': self.k_heading,
            'align_radius': self.align_radius,
        }


class ZeroController(Controller):
    """Does nothing. Useful as a metrics/no-op baseline in tests."""

    def act(self, obs):
        del obs
        return np.zeros(2, dtype=np.float32)

    @property
    def name(self):
        return 'zero'


class PurePursuitLeader:
    """Pure pursuit on a closed path, with speed reduced through curves.

    Part of the environment: it reads the leader's true state and the path.
    Lookahead scales with speed, which is standard and keeps the leader stable
    on both the straights and the caps.
    """

    def __init__(
        self,
        path,
        target_speed=0.6,
        lookahead=0.5,
        lookahead_gain=0.3,
        curvature_slowdown=0.6,
        limits=None,
    ):
        self.path = path
        self.target_speed = float(target_speed)
        self.lookahead = float(lookahead)
        self.lookahead_gain = float(lookahead_gain)
        #: Fraction of speed shed at maximum curvature; 0 disables slowdown.
        self.curvature_slowdown = float(curvature_slowdown)
        self.limits = limits

    def reset(self):
        pass

    def command(self, state):
        """Return the (v, w) command for the leader's current state."""
        lookahead = self.lookahead + self.lookahead_gain * max(state.v, 0.0)
        target = self.path.lookahead(state.xy, lookahead)

        delta = target - state.xy
        alpha = float(wrap_angle(np.arctan2(delta[1], delta[0]) - state.theta))
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return self.target_speed, 0.0

        # Pure pursuit curvature, then w = v * kappa.
        kappa = 2.0 * np.sin(alpha) / max(distance, 1e-6)
        speed = self.target_speed
        if self.curvature_slowdown > 0.0:
            # Shed speed in proportion to how hard the turn is.
            slow = 1.0 / (1.0 + self.curvature_slowdown * abs(kappa) * max(distance, 1e-6))
            speed *= float(np.clip(slow, 0.2, 1.0))
        return float(speed), float(speed * kappa)


CONTROLLERS = {
    'analytic': AnalyticFollower,
    'zero': ZeroController,
}


def make_controller(name='analytic', config=None, **kwargs):
    """Build a follower controller by name.

    Later phases register their controllers here (``'rl'``, ``'snn'``), and
    nothing else in the harness changes.
    """
    try:
        cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            f'unknown controller {name!r}; available: {sorted(CONTROLLERS)}') from None
    if cls is ZeroController:
        return cls()
    return cls(config=config, **kwargs)
