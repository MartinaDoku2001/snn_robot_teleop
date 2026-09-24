"""Controllers.

Since contract v2.0 there is ONE controller and it drives both robots:

* :class:`Controller` -- ``obs (16) -> normalized action (4)``. This is the slot
  an RL policy and later a spiking network drop into. It sees only the frozen
  contract observation, so it cannot cheat by reading ground truth, the path, or
  either robot's true state.
* :class:`AnalyticCentralized` -- the hand-written baseline occupying that slot,
  registered as ``'analytic'``. It is pure pursuit for the leader plus the
  formation law for the follower, both driven from the observation alone.
* :func:`pure_pursuit_command` and :func:`formation_follower_command` -- the two
  control laws, as plain functions. The centralized controller and the two Phase
  1 classes below both call them, so there is one implementation of each law.
* :class:`PurePursuitLeader` -- the Phase 1 scripted leader, kept because it
  still owns the path-side geometry (speed-scaled lookahead) that the analytic
  controller does not get to see. Nothing drives the leader with it under v2.0;
  it remains the reference implementation of the law.
* :class:`AnalyticFollower` -- the Phase 1 follower-only controller. Under v2.0
  it is a component, not a registered controller, because a controller must now
  emit four numbers.
"""

from __future__ import annotations

import abc
import math

import numpy as np

from .contract import ACTION_DIM, decode_observation, unscale_action
from .geometry import to_body_frame, wrap_angle


def pure_pursuit_command(lookahead_body, target_speed, curvature_slowdown=0.6):
    """``(v, w)`` steering toward a lookahead point given in the BODY frame.

    The geometric core of pure pursuit, factored out so that the scripted
    leader (which finds its lookahead on the path) and the centralized
    controller (which reads it out of the observation) run the same law.
    """
    lx, ly = float(lookahead_body[0]), float(lookahead_body[1])
    distance = math.hypot(lx, ly)
    if distance < 1e-6:
        return float(target_speed), 0.0

    alpha = math.atan2(ly, lx)
    kappa = 2.0 * math.sin(alpha) / max(distance, 1e-6)
    speed = float(target_speed)
    if curvature_slowdown > 0.0:
        # Shed speed in proportion to how hard the turn is.
        slow = 1.0 / (1.0 + curvature_slowdown * abs(kappa) * max(distance, 1e-6))
        speed *= float(np.clip(slow, 0.2, 1.0))
    return float(speed), float(speed * kappa)


def formation_follower_command(slot_rel, dtheta, v_leader_est, w_leader_est,
                               k_longitudinal=1.2, k_bearing=2.0, k_heading=1.2,
                               align_radius=0.25, reverse_guard=True):
    """``(v, w)`` holding the formation slot, all inputs in the FOLLOWER frame.

    * ``v`` tracks the estimated leader speed (feed-forward) plus a
      proportional term on the longitudinal slot error.
    * ``w`` steers toward the slot, blending the bearing to the slot with the
      leader's heading. Far from the slot, driving at it is right; on top of
      the slot, bearing is ill-conditioned, so heading alignment takes over.
      ``align_radius`` sets where the blend happens.
    """
    ex, ey = float(slot_rel[0]), float(slot_rel[1])
    distance = math.hypot(ex, ey)

    v_cmd = v_leader_est + k_longitudinal * ex
    if reverse_guard:
        # Do not drive backwards to correct a slot the follower has overshot; a
        # differential-drive robot corrects it by slowing down instead.
        v_cmd = max(v_cmd, 0.0)

    bearing = math.atan2(ey, ex) if distance > 1e-6 else 0.0
    blend = float(np.clip(distance / max(align_radius, 1e-6), 0.0, 1.0))
    heading_term = float(wrap_angle(dtheta))
    w_cmd = (
        blend * k_bearing * bearing
        + (1.0 - blend) * k_heading * heading_term
        + w_leader_est * (1.0 - blend))
    return float(v_cmd), float(w_cmd)


class Controller(abc.ABC):
    """Maps a contract observation to a normalized action in [-1, 1]^4."""

    #: Set by controllers that also need the leader's speed settings, which are
    #: parameters of the task rather than part of the observation.
    wants_leader_config = False

    @abc.abstractmethod
    def act(self, obs):
        """Return a length-4 array: normalized (v, w) for leader then follower."""

    def reset(self):
        """Per-episode hook for stateful controllers."""

    @property
    def name(self):
        return type(self).__name__

    @property
    def params(self):
        return {}


class FollowerGains:
    """Tunable gains of :func:`formation_follower_command`, in one place."""

    def __init__(self, k_longitudinal=1.2, k_bearing=2.0, k_heading=1.2,
                 align_radius=0.25, reverse_guard=True):
        self.k_longitudinal = float(k_longitudinal)
        self.k_bearing = float(k_bearing)
        self.k_heading = float(k_heading)
        self.align_radius = float(align_radius)
        self.reverse_guard = bool(reverse_guard)

    def command(self, slot_rel, dtheta, v_leader_est, w_leader_est):
        return formation_follower_command(
            slot_rel, dtheta, v_leader_est, w_leader_est,
            k_longitudinal=self.k_longitudinal, k_bearing=self.k_bearing,
            k_heading=self.k_heading, align_radius=self.align_radius,
            reverse_guard=self.reverse_guard)

    def as_dict(self):
        return {
            'k_longitudinal': self.k_longitudinal,
            'k_bearing': self.k_bearing,
            'k_heading': self.k_heading,
            'align_radius': self.align_radius,
        }


class AnalyticFollower:
    """Phase 1's follower-only law, kept as a COMPONENT of the v2.0 controller.

    Under contract v1.0 this was the registered controller. v2.0 controllers
    emit four numbers, so this is no longer one of them -- it is the follower
    half of :class:`AnalyticCentralized`, and is kept separate because the
    follower law is what a later decentralized experiment would reuse.
    """

    def __init__(self, config, **gains):
        self.config = config
        self.gains = FollowerGains(**gains)

    def command(self, decoded):
        """``(v, w)`` for the follower, from a decoded v2.0 observation."""
        return self.gains.command(
            decoded['slot_rel'], decoded['dtheta'],
            decoded['v_leader_est'], decoded['w_leader_est'])

    @property
    def name(self):
        return 'analytic_follower'

    @property
    def params(self):
        return self.gains.as_dict()


class AnalyticCentralized(Controller):
    """The v2.0 baseline: one controller, both robots, observation only.

    Leader: pure pursuit on the lookahead point carried in the observation.
    Follower: the Phase 1 formation law on the slot error.

    Neither half reads the path, the true states, or anything outside the
    16-dim contract -- which is exactly what makes this a fair baseline for the
    RL and spiking controllers that occupy the same slot later, and a test that
    the contract carries enough information to drive both robots at all.
    """

    wants_leader_config = True

    def __init__(self, config, leader=None, target_speed=None,
                 curvature_slowdown=None, **gains):
        self.config = config
        # Leader speed settings are task parameters, not observations: the
        # controller is told how fast the formation should travel, the same way
        # the scripted leader was.
        self.target_speed = float(
            target_speed if target_speed is not None
            else getattr(leader, 'target_speed', 0.6))
        self.curvature_slowdown = float(
            curvature_slowdown if curvature_slowdown is not None
            else getattr(leader, 'curvature_slowdown', 0.6))
        self.follower = AnalyticFollower(config, **gains)

    def act(self, obs):
        decoded = decode_observation(obs, self.config)
        leader_command = pure_pursuit_command(
            decoded['lookahead_rel'], self.target_speed, self.curvature_slowdown)
        follower_command = self.follower.command(decoded)
        return unscale_action(leader_command, follower_command, self.config)

    @property
    def name(self):
        return 'analytic'

    @property
    def params(self):
        return dict(self.follower.params,
                    target_speed=self.target_speed,
                    curvature_slowdown=self.curvature_slowdown)


class ZeroController(Controller):
    """Does nothing. Useful as a metrics/no-op baseline in tests."""

    def act(self, obs):
        del obs
        return np.zeros(ACTION_DIM, dtype=np.float32)

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

        # Same law the centralized controller runs; the only difference is that
        # this one finds its own lookahead on the path instead of reading it
        # out of an observation.
        return pure_pursuit_command(
            to_body_frame(target, state.xy, state.theta),
            self.target_speed, self.curvature_slowdown)


CONTROLLERS = {
    'analytic': AnalyticCentralized,
    'zero': ZeroController,
}


def register_controller(name, cls):
    """Add a controller to the registry.

    Used by optional packages so that importing them is what makes ``'rl'`` (and
    later ``'snn'``) available, and ``formation_core`` keeps no dependency on
    them.
    """
    CONTROLLERS[name] = cls
    return cls


def make_controller(name='analytic', config=None, leader=None, **kwargs):
    """Build the centralized controller by name.

    Later phases register their controllers here (``'rl'``, ``'snn'``), and
    nothing else in the harness changes.

    Args:
        config: :class:`~formation_core.contract.ContractConfig`.
        leader: optional :class:`~formation_core.config.LeaderConfig`, passed
            only to controllers that declare ``wants_leader_config``.
    """
    try:
        cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            f'unknown controller {name!r}; available: {sorted(CONTROLLERS)}') from None
    if cls is ZeroController:
        return cls()
    if getattr(cls, 'wants_leader_config', False):
        kwargs.setdefault('leader', leader)
    return cls(config=config, **kwargs)
