"""FROZEN observation/action contract for the follower controller.

Every controller -- the analytic one today, an RL policy tomorrow, a
population-coded spiking network (PopSAN-style) on SpiNNaker after that -- sees
exactly this interface. Change it and every trained policy becomes invalid, so
treat it as fixed: add new fields only by appending, never by reordering.

Design rules, chosen for spiking deployment:

* **Low dimensional** (10 observations, 2 actions). Population coding spends
  neurons per dimension, so width is expensive.
* **Relative, never absolute.** Every quantity is expressed in the follower's
  body frame, so the policy cannot memorise world coordinates or the path.
* **Normalized to [-1, 1]**, which is the input range population encoders
  expect, and the output range of a tanh policy head.
* **Angles only as (sin, cos)**, so there is no wraparound discontinuity for a
  network to model.
* **Built from the ESTIMATED leader state**, never ground truth. The estimate
  comes from the communication interface, which is the whole point of the task.

Observation layout (indices are part of the contract)::

    0  dx            leader position relative to follower, follower body frame, x / max_range
    1  dy            ... y / max_range
    2  sin_dtheta    leader heading relative to follower heading
    3  cos_dtheta
    4  ex            desired slot position relative to follower, body frame, x / max_range
    5  ey            ... y / max_range
    6  v_self        follower forward speed / v_max
    7  w_self        follower yaw rate / w_max
    8  v_leader_est  estimated leader forward speed / v_max
    9  w_leader_est  estimated leader yaw rate / w_max

Action layout::

    0  v   normalized forward speed, scaled to [-v_max, v_max]
    1  w   normalized yaw rate, scaled to [-w_max, w_max]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RobotState, to_body_frame, wrap_angle
from .spaces import Box

OBS_NAMES = (
    'dx',
    'dy',
    'sin_dtheta',
    'cos_dtheta',
    'ex',
    'ey',
    'v_self',
    'w_self',
    'v_leader_est',
    'w_leader_est',
)
ACTION_NAMES = ('v', 'w')

OBS_DIM = len(OBS_NAMES)
ACTION_DIM = len(ACTION_NAMES)

#: Version of the contract. Bump on any layout change; stored in every CSV.
CONTRACT_VERSION = '1.0'


@dataclass(frozen=True)
class ContractConfig:
    """Normalization limits. One source of truth for observations AND actions.

    ``max_range`` is the distance at which relative positions saturate at +-1.
    Defaults suit the Rover Mini: it can exceed 1 m/s, but the task does not
    need it, and a tighter range means better encoder resolution.
    """

    max_range: float = 3.0
    v_max: float = 1.0
    w_max: float = 2.0

    def __post_init__(self):
        for name in ('max_range', 'v_max', 'w_max'):
            if getattr(self, name) <= 0.0:
                raise ValueError(f'{name} must be positive, got {getattr(self, name)}')


def observation_space(config=None):
    """Normalized observation space (mirrors ``gymnasium.spaces.Box``)."""
    del config  # every entry is normalized, so the bounds never depend on it
    return Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32, names=OBS_NAMES)


def action_space(config=None):
    """Normalized action space (mirrors ``gymnasium.spaces.Box``)."""
    del config
    return Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32, names=ACTION_NAMES)


def slot_position(leader_state, offset_d):
    """World position of the follower's slot: ``offset_d`` metres behind the leader.

    The slot is defined in the LEADER's body frame, so it rotates with the
    leader through turns.
    """
    return leader_state.xy + np.array(
        [-offset_d * np.cos(leader_state.theta), -offset_d * np.sin(leader_state.theta)],
        dtype=float,
    )


def build_observation(follower_state, leader_estimate, offset_d, config):
    """Assemble the normalized observation vector.

    Args:
        follower_state: the follower's own state (it knows this exactly).
        leader_estimate: the leader state **as estimated by the comm
            interface**. Never pass ground truth here outside of tests.
        offset_d: formation offset in metres, behind the leader.
        config: :class:`ContractConfig` normalization limits.

    Returns:
        ``np.ndarray`` of shape (10,), float32, every entry within [-1, 1].
    """
    rel_leader = to_body_frame(leader_estimate.xy, follower_state.xy, follower_state.theta)
    slot_xy = slot_position(leader_estimate, offset_d)
    rel_slot = to_body_frame(slot_xy, follower_state.xy, follower_state.theta)
    dtheta = wrap_angle(leader_estimate.theta - follower_state.theta)

    obs = np.array(
        [
            rel_leader[0] / config.max_range,
            rel_leader[1] / config.max_range,
            np.sin(dtheta),
            np.cos(dtheta),
            rel_slot[0] / config.max_range,
            rel_slot[1] / config.max_range,
            follower_state.v / config.v_max,
            follower_state.w / config.w_max,
            leader_estimate.v / config.v_max,
            leader_estimate.w / config.w_max,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -1.0, 1.0)


def decode_observation(obs, config):
    """Inverse of :func:`build_observation`, in physical units.

    Controllers use this so their gains stay in metres and radians instead of
    normalized units. Values that saturated at +-1 stay saturated.
    """
    obs = np.asarray(obs, dtype=float)
    if obs.shape != (OBS_DIM,):
        raise ValueError(f'expected observation of shape ({OBS_DIM},), got {obs.shape}')
    return {
        'leader_rel': obs[0:2] * config.max_range,
        'dtheta': float(np.arctan2(obs[2], obs[3])),
        'slot_rel': obs[4:6] * config.max_range,
        'v_self': float(obs[6] * config.v_max),
        'w_self': float(obs[7] * config.w_max),
        'v_leader_est': float(obs[8] * config.v_max),
        'w_leader_est': float(obs[9] * config.w_max),
    }


def scale_action(action, config):
    """Normalized action in [-1, 1] -> physical (v [m/s], w [rad/s]).

    Out-of-range actions are clipped, so a policy that has not learned to
    respect its own output range cannot exceed the robot's limits.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f'expected action of shape ({ACTION_DIM},), got {action.shape}')
    clipped = np.clip(action, -1.0, 1.0)
    return float(clipped[0] * config.v_max), float(clipped[1] * config.w_max)


def unscale_action(v, w, config):
    """Physical (v, w) -> normalized action in [-1, 1]."""
    return np.clip(
        np.array([v / config.v_max, w / config.w_max], dtype=np.float32), -1.0, 1.0)


def formation_errors(follower_state, leader_true_state, offset_d):
    """Ground-truth formation error, in the LEADER's body frame.

    Evaluation only: uses the true leader state, so it never feeds a
    controller.

    Both components describe where the SLOT is relative to the follower, i.e.
    the correction the follower still owes:

    * ``longitudinal`` > 0: the slot is ahead, the follower is lagging.
    * ``lateral`` > 0: the slot is to the follower's left (in the leader's
      frame), i.e. the follower has drifted right of it.
    * ``heading``: leader heading minus follower heading, wrapped.
    * ``euclidean``: distance from the follower to the slot.
    """
    slot_xy = slot_position(leader_true_state, offset_d)
    rel = to_body_frame(follower_state.xy, slot_xy, leader_true_state.theta)
    return {
        'longitudinal': float(-rel[0]),
        'lateral': float(-rel[1]),
        'heading': float(wrap_angle(leader_true_state.theta - follower_state.theta)),
        'euclidean': float(np.hypot(rel[0], rel[1])),
    }
