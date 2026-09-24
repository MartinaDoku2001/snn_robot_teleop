"""FROZEN observation/action contract for the CENTRALIZED controller.

Every controller -- the analytic one today, an RL policy tomorrow, a
population-coded spiking network (PopSAN-style) on SpiNNaker after that -- sees
exactly this interface. Change it and every trained policy becomes invalid, so
treat it as fixed: add new fields only by appending, never by reordering.

**Version 2.0 (Phase 2) drives BOTH robots from one observation.** v1.0 was
follower-shaped: 10 observations, 2 actions, and no path information, on the
rule that the follower only had to chase the leader and must not be able to
memorise the route. A centralized controller *drives* the leader, so it needs a
path reference, and it emits commands for two robots. That is a re-freeze, not
a widening, hence the major version bump. v1.x results on disk stay readable --
every CSV records the version it was produced under.

Design rules, unchanged from v1.0 and chosen for spiking deployment:

* **Low dimensional** (16 observations, 4 actions). Population coding spends
  neurons per dimension, so width is expensive; centralization costs 6 inputs
  and this is the tightest layout that still drives both robots.
* **Relative, never absolute.** Every quantity is expressed in one robot's body
  frame, so the policy cannot memorise world coordinates. The path enters only
  as a lookahead point relative to the leader, never as a position on a map.
* **Normalized to [-1, 1]**, which is the input range population encoders
  expect, and the output range of a tanh policy head.
* **Angles only as (sin, cos)**, so there is no wraparound discontinuity.
* **Built from the per-robot ESTIMATES**, never ground truth. Both estimates
  come from that robot's communication interface, which is the whole point of
  the task. Ground truth appears only in :func:`formation_errors`, which is for
  evaluation.

Observation layout (indices are part of the contract)::

    leader block (7) -- everything in the LEADER estimate's body frame
    0   lead_look_x        path lookahead point, x / max_range
    1   lead_look_y        ... y / max_range
    2   lead_sin_tangent   path tangent at the lookahead, vs leader heading
    3   lead_cos_tangent
    4   lead_v             estimated forward speed / v_max
    5   lead_w             estimated yaw rate / w_max
    6   lead_age           steps since the leader's estimate was refreshed / age_max

    follower block (5) -- everything in the FOLLOWER estimate's body frame
    7   foll_v             estimated forward speed / v_max
    8   foll_w             estimated yaw rate / w_max
    9   foll_age           steps since the follower's estimate was refreshed / age_max
    10  slot_ex            desired slot position, x / max_range
    11  slot_ey            ... y / max_range

    coupling block (4) -- leader relative to follower, follower body frame
    12  rel_dx             leader position, x / max_range
    13  rel_dy             ... y / max_range
    14  rel_sin_dtheta     leader heading relative to follower heading
    15  rel_cos_dtheta

Action layout::

    0  lead_v   leader forward speed,  scaled to [-v_max, v_max]
    1  lead_w   leader yaw rate,       scaled to [-w_max, w_max]
    2  foll_v   follower forward speed
    3  foll_w   follower yaw rate

**The two age fields are mandatory and are the point of this layout.** Under
perfect communications they are ~0 every step and carry no information, which
is expected and is not a reason to drop them. They are the single place where
the control half of the project touches the communication half: once Phase 3
adds delay and loss and Phase 4 learns a per-robot transmission policy, the
controller has to know how stale each estimate is in order to act cautiously on
an old one. Adding them later would invalidate every policy trained before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RobotState, to_body_frame, wrap_angle
from .spaces import Box

OBS_NAMES = (
    # leader block
    'lead_look_x',
    'lead_look_y',
    'lead_sin_tangent',
    'lead_cos_tangent',
    'lead_v',
    'lead_w',
    'lead_age',
    # follower block
    'foll_v',
    'foll_w',
    'foll_age',
    'slot_ex',
    'slot_ey',
    # coupling block
    'rel_dx',
    'rel_dy',
    'rel_sin_dtheta',
    'rel_cos_dtheta',
)
ACTION_NAMES = ('lead_v', 'lead_w', 'foll_v', 'foll_w')

OBS_DIM = len(OBS_NAMES)
ACTION_DIM = len(ACTION_NAMES)

#: Index of the first element of each block, for slicing and for tests.
LEADER_BLOCK = slice(0, 7)
FOLLOWER_BLOCK = slice(7, 12)
COUPLING_BLOCK = slice(12, 16)

#: Names of the two mandatory staleness inputs (see the module docstring).
AGE_NAMES = ('lead_age', 'foll_age')

#: Version of the contract. Bump on any layout change; stored in every CSV.
CONTRACT_VERSION = '2.0'


@dataclass(frozen=True)
class ContractConfig:
    """Normalization limits. One source of truth for observations AND actions.

    ``max_range`` is the distance at which relative positions saturate at +-1.
    Defaults suit the Rover Mini: it can exceed 1 m/s, but the task does not
    need it, and a tighter range means better encoder resolution.

    ``age_max`` is the estimate age (in control steps) that normalizes to 1.0;
    older estimates saturate there. 50 steps is 2.5 s at the default dt, which
    is past the point where a constant-velocity prediction of this leader is
    worth anything, so the controller does not need to distinguish degrees of
    "hopelessly stale".

    ``lookahead_distance`` is how far along the path the leader's lookahead
    point is taken, in metres. It is FIXED here rather than scaled with speed
    (as :class:`~formation_core.controllers.PurePursuitLeader` does internally)
    so that the observation is a consistent geometric feature: a learned policy
    should not have the meaning of its inputs change with the robot's speed.
    """

    max_range: float = 3.0
    v_max: float = 1.0
    w_max: float = 2.0
    age_max: float = 50.0
    lookahead_distance: float = 0.6

    def __post_init__(self):
        for name in ('max_range', 'v_max', 'w_max', 'age_max', 'lookahead_distance'):
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


def build_observation(leader_estimate, follower_estimate, lookahead_xy, tangent_xy,
                      leader_age, follower_age, offset_d, config):
    """Assemble the normalized 16-dim observation for the centralized controller.

    Args:
        leader_estimate: leader state **as held by the coordinator**, i.e. the
            output of the leader's comm interface. Never ground truth.
        follower_estimate: same, for the follower.
        lookahead_xy: world-frame path lookahead point, taken
            ``config.lookahead_distance`` ahead of the leader estimate. Get it
            from :meth:`formation_core.paths.ClosedPath.lookahead_pose`.
        tangent_xy: world-frame unit tangent of the path at that point.
        leader_age / follower_age: steps since each estimate was last refreshed.
        offset_d: formation offset in metres, behind the leader.
        config: :class:`ContractConfig` normalization limits.

    Returns:
        ``np.ndarray`` of shape (16,), float32, every entry within [-1, 1].
    """
    # --- leader block, in the leader estimate's body frame
    look_body = to_body_frame(lookahead_xy, leader_estimate.xy, leader_estimate.theta)
    tangent_xy = np.asarray(tangent_xy, dtype=float)
    tangent_angle = float(np.arctan2(tangent_xy[1], tangent_xy[0]))
    tangent_rel = float(wrap_angle(tangent_angle - leader_estimate.theta))

    # --- follower block, in the follower estimate's body frame
    slot_xy = slot_position(leader_estimate, offset_d)
    rel_slot = to_body_frame(slot_xy, follower_estimate.xy, follower_estimate.theta)

    # --- coupling block: where the leader is, seen by the follower
    rel_leader = to_body_frame(
        leader_estimate.xy, follower_estimate.xy, follower_estimate.theta)
    dtheta = wrap_angle(leader_estimate.theta - follower_estimate.theta)

    obs = np.array(
        [
            look_body[0] / config.max_range,
            look_body[1] / config.max_range,
            np.sin(tangent_rel),
            np.cos(tangent_rel),
            leader_estimate.v / config.v_max,
            leader_estimate.w / config.w_max,
            float(leader_age) / config.age_max,

            follower_estimate.v / config.v_max,
            follower_estimate.w / config.w_max,
            float(follower_age) / config.age_max,
            rel_slot[0] / config.max_range,
            rel_slot[1] / config.max_range,

            rel_leader[0] / config.max_range,
            rel_leader[1] / config.max_range,
            np.sin(dtheta),
            np.cos(dtheta),
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
        'lookahead_rel': obs[0:2] * config.max_range,
        'tangent_rel': float(np.arctan2(obs[2], obs[3])),
        'v_leader_est': float(obs[4] * config.v_max),
        'w_leader_est': float(obs[5] * config.w_max),
        'leader_age': float(obs[6] * config.age_max),
        'v_follower_est': float(obs[7] * config.v_max),
        'w_follower_est': float(obs[8] * config.w_max),
        'follower_age': float(obs[9] * config.age_max),
        'slot_rel': obs[10:12] * config.max_range,
        'leader_rel': obs[12:14] * config.max_range,
        'dtheta': float(np.arctan2(obs[14], obs[15])),
    }


def scale_action(action, config):
    """Normalized action in [-1, 1]^4 -> physical commands for both robots.

    Returns ``((leader_v, leader_w), (follower_v, follower_w))`` in m/s and
    rad/s. Out-of-range actions are clipped, so a policy that has not learned
    to respect its own output range cannot exceed the robots' limits.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f'expected action of shape ({ACTION_DIM},), got {action.shape}')
    clipped = np.clip(action, -1.0, 1.0)
    return (
        (float(clipped[0] * config.v_max), float(clipped[1] * config.w_max)),
        (float(clipped[2] * config.v_max), float(clipped[3] * config.w_max)),
    )


def unscale_action(leader_command, follower_command, config):
    """Physical ``(v, w)`` per robot -> one normalized action in [-1, 1]^4."""
    leader_v, leader_w = leader_command
    follower_v, follower_w = follower_command
    return np.clip(
        np.array(
            [
                leader_v / config.v_max,
                leader_w / config.w_max,
                follower_v / config.v_max,
                follower_w / config.w_max,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )


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
