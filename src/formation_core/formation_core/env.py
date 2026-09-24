"""The environment interface and its fast pure-Python backend.

:class:`FormationEnv` is the ONE interface. Two backends implement it:

* :class:`FastFormationEnv` (here) -- a kinematic twin, thousands of steps per
  second, where RL will train.
* ``formation_gazebo.GazeboFormationEnv`` -- the same contract driven by the
  Phase 0 Gazebo topics.

The same controller, transmission policy and metrics code runs unchanged on
both, which is what makes sim-to-sim (and later sim-to-real) comparison a
like-for-like claim rather than an approximation.

The API mirrors Gymnasium's: ``reset(seed=None, options=None) -> (obs, info)``
and ``step(action) -> (obs, reward, terminated, truncated, info)``. No RL
library is imported.

Contract v2.0: the action drives BOTH robots, and there are TWO communication
interfaces -- one per robot, each with its own transmission policy, its own
predictor and its own estimate. Both report *up* to the coordinator, which is
where the controller sits; the command link back down is assumed reliable
(Phase 3 models the uplink, and says so).
"""

from __future__ import annotations

import abc

import numpy as np

from .comm import CommInterface
from .contract import (
    ACTION_DIM,
    action_space,
    build_observation,
    formation_errors,
    observation_space,
    scale_action,
    slot_position,
)
from .dynamics import NoiseProcess, unicycle_step
from .geometry import RobotState
from .paths import make_path
from .policies import make_policy
from .predictor import make_predictor


class FormationEnv(abc.ABC):
    """Gymnasium-shaped interface for the leader-follower task."""

    #: Normalized spaces from the frozen contract.
    observation_space = None
    action_space = None

    @abc.abstractmethod
    def reset(self, *, seed=None, options=None):
        """Return ``(observation, info)``."""

    @abc.abstractmethod
    def step(self, action):
        """Return ``(observation, reward, terminated, truncated, info)``."""

    def close(self):
        """Release resources. No-op for the fast twin."""


class FastFormationEnv(FormationEnv):
    """Pure-Python kinematic twin of the leader-follower task.

    One step is: the centralized action drives both robots -> both integrate
    (the leader with bounded process noise) -> each robot's comm interface
    refreshes or dead-reckons the coordinator's estimate of it -> metrics.

    Args:
        config: an :class:`~formation_core.config.EpisodeConfig`.
    """

    def __init__(self, config):
        self.config = config
        self.observation_space = observation_space(config.contract)
        self.action_space = action_space(config.contract)

        self.path = make_path(config.path.name, **config.path.params)
        self.leader_comm = None
        self.follower_comm = None
        self.leader_state = RobotState()
        self.follower_state = RobotState()
        self.step_index = 0
        self._noise = None
        self._rng = None
        self._last_leader_result = None
        self._last_follower_result = None

    # ------------------------------------------------------------------ setup

    def _build_comm(self):
        """One interface per robot.

        Both are built from the same ``policy`` config: Phase 2 asks the same
        question of each robot. Phase 4 can give them separate policies without
        touching anything here, because they are already separate objects with
        separate RNG streams.
        """
        cfg = self.config
        return CommInterface(
            policy=make_policy(cfg.policy.name, **cfg.policy.params),
            predictor=make_predictor(cfg.predictor.name, **cfg.predictor.params),
            dt=cfg.dt, channel=None)

    def _initial_states(self):
        """Leader on the path at s=0; follower exactly in its slot behind it."""
        x, y, theta = self.path.start_pose()
        leader = RobotState(x=x, y=y, theta=theta, v=0.0, w=0.0)
        slot = slot_position(leader, self.config.offset_d + self.config.start_offset)
        follower = RobotState(x=float(slot[0]), y=float(slot[1]), theta=theta, v=0.0, w=0.0)
        return leader, follower

    # ------------------------------------------------------------------- gym

    def reset(self, *, seed=None, options=None):
        """Deterministic reset. Same seed => identical episode."""
        del options
        seed = self.config.seed if seed is None else int(seed)
        # Independent streams: leader noise and each robot's stochastic policy
        # must not perturb one another, so changing one robot's policy never
        # alters the leader's motion or the other robot's draws.
        seeds = np.random.SeedSequence(seed).spawn(3)
        self._rng = np.random.default_rng(seeds[0])
        leader_policy_rng = np.random.default_rng(seeds[1])
        follower_policy_rng = np.random.default_rng(seeds[2])

        self._noise = NoiseProcess(self.config.noise, self._rng)
        self.leader_state, self.follower_state = self._initial_states()

        self.leader_comm = self._build_comm()
        self.follower_comm = self._build_comm()
        self.leader_comm.reset(self.leader_state, leader_policy_rng)
        self.follower_comm.reset(self.follower_state, follower_policy_rng)

        self.step_index = 0
        leader_result = self.leader_comm.update(self.leader_state, time=0.0)
        follower_result = self.follower_comm.update(self.follower_state, time=0.0)
        self._last_leader_result = leader_result
        self._last_follower_result = follower_result

        obs = self._observation(leader_result, follower_result)
        info = self._info(leader_result, follower_result,
                          action=np.zeros(ACTION_DIM), reward=0.0)
        return obs, info

    def step(self, action):
        cfg = self.config
        leader_command, follower_command = scale_action(action, cfg.contract)

        # 1. BOTH robots apply the command the centralized controller issued
        #    for this step. The leader carries the bounded process noise, which
        #    is what makes it imperfectly predictable.
        self.leader_state = unicycle_step(
            self.leader_state, leader_command[0], leader_command[1], cfg.dt, cfg.limits,
            noise=self._noise.sample())
        self.follower_state = unicycle_step(
            self.follower_state, follower_command[0], follower_command[1],
            cfg.dt, cfg.limits)

        self.step_index += 1
        time = self.step_index * cfg.dt

        # 2. each robot reports up to the coordinator, or is dead-reckoned
        leader_result = self.leader_comm.update(self.leader_state, time=time)
        follower_result = self.follower_comm.update(self.follower_state, time=time)
        self._last_leader_result = leader_result
        self._last_follower_result = follower_result

        # 3. observation for the NEXT action, from the two estimates only
        obs = self._observation(leader_result, follower_result)

        errors = formation_errors(self.follower_state, self.leader_state, cfg.offset_d)
        path_error = float(self.path.tracking_error(self.leader_state.xy))
        reward = self._reward(errors, path_error, leader_result, follower_result, action)
        terminated = bool(
            errors['euclidean'] > cfg.max_formation_error
            or path_error > cfg.max_path_error)
        truncated = bool(self.step_index >= cfg.steps)
        info = self._info(leader_result, follower_result, action=action,
                          reward=reward, errors=errors, path_error=path_error)
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- helpers

    def _observation(self, leader_result, follower_result):
        """The 16-dim contract observation, from ESTIMATES only.

        The lookahead is taken from the leader's ESTIMATED pose, not its true
        one: the coordinator is where the controller runs, and it only knows
        what has been transmitted to it.
        """
        cfg = self.config
        lookahead_xy, tangent_xy = self.path.lookahead_pose(
            leader_result.estimate.xy, cfg.contract.lookahead_distance)
        return build_observation(
            leader_result.estimate, follower_result.estimate,
            lookahead_xy, tangent_xy,
            leader_result.age, follower_result.age,
            cfg.offset_d, cfg.contract)

    def _reward(self, errors, path_error, leader_result, follower_result, action):
        w = self.config.reward
        action = np.asarray(action, dtype=float).reshape(-1)
        messages = int(leader_result.transmitted) + int(follower_result.transmitted)
        return float(-(
            w.w_formation * errors['euclidean']
            + w.w_heading * abs(errors['heading'])
            + w.w_path * path_error
            + w.w_comm * messages
            + w.w_action * float(np.sum(action ** 2))))

    def _info(self, leader_result, follower_result, action, reward,
              errors=None, path_error=None):
        """Per-step diagnostics. Ground truth lives HERE, never in the observation."""
        if errors is None:
            errors = formation_errors(
                self.follower_state, self.leader_state, self.config.offset_d)
        if path_error is None:
            path_error = float(self.path.tracking_error(self.leader_state.xy))
        return {
            'step': self.step_index,
            'time': self.step_index * self.config.dt,
            'leader_state': self.leader_state.copy(),
            'follower_state': self.follower_state.copy(),
            # v1.x names keep their v1.x meaning -- the LEADER's estimate --
            # so existing analysis and plotting code keeps working.
            'leader_estimate': leader_result.estimate.copy(),
            'transmitted': leader_result.transmitted,
            'received': leader_result.received,
            'age': leader_result.age,
            'prediction_error': leader_result.prediction_error,
            'estimate_error': _estimate_error(self.leader_state, leader_result.estimate),
            # v2.0 adds the follower's own uplink alongside it.
            'follower_estimate': follower_result.estimate.copy(),
            'follower_transmitted': follower_result.transmitted,
            'follower_received': follower_result.received,
            'follower_age': follower_result.age,
            'follower_prediction_error': follower_result.prediction_error,
            'follower_estimate_error': _estimate_error(
                self.follower_state, follower_result.estimate),
            'messages': int(leader_result.transmitted) + int(follower_result.transmitted),
            'errors': errors,
            'path_error': path_error,
            'action': np.asarray(action, dtype=float).reshape(-1).copy(),
            'reward': reward,
        }

    @property
    def comm_stats(self):
        """The LEADER's statistics, for v1.x-compatible reporting."""
        return self.leader_comm.stats if self.leader_comm is not None else None

    @property
    def comm_stats_per_robot(self):
        """Both robots' statistics, keyed by role."""
        if self.leader_comm is None:
            return {}
        return {'leader': self.leader_comm.stats, 'follower': self.follower_comm.stats}

    @property
    def policy(self):
        """The transmission policy in use, with its operative parameters.

        Both robots run the same policy configuration in Phase 2, so the
        leader's instance describes both.
        """
        return self.leader_comm.policy if self.leader_comm is not None else None


def _estimate_error(true_state, estimate):
    return float(np.hypot(true_state.x - estimate.x, true_state.y - estimate.y))
