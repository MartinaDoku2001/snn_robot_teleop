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
"""

from __future__ import annotations

import abc

import numpy as np

from .comm import CommInterface
from .contract import (
    action_space,
    build_observation,
    formation_errors,
    observation_space,
    scale_action,
    slot_position,
)
from .controllers import PurePursuitLeader
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

    One step is: leader pure-pursuit command -> leader integrates (with bounded
    process noise) -> comm interface updates the follower's estimate -> the
    follower applies the action given for THIS step -> metrics.

    Args:
        config: an :class:`~formation_core.config.EpisodeConfig`.
    """

    def __init__(self, config):
        self.config = config
        self.observation_space = observation_space(config.contract)
        self.action_space = action_space(config.contract)

        self.path = make_path(config.path.name, **config.path.params)
        self.leader_controller = PurePursuitLeader(
            self.path,
            target_speed=config.leader.target_speed,
            lookahead=config.leader.lookahead,
            lookahead_gain=config.leader.lookahead_gain,
            curvature_slowdown=config.leader.curvature_slowdown,
            limits=config.limits,
        )
        self.comm = None
        self.leader_state = RobotState()
        self.follower_state = RobotState()
        self.step_index = 0
        self._noise = None
        self._rng = None
        self._last_leader_command = (0.0, 0.0)
        self._last_comm = None

    # ------------------------------------------------------------------ setup

    def _build_comm(self, rng):
        cfg = self.config
        policy = make_policy(cfg.policy.name, **cfg.policy.params)
        predictor = make_predictor(cfg.predictor.name, **cfg.predictor.params)
        return CommInterface(policy=policy, predictor=predictor, dt=cfg.dt, channel=None)

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
        # Independent streams: leader noise and any stochastic policy must not
        # perturb one another, so a policy change never alters the leader.
        seeds = np.random.SeedSequence(seed).spawn(2)
        self._rng = np.random.default_rng(seeds[0])
        policy_rng = np.random.default_rng(seeds[1])

        self._noise = NoiseProcess(self.config.noise, self._rng)
        self.leader_state, self.follower_state = self._initial_states()
        self.leader_controller.reset()

        self.comm = self._build_comm(self._rng)
        self.comm.reset(self.leader_state, policy_rng)

        self.step_index = 0
        self._last_leader_command = (0.0, 0.0)

        result = self.comm.update(self.leader_state, time=0.0)
        self._last_comm = result
        obs = build_observation(
            self.follower_state, result.estimate, self.config.offset_d, self.config.contract)
        return obs, self._info(result, action=np.zeros(2), reward=0.0)

    def step(self, action):
        cfg = self.config
        v_cmd, w_cmd = scale_action(action, cfg.contract)

        # 1. follower applies the action it was given for this step
        self.follower_state = unicycle_step(
            self.follower_state, v_cmd, w_cmd, cfg.dt, cfg.limits)

        # 2. leader tracks the path, with bounded process noise
        leader_v, leader_w = self.leader_controller.command(self.leader_state)
        self._last_leader_command = (leader_v, leader_w)
        self.leader_state = unicycle_step(
            self.leader_state, leader_v, leader_w, cfg.dt, cfg.limits,
            noise=self._noise.sample())

        self.step_index += 1
        time = self.step_index * cfg.dt

        # 3. communication: refresh or dead-reckon the follower's estimate
        result = self.comm.update(self.leader_state, time=time)
        self._last_comm = result

        # 4. observation for the NEXT action, from the estimate only
        obs = build_observation(
            self.follower_state, result.estimate, cfg.offset_d, cfg.contract)

        errors = formation_errors(self.follower_state, self.leader_state, cfg.offset_d)
        reward = self._reward(errors, result, action)
        terminated = bool(errors['euclidean'] > cfg.max_formation_error)
        truncated = bool(self.step_index >= cfg.steps)
        info = self._info(result, action=action, reward=reward, errors=errors)
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- helpers

    def _reward(self, errors, comm_result, action):
        w = self.config.reward
        action = np.asarray(action, dtype=float).reshape(-1)
        return float(-(
            w.w_formation * errors['euclidean']
            + w.w_heading * abs(errors['heading'])
            + w.w_comm * float(comm_result.transmitted)
            + w.w_action * float(np.sum(action ** 2))))

    def _info(self, comm_result, action, reward, errors=None):
        """Per-step diagnostics. Ground truth lives HERE, never in the observation."""
        if errors is None:
            errors = formation_errors(
                self.follower_state, self.leader_state, self.config.offset_d)
        return {
            'step': self.step_index,
            'time': self.step_index * self.config.dt,
            'leader_state': self.leader_state.copy(),
            'follower_state': self.follower_state.copy(),
            'leader_estimate': comm_result.estimate.copy(),
            'transmitted': comm_result.transmitted,
            'received': comm_result.received,
            'age': comm_result.age,
            'prediction_error': comm_result.prediction_error,
            'estimate_error': float(np.hypot(
                self.leader_state.x - comm_result.estimate.x,
                self.leader_state.y - comm_result.estimate.y)),
            'errors': errors,
            'path_error': float(self.path.tracking_error(self.leader_state.xy)),
            'leader_command': self._last_leader_command,
            'action': np.asarray(action, dtype=float).reshape(-1).copy(),
            'reward': reward,
        }

    @property
    def comm_stats(self):
        return self.comm.stats if self.comm is not None else None
