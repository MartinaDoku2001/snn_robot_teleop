"""Gazebo backend of the shared environment interface.

:class:`GazeboFormationEnv` implements
:class:`formation_core.env.FormationEnv`, so the same controller, transmission
policy, metrics and runner that drive the fast twin drive the real simulator::

    from formation_core.runner import run_episode
    from formation_gazebo.env import GazeboFormationEnv

    result = run_episode(config, env=GazeboFormationEnv(config))

That is the sim-to-sim consistency claim in one line of code, and it is the
same line that will later point at hardware.

Differences from the fast twin, all of them inherent to using a real simulator
rather than a kinematic model:

* Gazebo integrates the dynamics, so wheel slip, inertia and the diff-drive
  plugin's own limits apply; trajectories will not match step for step.
* ``reset()`` does NOT teleport the robots by default. It waits for odometry
  and starts from wherever they are, so an episode should be run on a freshly
  started sim (the launch file spawns them on the path).
* Leader process noise is added to commands rather than to realised velocities.

Contract v2.0: this env drives BOTH robots from one action and runs one
communication interface per robot, matching :class:`FastFormationEnv`. There is
no ``drive_leader`` switch any more -- a centralized controller that did not
command the leader would not be a centralized controller.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from formation_core.comm import CommInterface
from formation_core.contract import (
    ACTION_DIM,
    action_space,
    build_observation,
    formation_errors,
    observation_space,
    scale_action,
)
from formation_core.dynamics import NoiseProcess
from formation_core.env import FormationEnv
from formation_core.paths import make_path
from formation_core.policies import make_policy
from formation_core.predictor import make_predictor

from .ros_interface import RobotBridge, wait_for_bridges


class GazeboFormationEnv(FormationEnv):
    """The formation task, stepped against a running Gazebo simulation.

    Args:
        config: an :class:`~formation_core.config.EpisodeConfig`.
        node: optional existing rclpy Node (one is created if omitted).
        leader_namespace / follower_namespace: Phase 0 robot namespaces.
        leader_noise: add the leader's process noise to its command. Gazebo's
            physics cannot be perturbed directly, so this is where the fast
            twin's realised-velocity noise goes; without it the leader is
            perfectly predictable and the task has no content.
    """

    def __init__(self, config, node=None, leader_namespace='robot1',
                 follower_namespace='robot2', leader_noise=True):
        self.config = config
        self.observation_space = observation_space(config.contract)
        self.action_space = action_space(config.contract)

        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._owns_node = node is None
        self.node = node if node is not None else Node('formation_gazebo_env')

        self.leader_bridge = RobotBridge(self.node, leader_namespace)
        self.follower_bridge = RobotBridge(self.node, follower_namespace)
        self.leader_noise = bool(leader_noise)

        self.path = make_path(config.path.name, **config.path.params)
        self.leader_comm = None
        self.follower_comm = None
        self.noise = None
        self.step_index = 0
        self._last_leader_result = None
        self._last_follower_result = None
        self._last_leader_command = (0.0, 0.0)
        self._next_tick = None

    # ------------------------------------------------------------- plumbing

    def _spin(self, seconds):
        """Spin the node for ``seconds`` of WALL time."""
        end = time.monotonic() + max(seconds, 0.0)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=min(0.01, max(seconds, 0.001)))

    def _wait_for_tick(self):
        """Hold the control period, spinning so callbacks keep arriving.

        Wall time is used, so an episode runs in real time against Gazebo
        whatever real-time factor the simulator achieves.
        """
        if self._next_tick is None:
            self._next_tick = time.monotonic()
        self._next_tick += self.config.dt
        while rclpy.ok():
            remaining = self._next_tick - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self.node, timeout_sec=min(remaining, 0.01))

    @property
    def leader_state(self):
        return self.leader_bridge.state

    @property
    def follower_state(self):
        return self.follower_bridge.state

    # ------------------------------------------------------------------ gym

    def reset(self, *, seed=None, options=None):
        del options
        seed = self.config.seed if seed is None else int(seed)
        # Same three-stream layout as the fast twin: leader noise, then one
        # policy stream per robot.
        seeds = np.random.SeedSequence(seed).spawn(3)
        self.noise = NoiseProcess(self.config.noise, np.random.default_rng(seeds[0]))

        if not wait_for_bridges(
                self.node, [self.leader_bridge, self.follower_bridge], timeout=60.0):
            raise RuntimeError(
                'no odometry on /robot1/odom and /robot2/odom; is the Phase 0 sim running?')

        self.leader_comm = self._build_comm()
        self.follower_comm = self._build_comm()
        self.leader_comm.reset(self.leader_state, np.random.default_rng(seeds[1]))
        self.follower_comm.reset(self.follower_state, np.random.default_rng(seeds[2]))

        self.step_index = 0
        self._next_tick = None
        self._last_leader_command = (0.0, 0.0)
        leader_result = self.leader_comm.update(self.leader_state, time=0.0)
        follower_result = self.follower_comm.update(self.follower_state, time=0.0)
        self._last_leader_result = leader_result
        self._last_follower_result = follower_result

        obs = self._observation(leader_result, follower_result)
        return obs, self._info(leader_result, follower_result,
                               action=np.zeros(ACTION_DIM), reward=0.0)

    def _build_comm(self):
        return CommInterface(
            policy=make_policy(self.config.policy.name, **self.config.policy.params),
            predictor=make_predictor(
                self.config.predictor.name, **self.config.predictor.params),
            dt=self.config.dt)

    def _observation(self, leader_result, follower_result):
        cfg = self.config
        lookahead_xy, tangent_xy = self.path.lookahead_pose(
            leader_result.estimate.xy, cfg.contract.lookahead_distance)
        return build_observation(
            leader_result.estimate, follower_result.estimate,
            lookahead_xy, tangent_xy,
            leader_result.age, follower_result.age,
            cfg.offset_d, cfg.contract)

    def step(self, action):
        cfg = self.config
        leader_command, follower_command = scale_action(action, cfg.contract)
        if self.leader_noise:
            dv, dw = self.noise.sample()
            leader_command = cfg.limits.clamp_command(
                leader_command[0] + dv, leader_command[1] + dw)
        self._last_leader_command = leader_command
        self.leader_bridge.publish_command(*leader_command)
        self.follower_bridge.publish_command(*follower_command)

        self._wait_for_tick()
        self.step_index += 1
        time_s = self.step_index * cfg.dt

        leader_result = self.leader_comm.update(self.leader_state, time=time_s)
        follower_result = self.follower_comm.update(self.follower_state, time=time_s)
        self._last_leader_result = leader_result
        self._last_follower_result = follower_result

        obs = self._observation(leader_result, follower_result)
        errors = formation_errors(self.follower_state, self.leader_state, cfg.offset_d)
        path_error = float(self.path.tracking_error(self.leader_state.xy))
        messages = int(leader_result.transmitted) + int(follower_result.transmitted)
        reward = -(cfg.reward.w_formation * errors['euclidean']
                   + cfg.reward.w_heading * abs(errors['heading'])
                   + cfg.reward.w_path * path_error
                   + cfg.reward.w_comm * messages)
        terminated = bool(
            errors['euclidean'] > cfg.max_formation_error
            or path_error > cfg.max_path_error)
        truncated = bool(self.step_index >= cfg.steps)
        return obs, reward, terminated, truncated, self._info(
            leader_result, follower_result, action=action, reward=reward,
            errors=errors, path_error=path_error)

    def _info(self, leader_result, follower_result, action, reward,
              errors=None, path_error=None):
        leader = self.leader_state
        follower = self.follower_state
        if errors is None:
            errors = formation_errors(follower, leader, self.config.offset_d)
        if path_error is None:
            path_error = float(self.path.tracking_error(leader.xy))
        return {
            'step': self.step_index,
            'time': self.step_index * self.config.dt,
            'leader_state': leader.copy(),
            'follower_state': follower.copy(),
            'leader_estimate': leader_result.estimate.copy(),
            'transmitted': leader_result.transmitted,
            'received': leader_result.received,
            'age': leader_result.age,
            'prediction_error': leader_result.prediction_error,
            'estimate_error': _distance(leader, leader_result.estimate),
            'follower_estimate': follower_result.estimate.copy(),
            'follower_transmitted': follower_result.transmitted,
            'follower_received': follower_result.received,
            'follower_age': follower_result.age,
            'follower_prediction_error': follower_result.prediction_error,
            'follower_estimate_error': _distance(follower, follower_result.estimate),
            'messages': (int(leader_result.transmitted)
                         + int(follower_result.transmitted)),
            'errors': errors,
            'path_error': path_error,
            'leader_command': self._last_leader_command,
            'action': np.asarray(action, dtype=float).reshape(-1).copy(),
            'reward': reward,
            'backend': 'gazebo',
        }

    @property
    def comm_stats(self):
        """The LEADER's statistics, for v1.x-compatible reporting."""
        return self.leader_comm.stats if self.leader_comm is not None else None

    @property
    def comm_stats_per_robot(self):
        if self.leader_comm is None:
            return {}
        return {'leader': self.leader_comm.stats, 'follower': self.follower_comm.stats}

    @property
    def policy(self):
        return self.leader_comm.policy if self.leader_comm is not None else None

    def close(self):
        self.follower_bridge.stop()
        self.leader_bridge.stop()
        self._spin(0.2)
        if self._owns_node:
            self.node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()


def _distance(true_state, estimate):
    return float(np.hypot(true_state.x - estimate.x, true_state.y - estimate.y))
