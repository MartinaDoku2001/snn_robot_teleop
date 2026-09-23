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
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from formation_core.comm import CommInterface
from formation_core.contract import build_observation, formation_errors, scale_action
from formation_core.controllers import PurePursuitLeader
from formation_core.dynamics import NoiseProcess
from formation_core.env import FormationEnv
from formation_core.contract import action_space, observation_space
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
        drive_leader: if True this env also commands the leader; set False when
            the separate ``leader_node`` is running.
    """

    def __init__(self, config, node=None, leader_namespace='robot1',
                 follower_namespace='robot2', drive_leader=True):
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
        self.drive_leader = drive_leader

        self.path = make_path(config.path.name, **config.path.params)
        self.leader_controller = PurePursuitLeader(
            self.path,
            target_speed=config.leader.target_speed,
            lookahead=config.leader.lookahead,
            lookahead_gain=config.leader.lookahead_gain,
            curvature_slowdown=config.leader.curvature_slowdown,
            limits=config.limits)
        self.comm = None
        self.noise = None
        self.step_index = 0
        self._last_comm = None
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
        seeds = np.random.SeedSequence(seed).spawn(2)
        self.noise = NoiseProcess(self.config.noise, np.random.default_rng(seeds[0]))

        if not wait_for_bridges(
                self.node, [self.leader_bridge, self.follower_bridge], timeout=60.0):
            raise RuntimeError(
                'no odometry on /robot1/odom and /robot2/odom; is the Phase 0 sim running?')

        self.comm = CommInterface(
            policy=make_policy(self.config.policy.name, **self.config.policy.params),
            predictor=make_predictor(
                self.config.predictor.name, **self.config.predictor.params),
            dt=self.config.dt)
        self.comm.reset(self.leader_state, np.random.default_rng(seeds[1]))

        self.step_index = 0
        self._next_tick = None
        self._last_leader_command = (0.0, 0.0)
        result = self.comm.update(self.leader_state, time=0.0)
        self._last_comm = result
        obs = build_observation(
            self.follower_state, result.estimate, self.config.offset_d,
            self.config.contract)
        return obs, self._info(result, action=np.zeros(2), reward=0.0)

    def step(self, action):
        cfg = self.config
        v_cmd, w_cmd = scale_action(action, cfg.contract)
        self.follower_bridge.publish_command(v_cmd, w_cmd)

        if self.drive_leader:
            leader_v, leader_w = self.leader_controller.command(self.leader_state)
            dv, dw = self.noise.sample()
            leader_v, leader_w = cfg.limits.clamp_command(leader_v + dv, leader_w + dw)
            self._last_leader_command = (leader_v, leader_w)
            self.leader_bridge.publish_command(leader_v, leader_w)

        self._wait_for_tick()
        self.step_index += 1

        result = self.comm.update(self.leader_state, time=self.step_index * cfg.dt)
        self._last_comm = result
        obs = build_observation(
            self.follower_state, result.estimate, cfg.offset_d, cfg.contract)

        errors = formation_errors(self.follower_state, self.leader_state, cfg.offset_d)
        reward = -(cfg.reward.w_formation * errors['euclidean']
                   + cfg.reward.w_heading * abs(errors['heading'])
                   + cfg.reward.w_comm * float(result.transmitted))
        terminated = bool(errors['euclidean'] > cfg.max_formation_error)
        truncated = bool(self.step_index >= cfg.steps)
        return obs, reward, terminated, truncated, self._info(
            result, action=action, reward=reward, errors=errors)

    def _info(self, comm_result, action, reward, errors=None):
        leader = self.leader_state
        follower = self.follower_state
        if errors is None:
            errors = formation_errors(follower, leader, self.config.offset_d)
        return {
            'step': self.step_index,
            'time': self.step_index * self.config.dt,
            'leader_state': leader.copy(),
            'follower_state': follower.copy(),
            'leader_estimate': comm_result.estimate.copy(),
            'transmitted': comm_result.transmitted,
            'received': comm_result.received,
            'age': comm_result.age,
            'prediction_error': comm_result.prediction_error,
            'estimate_error': float(np.hypot(
                leader.x - comm_result.estimate.x, leader.y - comm_result.estimate.y)),
            'errors': errors,
            'path_error': float(self.path.tracking_error(leader.xy)),
            'leader_command': self._last_leader_command,
            'action': np.asarray(action, dtype=float).reshape(-1).copy(),
            'reward': reward,
            'backend': 'gazebo',
        }

    @property
    def comm_stats(self):
        return self.comm.stats if self.comm is not None else None

    def close(self):
        self.follower_bridge.stop()
        if self.drive_leader:
            self.leader_bridge.stop()
        self._spin(0.2)
        if self._owns_node:
            self.node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()
