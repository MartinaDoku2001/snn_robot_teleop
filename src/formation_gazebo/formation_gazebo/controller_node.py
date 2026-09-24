"""Centralized controller node (contract v2.0).

ONE controller drives BOTH robots. It builds the frozen 16-dim observation from
the two estimates the per-robot communication interfaces publish, calls
:meth:`formation_core.controllers.Controller.act` once, and publishes the four
resulting numbers as two ``cmd_vel`` messages.

It never reads either robot's odometry directly. Everything it knows about the
robots arrives through ``/formation/<ns>/estimate`` and ``/formation/<ns>/age``,
which is what makes the communication policy matter and what keeps the Gazebo
backend honest against the fast twin.

This node is the drop-in point for later phases: swapping the ``controller``
parameter from ``analytic`` to ``rl`` or ``snn`` is the ONLY change needed here,
because the observation, the action scaling and the topics are fixed by the
contract.

The reference path is built from the episode config, not received over a topic.
It is a static definition of the task rather than a measurement, and the
reference-path publisher builds it from the same config (see ``leader_node``).

Subscribed:  /formation/<leader>/estimate, /formation/<leader>/age
             /formation/<follower>/estimate, /formation/<follower>/age
Published:   /<leader>/cmd_vel, /<follower>/cmd_vel,
             /formation/observation (Float32MultiArray, 16),
             /formation/action (Float32MultiArray, 4),
             /formation/controller (std_msgs/String, latched: the controller)
"""

from __future__ import annotations

import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32

from formation_core.config import ComponentConfig, EpisodeConfig
from formation_core.contract import build_observation, scale_action
from formation_core.controllers import CONTROLLERS, make_controller
from formation_core.dynamics import NoiseProcess
from formation_core.paths import make_path

from .ros_interface import RobotBridge, announce, odometry_to_state


#: Optional packages that register a controller when imported.
CONTROLLER_PROVIDERS = {'rl': 'formation_rl', 'snn': 'formation_snn'}


def _import_provider(name):
    provider = CONTROLLER_PROVIDERS.get(name)
    if provider is None:
        return
    import importlib

    importlib.import_module(provider)


class ControllerNode(Node):

    def __init__(self):
        super().__init__('formation_controller')
        self.declare_parameter('config', '')
        self.declare_parameter('leader_namespace', 'robot1')
        self.declare_parameter('follower_namespace', 'robot2')
        self.declare_parameter('controller', '')
        #: Learned controllers only: which trained actor to run. Left empty,
        #: RLController falls back to its own default path, which depends on
        #: the working directory -- fine from a shell, unreliable under a
        #: launch file, so the launch file passes this explicitly.
        self.declare_parameter('weights', '')
        # Stop commanding if either estimate stream dies, rather than driving
        # on stale data forever.
        self.declare_parameter('estimate_timeout', 1.0)
        #: Leader process noise. The fast twin perturbs the leader's REALISED
        #: velocities; Gazebo's physics is not reachable, so the same
        #: disturbance is injected into the leader's command here. Without it
        #: the leader is perfectly predictable and the task has no content.
        self.declare_parameter('leader_noise', True)
        self.declare_parameter('seed', -1)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        seed = int(self.get_parameter('seed').value)
        if seed >= 0:
            self.config = self.config.with_overrides(seed=seed)
        name = self.get_parameter('controller').value or self.config.controller.name
        params = dict(
            self.config.controller.params
            if name == self.config.controller.name else {})
        weights = self.get_parameter('weights').value
        if weights:
            params['weights'] = weights
        if name not in CONTROLLERS:
            # Importing the provider is what registers a learned controller;
            # formation_core never imports it for us.
            _import_provider(name)
        # The REQUESTED component, as the fast twin records it: the constructed
        # controller would also carry its default gains, and the two backends'
        # CSVs have to stay comparable string-for-string. A weights path is
        # recorded by basename for the same reason -- the fast twin does, and
        # the absolute path differs between a container and a workstation.
        announced = dict(params)
        if 'weights' in announced:
            announced['weights'] = os.path.basename(announced['weights'])
        self.controller_cfg = ComponentConfig(name, announced)
        self.controller = make_controller(
            name, config=self.config.contract, leader=self.config.leader, **params)
        self.controller.reset()

        self.path = make_path(self.config.path.name, **self.config.path.params)
        # Stream 0, matching the fast twin's leader-noise stream.
        rng = np.random.default_rng(
            np.random.SeedSequence(self.config.seed).spawn(3)[0])
        self.noise = NoiseProcess(self.config.noise, rng)

        self.leader = self.get_parameter('leader_namespace').value.strip('/')
        self.follower = self.get_parameter('follower_namespace').value.strip('/')
        # Command-only bridges: this node publishes cmd_vel and must not read
        # odometry, or it would be seeing past the communication link.
        self.leader_bridge = RobotBridge(
            self, self.leader, subscribe_odometry=False, require_transform=False)
        self.follower_bridge = RobotBridge(
            self, self.follower, subscribe_odometry=False, require_transform=False)

        self.estimates = {self.leader: None, self.follower: None}
        self.ages = {self.leader: 0, self.follower: 0}
        self.estimate_times = {self.leader: None, self.follower: None}
        for robot in (self.leader, self.follower):
            self.create_subscription(
                Odometry, f'/formation/{robot}/estimate',
                lambda msg, r=robot: self._on_estimate(r, msg), 10)
            self.create_subscription(
                Int32, f'/formation/{robot}/age',
                lambda msg, r=robot: self.ages.__setitem__(r, int(msg.data)), 10)

        self.obs_pub = self.create_publisher(
            Float32MultiArray, '/formation/observation', 10)
        self.action_pub = self.create_publisher(
            Float32MultiArray, '/formation/action', 10)
        # Same reason as the comm interface's policy announcement: the
        # controller:= override lives here, so the effective name is published
        # here too.
        self.controller_pub = announce(
            self, '/formation/controller', self.controller_cfg)

        self.get_logger().info(
            f'centralized controller ready: {self.controller.name} driving '
            f'{self.leader} and {self.follower}, offset {self.config.offset_d} m')
        self.timer = self.create_timer(self.config.dt, self.on_timer)

    # ------------------------------------------------------------- callbacks

    def _on_estimate(self, robot, msg):
        # Already in the world frame: the comm interface publishes it there.
        self.estimates[robot] = odometry_to_state(msg)
        self.estimate_times[robot] = self.get_clock().now()

    def _stale(self):
        """Seconds since the oldest estimate, or None if one has never arrived."""
        if any(t is None for t in self.estimate_times.values()):
            return None
        now = self.get_clock().now()
        return max((now - t).nanoseconds * 1e-9 for t in self.estimate_times.values())

    # ------------------------------------------------------------------ loop

    def on_timer(self):
        stale = self._stale()
        if stale is None:
            return

        timeout = float(self.get_parameter('estimate_timeout').value)
        if timeout > 0.0 and stale > timeout:
            self.leader_bridge.stop()
            self.follower_bridge.stop()
            self.get_logger().warn(
                f'no estimate for {stale:.2f} s; holding both robots still',
                throttle_duration_sec=5.0)
            return

        leader_estimate = self.estimates[self.leader]
        follower_estimate = self.estimates[self.follower]
        lookahead_xy, tangent_xy = self.path.lookahead_pose(
            leader_estimate.xy, self.config.contract.lookahead_distance)
        obs = build_observation(
            leader_estimate, follower_estimate, lookahead_xy, tangent_xy,
            self.ages[self.leader], self.ages[self.follower],
            self.config.offset_d, self.config.contract)

        action = self.controller.act(obs)
        leader_command, follower_command = scale_action(action, self.config.contract)

        if bool(self.get_parameter('leader_noise').value):
            dv, dw = self.noise.sample()
            leader_command = self.config.limits.clamp_command(
                leader_command[0] + dv, leader_command[1] + dw)

        self.leader_bridge.publish_command(*leader_command)
        self.follower_bridge.publish_command(*follower_command)

        self.obs_pub.publish(Float32MultiArray(data=[float(x) for x in obs]))
        self.action_pub.publish(Float32MultiArray(data=[float(x) for x in action]))

    def destroy_node(self):
        self.leader_bridge.stop()
        self.follower_bridge.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
