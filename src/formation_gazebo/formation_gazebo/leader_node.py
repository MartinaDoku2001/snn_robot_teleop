"""Leader node: drives /robot1 around the reference path with pure pursuit.

Uses formation_core's :class:`PurePursuitLeader` and :class:`NoiseProcess`
unchanged, so the leader behaves the same way as in the fast twin.

One deliberate difference from the fast twin, because we cannot reach inside
Gazebo's physics: the fast twin adds process noise to the leader's REALISED
velocities, while here it is added to the COMMANDED velocities, which the robot
then tracks through its own dynamics. The effect on predictability -- the thing
the task cares about -- is the same, but the two are not step-identical.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from formation_core.config import EpisodeConfig
from formation_core.controllers import PurePursuitLeader
from formation_core.dynamics import NoiseProcess
from formation_core.paths import make_path

from .ros_interface import RobotBridge, wait_for_bridges


class LeaderNode(Node):

    def __init__(self):
        super().__init__('formation_leader')
        self.declare_parameter('config', '')
        self.declare_parameter('namespace', 'robot1')
        self.declare_parameter('seed', -1)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        seed = int(self.get_parameter('seed').value)
        if seed >= 0:
            self.config = self.config.with_overrides(seed=seed)

        self.path = make_path(self.config.path.name, **self.config.path.params)
        self.controller = PurePursuitLeader(
            self.path,
            target_speed=self.config.leader.target_speed,
            lookahead=self.config.leader.lookahead,
            lookahead_gain=self.config.leader.lookahead_gain,
            curvature_slowdown=self.config.leader.curvature_slowdown,
            limits=self.config.limits)
        # Same stream layout as the fast twin: leader noise is stream 0.
        rng = np.random.default_rng(np.random.SeedSequence(self.config.seed).spawn(2)[0])
        self.noise = NoiseProcess(self.config.noise, rng)

        self.bridge = RobotBridge(self, self.get_parameter('namespace').value)
        if not wait_for_bridges(self, [self.bridge], timeout=60.0):
            self.get_logger().error('no odometry from the leader; is the sim running?')
        else:
            self.get_logger().info(
                f'leader ready on {self.bridge.namespace}, path {self.path.name}, '
                f'target speed {self.config.leader.target_speed} m/s')
        self.timer = self.create_timer(self.config.dt, self.on_timer)

    def on_timer(self):
        state = self.bridge.state
        if state is None:
            return
        v, w = self.controller.command(state)
        dv, dw = self.noise.sample()
        v, w = self.config.limits.clamp_command(v + dv, w + dw)
        self.bridge.publish_command(v, w)

    def destroy_node(self):
        self.bridge.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LeaderNode()
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
