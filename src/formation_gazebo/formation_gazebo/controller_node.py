"""Follower controller node.

Builds the FROZEN contract observation from the follower's own odometry plus
the leader ESTIMATE published by the comm interface, runs a
:class:`formation_core.controllers.Controller`, and publishes the scaled action
to ``/robot2/cmd_vel``.

This node is the drop-in point for later phases: swapping the ``controller``
parameter from ``analytic`` to ``rl`` or ``snn`` is the ONLY change needed here,
because the observation, the action scaling and the topics are fixed by the
contract.

Subscribed:  /<follower>/odom, /formation/leader_estimate
Published:   /<follower>/cmd_vel, /formation/observation (Float32MultiArray),
             /formation/action (Float32MultiArray),
             /formation/controller (std_msgs/String, latched: the EFFECTIVE controller)
"""

from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from formation_core.config import ComponentConfig, EpisodeConfig
from formation_core.contract import build_observation, scale_action
from formation_core.controllers import make_controller

from .ros_interface import RobotBridge, announce, odometry_to_state, wait_for_bridges


class ControllerNode(Node):

    def __init__(self):
        super().__init__('formation_controller')
        self.declare_parameter('config', '')
        self.declare_parameter('follower_namespace', 'robot2')
        self.declare_parameter('controller', '')
        # Stop commanding if the estimate stream dies, rather than driving on
        # stale data forever.
        self.declare_parameter('estimate_timeout', 1.0)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        name = self.get_parameter('controller').value or self.config.controller.name
        params = self.config.controller.params if name == self.config.controller.name else {}
        # The REQUESTED component, as the fast twin records it: the constructed
        # controller would also carry its default gains, and the two backends'
        # CSVs have to stay comparable string-for-string.
        self.controller_cfg = ComponentConfig(name, params)
        self.controller = make_controller(name, config=self.config.contract, **params)
        self.controller.reset()

        self.bridge = RobotBridge(self, self.get_parameter('follower_namespace').value)
        self.estimate = None
        self.estimate_time = None
        self.create_subscription(
            Odometry, '/formation/leader_estimate', self._on_estimate, 10)
        self.obs_pub = self.create_publisher(
            Float32MultiArray, '/formation/observation', 10)
        self.action_pub = self.create_publisher(
            Float32MultiArray, '/formation/action', 10)
        # Same reason as the comm interface's policy announcement: the
        # controller:= override lives here, so the effective name is published
        # here too.
        self.controller_pub = announce(self, '/formation/controller', self.controller_cfg)

        if not wait_for_bridges(self, [self.bridge], timeout=60.0):
            self.get_logger().error('no follower odometry; is the sim running?')
        self.get_logger().info(
            f'controller ready: {self.controller.name} on '
            f'{self.bridge.namespace}, offset {self.config.offset_d} m')
        self.timer = self.create_timer(self.config.dt, self.on_timer)

    def _on_estimate(self, msg):
        # Already in the world frame: the comm interface publishes it there.
        self.estimate = odometry_to_state(msg)
        self.estimate_time = self.get_clock().now()

    def on_timer(self):
        follower = self.bridge.state
        if follower is None or self.estimate is None:
            return

        timeout = float(self.get_parameter('estimate_timeout').value)
        age = (self.get_clock().now() - self.estimate_time).nanoseconds * 1e-9
        if timeout > 0.0 and age > timeout:
            self.bridge.stop()
            self.get_logger().warn(
                f'no leader estimate for {age:.2f} s; holding still',
                throttle_duration_sec=5.0)
            return

        obs = build_observation(
            follower, self.estimate, self.config.offset_d, self.config.contract)
        action = self.controller.act(obs)
        v, w = scale_action(action, self.config.contract)
        self.bridge.publish_command(v, w)

        self.obs_pub.publish(Float32MultiArray(data=[float(x) for x in obs]))
        self.action_pub.publish(Float32MultiArray(data=[float(x) for x in action]))

    def destroy_node(self):
        self.bridge.stop()
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
