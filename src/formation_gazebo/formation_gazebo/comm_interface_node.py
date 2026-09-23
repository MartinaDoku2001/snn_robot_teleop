"""Communication interface node.

Sits on the leader's side: reads the leader's true state, asks the transmission
policy whether to send it, and publishes what the FOLLOWER is allowed to know.
Between transmissions it publishes the dead-reckoned prediction instead, so a
subscriber physically cannot see ground truth.

It runs formation_core's :class:`CommInterface` unchanged -- the same policies,
predictor and bookkeeping as the fast twin.

Published:
    /formation/leader_estimate   nav_msgs/Odometry  (world frame)
    /formation/transmitted       std_msgs/Bool      (was a message sent this step)
    /formation/age               std_msgs/Int32     (steps since last refresh)
    /formation/prediction_error  std_msgs/Float64   (error a message would erase)
    /formation/policy            std_msgs/String    (latched: the EFFECTIVE policy)
"""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Int32

from formation_core.comm import CommInterface
from formation_core.config import ComponentConfig, EpisodeConfig
from formation_core.policies import make_policy
from formation_core.predictor import make_predictor

from .ros_interface import RobotBridge, announce, state_to_odometry, wait_for_bridges


class CommInterfaceNode(Node):

    def __init__(self):
        super().__init__('formation_comm_interface')
        self.declare_parameter('config', '')
        self.declare_parameter('leader_namespace', 'robot1')
        # Policy overrides, so a sweep can vary the policy from the launch file
        # without rewriting the YAML.
        self.declare_parameter('policy', '')
        self.declare_parameter('k', 0)
        self.declare_parameter('p', -1.0)
        self.declare_parameter('delta', -1.0)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        policy_cfg = self._policy_config()

        policy = make_policy(policy_cfg.name, **policy_cfg.params)
        predictor = make_predictor(
            self.config.predictor.name, **self.config.predictor.params)
        self.comm = CommInterface(
            policy=policy, predictor=predictor, dt=self.config.dt)

        self.bridge = RobotBridge(
            self, self.get_parameter('leader_namespace').value, publish_commands=False)
        self.estimate_pub = self.create_publisher(
            Odometry, '/formation/leader_estimate', 10)
        self.transmitted_pub = self.create_publisher(Bool, '/formation/transmitted', 10)
        self.age_pub = self.create_publisher(Int32, '/formation/age', 10)
        self.error_pub = self.create_publisher(Float64, '/formation/prediction_error', 10)
        # Say which policy the overrides actually produced, so the evaluation
        # node records THAT rather than re-reading the un-overridden YAML.
        # The built policy, not policy_cfg: its params are the operative
        # k/p/delta, so a value that came from a default is recorded too.
        self.policy_pub = announce(self, '/formation/policy', policy)

        if not wait_for_bridges(self, [self.bridge], timeout=60.0):
            self.get_logger().error('no leader odometry; is the sim running?')
        else:
            self.comm.reset(
                self.bridge.state,
                np.random.default_rng(np.random.SeedSequence(self.config.seed).spawn(2)[1]))
            self.get_logger().info(
                f'comm interface ready: policy {policy.describe()}, '
                f'predictor {predictor.name}, dt {self.config.dt}')
        self.timer = self.create_timer(self.config.dt, self.on_timer)

    def _policy_config(self):
        """Launch-parameter overrides beat the YAML."""
        name = self.get_parameter('policy').value
        if not name:
            return self.config.policy
        params = {}
        k = int(self.get_parameter('k').value)
        p = float(self.get_parameter('p').value)
        delta = float(self.get_parameter('delta').value)
        if name == 'periodic' and k > 0:
            params['k'] = k
        if name == 'random' and p >= 0.0:
            params['p'] = p
        if name == 'event_triggered' and delta >= 0.0:
            params['delta'] = delta
        return ComponentConfig(name, params)

    def on_timer(self):
        state = self.bridge.state
        if state is None:
            return
        result = self.comm.update(state)
        stamp = self.get_clock().now().to_msg()
        self.estimate_pub.publish(state_to_odometry(
            result.estimate, stamp, child_frame_id='leader_estimate'))
        self.transmitted_pub.publish(Bool(data=bool(result.transmitted)))
        self.age_pub.publish(Int32(data=int(result.age)))
        self.error_pub.publish(Float64(data=float(result.prediction_error)))

    def destroy_node(self):
        stats = self.comm.stats
        if stats.steps:
            self.get_logger().info(
                f'comm summary: {stats.transmissions} messages in {stats.steps} steps '
                f'(rate {stats.rate:.4f})')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CommInterfaceNode()
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
