"""Communication interface node -- ONE INSTANCE PER ROBOT (contract v2.0).

Sits on one robot's side: reads that robot's true state and asks its
transmission policy whether to send it *up* to the coordinator, where the
centralized controller runs. Between transmissions it publishes the
dead-reckoned prediction instead, so no subscriber can see ground truth.

Under v1.x a single instance carried the leader's state to the follower. Under
v2.0 both robots report to the coordinator, so the launch file starts one of
these per robot, each with its own policy, its own predictor and its own
estimate. Every topic is namespaced by the robot it describes.

It runs formation_core's :class:`CommInterface` unchanged -- the same policies,
predictor and bookkeeping as the fast twin.

Published (for robot ``<ns>``):
    /formation/<ns>/estimate          nav_msgs/Odometry  (world frame)
    /formation/<ns>/transmitted       std_msgs/Bool      (sent this step?)
    /formation/<ns>/age               std_msgs/Int32     (steps since refresh)
    /formation/<ns>/prediction_error  std_msgs/Float64   (error a message erases)
    /formation/<ns>/policy            std_msgs/String    (latched: the policy)

Perfect communications in Phase 2 means ``always`` -- a message every step. The
:class:`~formation_core.comm.Channel` hook that delay and loss will use is
already inside :class:`CommInterface` and stays dormant here; Phase 3 fills it.
The downlink, the commands going back to the robots, is assumed reliable.
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

#: Which SeedSequence stream each robot's policy draws from. The fast twin
#: spawns three streams from the episode seed -- leader noise, then one policy
#: stream per robot -- and these indices keep the two backends aligned.
POLICY_STREAM = {'leader': 1, 'follower': 2}


class CommInterfaceNode(Node):

    def __init__(self):
        super().__init__('formation_comm_interface')
        self.declare_parameter('config', '')
        #: The robot this interface belongs to, e.g. 'robot1'.
        self.declare_parameter('robot', 'robot1')
        #: 'leader' or 'follower' -- picks the RNG stream, so a stochastic
        #: policy on one robot never consumes the other's draws.
        self.declare_parameter('role', 'leader')
        self.declare_parameter('seed', -1)
        # Policy overrides, so a sweep can vary the policy from the launch file
        # without rewriting the YAML.
        self.declare_parameter('policy', '')
        self.declare_parameter('k', 0)
        self.declare_parameter('p', -1.0)
        self.declare_parameter('delta', -1.0)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        seed = int(self.get_parameter('seed').value)
        if seed >= 0:
            self.config = self.config.with_overrides(seed=seed)
        self.robot = self.get_parameter('robot').value.strip('/')
        self.role = self.get_parameter('role').value
        policy_cfg = self._policy_config()

        policy = make_policy(policy_cfg.name, **policy_cfg.params)
        predictor = make_predictor(
            self.config.predictor.name, **self.config.predictor.params)
        self.comm = CommInterface(
            policy=policy, predictor=predictor, dt=self.config.dt)

        self.bridge = RobotBridge(self, self.robot, publish_commands=False)
        prefix = f'/formation/{self.robot}'
        self.estimate_pub = self.create_publisher(Odometry, f'{prefix}/estimate', 10)
        self.transmitted_pub = self.create_publisher(Bool, f'{prefix}/transmitted', 10)
        self.age_pub = self.create_publisher(Int32, f'{prefix}/age', 10)
        self.error_pub = self.create_publisher(Float64, f'{prefix}/prediction_error', 10)
        # Say which policy the overrides actually produced, so the evaluation
        # node records THAT rather than re-reading the un-overridden YAML.
        # The built policy, not policy_cfg: its params are the operative
        # k/p/delta, so a value that came from a default is recorded too.
        self.policy_pub = announce(self, f'{prefix}/policy', policy)

        if not wait_for_bridges(self, [self.bridge], timeout=60.0):
            self.get_logger().error(
                f'no odometry from {self.robot}; is the sim running?')
        else:
            self.comm.reset(self.bridge.state, self._policy_rng())
            self.get_logger().info(
                f'comm interface ready for {self.robot} ({self.role}): '
                f'policy {policy.describe()}, predictor {predictor.name}, '
                f'dt {self.config.dt}')
        self.timer = self.create_timer(self.config.dt, self.on_timer)

    def _policy_rng(self):
        stream = POLICY_STREAM.get(self.role, 1)
        seeds = np.random.SeedSequence(self.config.seed).spawn(3)
        return np.random.default_rng(seeds[stream])

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
            result.estimate, stamp, child_frame_id=f'{self.robot}/estimate'))
        self.transmitted_pub.publish(Bool(data=bool(result.transmitted)))
        self.age_pub.publish(Int32(data=int(result.age)))
        self.error_pub.publish(Float64(data=float(result.prediction_error)))

    def destroy_node(self):
        stats = self.comm.stats
        if stats.steps:
            self.get_logger().info(
                f'comm summary for {self.robot}: {stats.transmissions} messages '
                f'in {stats.steps} steps (rate {stats.rate:.4f})')
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
