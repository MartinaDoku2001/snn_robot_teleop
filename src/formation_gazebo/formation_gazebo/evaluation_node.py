"""Evaluation node: measures a Gazebo episode with the SAME metrics code.

It subscribes to ground truth for BOTH robots (evaluation may look at the truth
-- the controller may not), plus the estimate and the transmit flag, assembles
exactly the ``info`` dictionary the fast twin produces, and feeds it to
formation_core's :class:`EpisodeRecorder`. The CSV columns, the metric
definitions and the summary are therefore identical to the fast-twin ones,
which is what makes sim-to-sim comparison meaningful.

The recorded ``policy`` and ``controller`` are the ones the comm interface and
controller nodes announce on ``/formation/policy`` and ``/formation/controller``
-- the effective components, after launch-argument overrides -- not this node's
copy of the YAML.

Writes ``<out>/episode.csv``, ``<out>/metrics.csv`` and ``<out>/config.yaml``,
then shuts the episode down.
"""

from __future__ import annotations

import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Float64

from formation_core.config import EpisodeConfig
from formation_core.contract import formation_errors
from formation_core.metrics import EpisodeRecorder, write_rows_csv
from formation_core.paths import make_path

from .ros_interface import (
    RobotBridge,
    odometry_to_state,
    subscribe_announced,
    wait_for_bridges,
)


class EvaluationNode(Node):

    def __init__(self):
        super().__init__('formation_evaluation')
        self.declare_parameter('config', '')
        self.declare_parameter('leader_namespace', 'robot1')
        self.declare_parameter('follower_namespace', 'robot2')
        self.declare_parameter('out', 'results/gazebo')
        self.declare_parameter('duration', -1.0)
        self.declare_parameter('label', 'gazebo')
        # Skip the start-up transient, where the follower is still closing on
        # its slot and the numbers say nothing about the policy.
        self.declare_parameter('settle_seconds', 2.0)
        self.declare_parameter('shutdown_when_done', True)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        duration = float(self.get_parameter('duration').value)
        if duration > 0.0:
            self.config = self.config.with_overrides(duration=duration)

        self.out_dir = self.get_parameter('out').value
        self.settle = float(self.get_parameter('settle_seconds').value)
        self.path = make_path(self.config.path.name, **self.config.path.params)
        self.recorder = EpisodeRecorder(dt=self.config.dt)
        self.recorder.metadata['backend'] = 'gazebo'
        self.recorder.metadata['seed'] = self.config.seed

        self.leader = RobotBridge(
            self, self.get_parameter('leader_namespace').value, publish_commands=False)
        self.follower = RobotBridge(
            self, self.get_parameter('follower_namespace').value, publish_commands=False)
        self.estimate = None
        # The policy and controller actually in use, as announced by the nodes
        # that built them. The launch file's policy:= / controller:= overrides
        # never reach THIS node's config, so reading them from the YAML would
        # report the default no matter what the episode really ran.
        self.announced_policy = None
        self.announced_controller = None
        self.transmitted = False
        self.prediction_error = float('nan')
        self.action = (float('nan'), float('nan'))

        self.create_subscription(
            Odometry, '/formation/leader_estimate', self._on_estimate, 10)
        self.create_subscription(Bool, '/formation/transmitted', self._on_transmitted, 10)
        self.create_subscription(
            Float64, '/formation/prediction_error', self._on_prediction_error, 10)
        self.create_subscription(
            Float32MultiArray, '/formation/action', self._on_action, 10)
        subscribe_announced(self, '/formation/policy', self._on_policy)
        subscribe_announced(self, '/formation/controller', self._on_controller)

        if not wait_for_bridges(self, [self.leader, self.follower], timeout=60.0):
            self.get_logger().error('missing odometry; is the sim running?')

        self.step_index = 0
        self.start_time = None
        self.finished = False
        self.timer = self.create_timer(self.config.dt, self.on_timer)
        self.get_logger().info(
            f'evaluating {self.config.steps} steps ({self.config.duration:.1f} s) '
            f'after a {self.settle:.1f} s settling period')

    # ------------------------------------------------------------- callbacks

    def _on_estimate(self, msg):
        self.estimate = odometry_to_state(msg)

    def _on_transmitted(self, msg):
        # Latched until the next recorded step, so a message is never missed
        # between control ticks.
        self.transmitted = self.transmitted or bool(msg.data)

    def _on_prediction_error(self, msg):
        self.prediction_error = float(msg.data)

    def _on_policy(self, component):
        self.announced_policy = component

    def _on_controller(self, component):
        self.announced_controller = component

    def _on_action(self, msg):
        if len(msg.data) >= 2:
            self.action = (float(msg.data[0]), float(msg.data[1]))

    # ------------------------------------------------------------ recording

    def on_timer(self):
        if self.finished:
            return
        leader, follower = self.leader.state, self.follower.state
        if leader is None or follower is None or self.estimate is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        if elapsed < self.settle:
            self.transmitted = False
            return

        errors = formation_errors(follower, leader, self.config.offset_d)
        self.recorder.record({
            'step': self.step_index,
            'time': elapsed - self.settle,
            'leader_state': leader,
            'follower_state': follower,
            'leader_estimate': self.estimate,
            'transmitted': self.transmitted,
            'received': self.transmitted,
            'age': 0,
            'prediction_error': self.prediction_error,
            'estimate_error': float(
                ((leader.x - self.estimate.x) ** 2 + (leader.y - self.estimate.y) ** 2) ** 0.5),
            'errors': errors,
            'path_error': float(self.path.tracking_error(leader.xy)),
            'action': self.action,
            'reward': float('nan'),
        })
        self.transmitted = False
        self.step_index += 1

        if self.step_index >= self.config.steps:
            self.finish()

    def effective_config(self):
        """This node's config with the ANNOUNCED policy and controller in it.

        Falls back to the YAML for anything not announced, so a run with no
        overrides -- or with a node that did not come up -- still records
        something truthful.
        """
        overrides = {}
        if self.announced_policy is not None:
            overrides['policy'] = self.announced_policy
        if self.announced_controller is not None:
            overrides['controller'] = self.announced_controller
        return self.config.with_overrides(**overrides) if overrides else self.config

    def finish(self):
        if self.finished:
            return
        self.finished = True
        os.makedirs(self.out_dir, exist_ok=True)
        csv_path = self.recorder.write_csv(os.path.join(self.out_dir, 'episode.csv'))
        metrics = self.recorder.metrics()
        metrics['label'] = self.get_parameter('label').value
        effective = self.effective_config()
        metrics['policy'] = effective.policy.describe()
        metrics['controller'] = effective.controller.describe()
        write_rows_csv(os.path.join(self.out_dir, 'metrics.csv'), [metrics])
        effective.to_yaml(os.path.join(self.out_dir, 'config.yaml'))

        self.get_logger().info(
            'EPISODE COMPLETE  '
            f"steps={metrics['steps']}  "
            f"formation_rms={metrics['formation_rms']:.4f} m  "
            f"formation_max={metrics['formation_max']:.4f} m  "
            f"lateral_rms={metrics['lateral_rms']:.4f} m  "
            f"path_rms={metrics['path_rms']:.4f} m  "
            f"comm_rate={metrics['comm_rate']:.4f} ({metrics['messages']} messages)")
        self.get_logger().info(f'wrote {csv_path}')
        if bool(self.get_parameter('shutdown_when_done').value):
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        # Write whatever was recorded, even on an early Ctrl-C.
        node.finish()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
