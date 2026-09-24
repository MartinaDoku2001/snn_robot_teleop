"""Evaluation node: measures a Gazebo episode with the SAME metrics code.

It subscribes to ground truth for BOTH robots (evaluation may look at the truth
-- the controller may not), plus both estimates and both transmit flags,
assembles exactly the ``info`` dictionary the fast twin produces, and feeds it
to formation_core's :class:`EpisodeRecorder`. The CSV columns, the metric
definitions and the summary are therefore identical to the fast-twin ones,
which is what makes sim-to-sim comparison meaningful.

Contract v2.0: both robots have an uplink, so both are recorded. The
unprefixed columns stay the leader's, as in v1.x, and the follower's are
appended -- see :data:`formation_core.metrics.STEP_FIELDS`.

The recorded ``policy`` and ``controller`` are the ones the comm interface and
controller nodes announce on ``/formation/<ns>/policy`` and
``/formation/controller`` -- the effective components, after launch-argument
overrides -- not this node's copy of the YAML.

Writes ``<out>/episode.csv``, ``<out>/metrics.csv`` and ``<out>/config.yaml``,
then shuts the episode down.
"""

from __future__ import annotations

import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Float64, Int32

from formation_core.config import EpisodeConfig
from formation_core.contract import ACTION_DIM, formation_errors
from formation_core.metrics import EpisodeRecorder, write_rows_csv
from formation_core.paths import make_path

from .ros_interface import (
    RobotBridge,
    odometry_to_state,
    subscribe_announced,
    wait_for_bridges,
)


class _Uplink:
    """Everything this node hears about one robot's communication interface."""

    def __init__(self):
        self.estimate = None
        self.age = 0
        self.transmitted = False
        self.prediction_error = float('nan')
        self.policy = None

    def take_transmitted(self):
        """Read and clear the latch (see :meth:`EvaluationNode._on_transmitted`)."""
        value, self.transmitted = self.transmitted, False
        return value


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
        self.declare_parameter('seed', -1)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        seed = int(self.get_parameter('seed').value)
        if seed >= 0:
            self.config = self.config.with_overrides(seed=seed)
        duration = float(self.get_parameter('duration').value)
        if duration > 0.0:
            self.config = self.config.with_overrides(duration=duration)

        self.out_dir = self.get_parameter('out').value
        self.settle = float(self.get_parameter('settle_seconds').value)
        self.path = make_path(self.config.path.name, **self.config.path.params)
        self.recorder = EpisodeRecorder(dt=self.config.dt)
        self.recorder.metadata['backend'] = 'gazebo'
        self.recorder.metadata['seed'] = self.config.seed

        self.leader_ns = self.get_parameter('leader_namespace').value.strip('/')
        self.follower_ns = self.get_parameter('follower_namespace').value.strip('/')
        self.leader = RobotBridge(self, self.leader_ns, publish_commands=False)
        self.follower = RobotBridge(self, self.follower_ns, publish_commands=False)

        # The controller actually in use, as announced by the node that built
        # it. The launch file's controller:= override never reaches THIS node's
        # config, so reading it from the YAML would report the default no
        # matter what the episode really ran. Same for each robot's policy.
        self.announced_controller = None
        self.uplinks = {self.leader_ns: _Uplink(), self.follower_ns: _Uplink()}
        self.action = tuple([float('nan')] * ACTION_DIM)

        for robot in (self.leader_ns, self.follower_ns):
            uplink = self.uplinks[robot]
            prefix = f'/formation/{robot}'
            self.create_subscription(
                Odometry, f'{prefix}/estimate',
                lambda msg, u=uplink: setattr(u, 'estimate', odometry_to_state(msg)), 10)
            self.create_subscription(
                Bool, f'{prefix}/transmitted',
                lambda msg, u=uplink: self._on_transmitted(u, msg), 10)
            self.create_subscription(
                Int32, f'{prefix}/age',
                lambda msg, u=uplink: setattr(u, 'age', int(msg.data)), 10)
            self.create_subscription(
                Float64, f'{prefix}/prediction_error',
                lambda msg, u=uplink: setattr(u, 'prediction_error', float(msg.data)), 10)
            subscribe_announced(
                self, f'{prefix}/policy',
                lambda component, u=uplink: setattr(u, 'policy', component))
        self.create_subscription(
            Float32MultiArray, '/formation/action', self._on_action, 10)
        subscribe_announced(self, '/formation/controller', self._on_controller)

        if not wait_for_bridges(self, [self.leader, self.follower], timeout=60.0):
            self.get_logger().error('missing odometry; is the sim running?')

        self.step_index = 0
        self.start_time = None
        self.finished = False
        #: Leader arc length, so `distance` and the progress column mean the
        #: same thing here as they do in the fast twin instead of reading 0.
        self._arc = None
        self.timer = self.create_timer(self.config.dt, self.on_timer)
        self.get_logger().info(
            f'evaluating {self.config.steps} steps ({self.config.duration:.1f} s) '
            f'after a {self.settle:.1f} s settling period')

    # ------------------------------------------------------------- callbacks

    @staticmethod
    def _on_transmitted(uplink, msg):
        # Latched until the next recorded step, so a message is never missed
        # between control ticks.
        uplink.transmitted = uplink.transmitted or bool(msg.data)

    def _on_controller(self, component):
        self.announced_controller = component

    def _on_action(self, msg):
        if len(msg.data) >= ACTION_DIM:
            self.action = tuple(float(x) for x in msg.data[:ACTION_DIM])

    # ------------------------------------------------------------ recording

    def on_timer(self):
        if self.finished:
            return
        leader, follower = self.leader.state, self.follower.state
        leader_up = self.uplinks[self.leader_ns]
        follower_up = self.uplinks[self.follower_ns]
        if leader is None or follower is None:
            return
        if leader_up.estimate is None or follower_up.estimate is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        if elapsed < self.settle:
            leader_up.take_transmitted()
            follower_up.take_transmitted()
            return

        leader_tx = leader_up.take_transmitted()
        follower_tx = follower_up.take_transmitted()
        arc = self.path.arc_length_at(leader.xy)
        progress = 0.0 if self._arc is None else self.path.arc_delta(self._arc, arc)
        self._arc = arc
        errors = formation_errors(follower, leader, self.config.offset_d)
        self.recorder.record({
            'step': self.step_index,
            'time': elapsed - self.settle,
            'leader_state': leader,
            'follower_state': follower,
            'leader_estimate': leader_up.estimate,
            'transmitted': leader_tx,
            'received': leader_tx,
            'age': leader_up.age,
            'prediction_error': leader_up.prediction_error,
            'estimate_error': _distance(leader, leader_up.estimate),
            'follower_estimate': follower_up.estimate,
            'follower_transmitted': follower_tx,
            'follower_received': follower_tx,
            'follower_age': follower_up.age,
            'follower_prediction_error': follower_up.prediction_error,
            'follower_estimate_error': _distance(follower, follower_up.estimate),
            'messages': int(leader_tx) + int(follower_tx),
            'errors': errors,
            'path_error': float(self.path.tracking_error(leader.xy)),
            'progress': progress,
            'action': self.action,
            'reward': float('nan'),
        })
        self.step_index += 1

        if self.step_index >= self.config.steps:
            self.finish()

    def effective_config(self):
        """This node's config with the ANNOUNCED policy and controller in it.

        Falls back to the YAML for anything not announced, so a run with no
        overrides -- or with a node that did not come up -- still records
        something truthful. Both robots run the same policy configuration in
        Phase 2, so the leader's announcement describes the episode.
        """
        overrides = {}
        leader_policy = self.uplinks[self.leader_ns].policy
        if leader_policy is not None:
            overrides['policy'] = leader_policy
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
        metrics['contract_version'] = effective.contract_version
        write_rows_csv(os.path.join(self.out_dir, 'metrics.csv'), [metrics])
        effective.to_yaml(os.path.join(self.out_dir, 'config.yaml'))

        self.get_logger().info(
            'EPISODE COMPLETE  '
            f"steps={metrics['steps']}  "
            f"formation_rms={metrics['formation_rms']:.4f} m  "
            f"formation_max={metrics['formation_max']:.4f} m  "
            f"lateral_rms={metrics['lateral_rms']:.4f} m  "
            f"path_rms={metrics['path_rms']:.4f} m  "
            f"comm_rate={metrics['comm_rate']:.4f} ({metrics['messages']} messages)  "
            f"comm_rate_total={metrics['comm_rate_total']:.4f} "
            f"({metrics['messages_total']} messages)  "
            f"distance={metrics['distance']:.2f} m")
        self.get_logger().info(f'wrote {csv_path}')
        if bool(self.get_parameter('shutdown_when_done').value):
            raise SystemExit(0)


def _distance(true_state, estimate):
    return float(
        ((true_state.x - estimate.x) ** 2 + (true_state.y - estimate.y) ** 2) ** 0.5)


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
