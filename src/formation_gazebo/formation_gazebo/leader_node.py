"""Reference-path publisher (contract v2.0).

Under v1.x this node DROVE ``/robot1`` with pure pursuit. It no longer drives
anything: the centralized controller commands both robots, so the leader's
scripted tracker is gone and what remains of this node is the reference the
task is defined against.

It publishes:

* ``/formation/reference_path``  nav_msgs/Path, latched -- the closed path, for
  RViz and for anything that wants to see the task geometry.
* ``/formation/lookahead``       geometry_msgs/PointStamped -- the point the
  controller is currently steering the leader at, republished for
  visualization as the leader estimate moves.

The controller does NOT consume either topic. The path is a static reference
derived from the episode config, not a measurement, so both the controller and
this node build it from the same config with
``formation_core.paths.make_path``. Sending it over the wire every step would
add a failure mode (a dropped or late path message steering the robots) in
exchange for nothing. The topics exist so a human, or a later external
consumer, can see what the controller is aiming at.

The leader's process noise moved to the controller node, which is now the only
thing publishing ``/robot1/cmd_vel``.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from formation_core.config import EpisodeConfig
from formation_core.paths import make_path

from .ros_interface import ANNOUNCE_QOS, WORLD_FRAME, pose_stamped, odometry_to_state


class ReferencePathNode(Node):
    """Publishes the reference path, and the leader's current lookahead point."""

    def __init__(self):
        super().__init__('formation_reference_path')
        self.declare_parameter('config', '')
        self.declare_parameter('leader_namespace', 'robot1')
        #: Vertices are ~1 cm apart; RViz does not need all of them.
        self.declare_parameter('path_stride', 20)

        config_path = self.get_parameter('config').value
        self.config = (
            EpisodeConfig.from_yaml(config_path) if config_path else EpisodeConfig())
        self.path = make_path(self.config.path.name, **self.config.path.params)
        self.leader = self.get_parameter('leader_namespace').value.strip('/')

        # Latched: RViz and late subscribers get the path without it being
        # republished on a timer.
        self.path_pub = self.create_publisher(Path, '/formation/reference_path',
                                              ANNOUNCE_QOS)
        self.path_pub.publish(self._path_message())

        self.lookahead_pub = self.create_publisher(
            PointStamped, '/formation/lookahead', 10)
        self.create_subscription(
            Odometry, f'/formation/{self.leader}/estimate', self._on_estimate, 10)

        self.get_logger().info(
            f'reference path ready: {self.path.name}, {self.path.length:.2f} m, '
            f'lookahead {self.config.contract.lookahead_distance} m')

    def _path_message(self):
        stride = max(int(self.get_parameter('path_stride').value), 1)
        msg = Path()
        msg.header.frame_id = WORLD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        points = self.path.points[::stride]
        msg.poses = [pose_stamped(point, msg.header) for point in points]
        # Close the loop so RViz draws the full circuit.
        msg.poses.append(pose_stamped(self.path.points[0], msg.header))
        return msg

    def _on_estimate(self, msg):
        """Show where the controller is steering the leader, from its estimate."""
        estimate = odometry_to_state(msg)
        target, _ = self.path.lookahead_pose(
            estimate.xy, self.config.contract.lookahead_distance)
        point = PointStamped()
        point.header.frame_id = WORLD_FRAME
        point.header.stamp = msg.header.stamp
        point.point.x = float(target[0])
        point.point.y = float(target[1])
        self.lookahead_pub.publish(point)


def main(args=None):
    rclpy.init(args=args)
    node = ReferencePathNode()
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
