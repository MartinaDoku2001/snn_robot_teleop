"""Bridge between Phase 0 ROS topics and formation_core's state record.

Phase 0 publishes each robot's odometry in its OWN frame (``/robot1/odom`` is
expressed in ``robot1/odom``, which sits at that robot's spawn pose). The
formation task is about the geometry BETWEEN the robots, so everything is
converted into the shared ``world`` frame first, using the static transforms
Phase 0 already publishes.

Verified Phase 0 interfaces (re-checked against the running sim):

* ``/<ns>/odom``    nav_msgs/Odometry, frame ``<ns>/odom``, child ``<ns>/base_link``
* ``/<ns>/cmd_vel`` geometry_msgs/Twist
* tf ``world -> <ns>/odom`` static, at the spawn pose
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener

from formation_core.geometry import RobotState, wrap_angle

WORLD_FRAME = 'world'


def yaw_from_quaternion(q):
    """Yaw (rad) from a geometry_msgs Quaternion."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion_from_yaw(yaw):
    """(x, y, z, w) for a planar rotation."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def odometry_to_state(msg, transform=None):
    """Convert Odometry to a :class:`RobotState`, optionally into ``world``.

    ``transform`` is a geometry_msgs TransformStamped for
    ``world <- msg.header.frame_id``. Linear/angular velocities are body-frame
    in both frames, so only the pose is transformed.
    """
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = yaw_from_quaternion(msg.pose.pose.orientation)

    if transform is not None:
        t = transform.transform.translation
        offset_yaw = yaw_from_quaternion(transform.transform.rotation)
        cos_a, sin_a = math.cos(offset_yaw), math.sin(offset_yaw)
        x, y = t.x + cos_a * x - sin_a * y, t.y + sin_a * x + cos_a * y
        theta = float(wrap_angle(theta + offset_yaw))

    return RobotState(
        x=float(x), y=float(y), theta=float(theta),
        v=float(msg.twist.twist.linear.x), w=float(msg.twist.twist.angular.z))


def state_to_odometry(state, stamp=None, frame_id=WORLD_FRAME, child_frame_id=''):
    """Pack a :class:`RobotState` into an Odometry message (world frame).

    ``stamp`` may be None (the message keeps its zero stamp), which is handy
    for tests and offline conversion.
    """
    msg = Odometry()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(state.x)
    msg.pose.pose.position.y = float(state.y)
    qx, qy, qz, qw = quaternion_from_yaw(state.theta)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = float(state.v)
    msg.twist.twist.angular.z = float(state.w)
    return msg


class RobotBridge:
    """Odometry in, commands out, for one namespaced robot.

    Caches the static ``world <- <ns>/odom`` transform after the first
    successful lookup, so the control loop never blocks on tf.
    """

    def __init__(self, node, namespace, world_frame=WORLD_FRAME,
                 tf_buffer=None, publish_commands=True, require_transform=True):
        self.node = node
        self.namespace = namespace.strip('/')
        self.world_frame = world_frame
        #: Withhold state until the odom->world transform is known. Phase 0
        #: always publishes it, and acting on a pose that is still in the
        #: robot's own odom frame means acting on a pose that is wrong by the
        #: spawn offset -- which sends the robot off in the wrong direction
        #: for as long as the lookup takes.
        self.require_transform = bool(require_transform)
        self.state = None
        self.last_stamp = None
        self._transform = None
        self._warned = False

        self.tf_buffer = tf_buffer if tf_buffer is not None else Buffer()
        if tf_buffer is None:
            self._listener = TransformListener(self.tf_buffer, node)

        self.odom_sub = node.create_subscription(
            Odometry, f'/{self.namespace}/odom', self._on_odom, 10)
        self.cmd_pub = node.create_publisher(
            Twist, f'/{self.namespace}/cmd_vel', 10) if publish_commands else None

    def _lookup(self, frame_id):
        if self._transform is not None:
            return self._transform
        try:
            self._transform = self.tf_buffer.lookup_transform(
                self.world_frame, frame_id, rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except Exception as exc:  # tf2 raises several unrelated types
            if not self._warned:
                fallback = ('waiting for it' if self.require_transform
                            else 'using the odom frame directly meanwhile')
                self.node.get_logger().warn(
                    f'no transform {self.world_frame} <- {frame_id} yet ({exc}); '
                    f'{fallback}')
                self._warned = True
            return None
        self.node.get_logger().info(
            f'resolved static transform {self.world_frame} <- {frame_id}')
        return self._transform

    def _on_odom(self, msg):
        transform = self._lookup(msg.header.frame_id)
        if transform is None and self.require_transform:
            return  # stay "not ready" rather than report a wrong-frame pose
        self.state = odometry_to_state(msg, transform)
        self.last_stamp = msg.header.stamp

    @property
    def ready(self):
        return self.state is not None

    def publish_command(self, v, w):
        if self.cmd_pub is None:
            raise RuntimeError(f'{self.namespace} bridge was created read-only')
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def stop(self):
        """Send a zero command, tolerating an already-shut-down context.

        This runs on the way out, when rclpy may have torn the context down
        already; a failure here must not mask the real exit reason.
        """
        if self.cmd_pub is None or not rclpy.ok():
            return
        try:
            self.cmd_pub.publish(Twist())
        except Exception:  # noqa: BLE001 - shutdown races are not actionable
            pass


def wait_for_bridges(node, bridges, timeout=30.0, poll=0.05):
    """Spin until every bridge has odometry (or the timeout expires).

    Uses WALL time deliberately: with ``use_sim_time`` the ROS clock does not
    advance until /clock arrives, which is one of the things being waited for.
    """
    deadline = time.monotonic() + timeout
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=poll)
        if all(b.ready for b in bridges):
            return True
        if time.monotonic() > deadline:
            return False
    return False


def array_to_float_list(array):
    return [float(v) for v in np.asarray(array).reshape(-1)]
