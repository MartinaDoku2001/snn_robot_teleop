"""Headless smoke check for the multi-robot sim. Run while the sim is up:

    ros2 run rover_multi_bringup smoke_check --robots robot1 robot2

Checks, per robot: topics and nodes exist, odometry flows with prefixed frames,
the full prefixed tf tree is present, no tf frame has two parents, and driving
one robot moves only that robot. Exits non-zero on any failure.
"""

import argparse
import math
import sys
import time
from collections import defaultdict

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

ROBOT_TOPICS = ('cmd_vel', 'odom', 'joint_states', 'robot_description')
OPTIONAL_TOPICS = ('scan', 'imu/data')
ROBOT_NODES = ('robot_state_publisher', 'gz_bridge')
ROBOT_FRAMES = (
    'odom', 'base_link', 'chassis_link', 'payload_link', 'lidar_link', 'imu_link',
    'fl_wheel_link', 'fr_wheel_link', 'rl_wheel_link', 'rr_wheel_link',
)
SHARED_FRAMES = ('world',)


class SmokeCheck(Node):

    def __init__(self, robots):
        super().__init__('smoke_check')
        self.robots = robots
        self.failures = []
        self.odom = {}
        self.odom_frames = {}
        self.tf_parents = defaultdict(set)
        self.cmd_pubs = {r: self.create_publisher(Twist, f'/{r}/cmd_vel', 10) for r in robots}
        for r in robots:
            self.create_subscription(Odometry, f'/{r}/odom', self._odom_cb(r), 10)
        self.create_subscription(TFMessage, '/tf', self._tf_cb, 100)
        static_qos = QoSProfile(
            depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(TFMessage, '/tf_static', self._tf_cb, static_qos)

    def _odom_cb(self, robot):
        def cb(msg):
            p = msg.pose.pose.position
            self.odom[robot] = (p.x, p.y)
            self.odom_frames[robot] = (msg.header.frame_id, msg.child_frame_id)
        return cb

    def _tf_cb(self, msg):
        for t in msg.transforms:
            self.tf_parents[t.child_frame_id].add(t.header.frame_id)

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def check(self, ok, msg):
        print(f'  [{"PASS" if ok else "FAIL"}] {msg}')
        if not ok:
            self.failures.append(msg)

    def check_graph(self, timeout):
        print('Graph:')
        deadline = time.monotonic() + timeout
        wanted = {f'/{r}/{t}' for r in self.robots for t in ROBOT_TOPICS}
        while time.monotonic() < deadline:
            topics = {name for name, _ in self.get_topic_names_and_types()}
            if wanted <= topics:
                break
            self.spin_for(0.5)
        for name in sorted(wanted):
            self.check(name in topics, f'topic {name}')
        for r in self.robots:
            for t in OPTIONAL_TOPICS:
                name = f'/{r}/{t}'
                if name not in topics:
                    print(f'  [WARN] optional topic {name} missing')
        nodes = {f'{ns.rstrip("/")}/{n}' for n, ns in self.get_node_names_and_namespaces()}
        for r in self.robots:
            for n in ROBOT_NODES:
                self.check(f'/{r}/{n}' in nodes, f'node /{r}/{n}')

    def check_odom(self, timeout):
        print('Odometry:')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(self.odom) < len(self.robots):
            self.spin_for(0.2)
        for r in self.robots:
            got = self.odom_frames.get(r)
            self.check(got is not None, f'/{r}/odom is publishing')
            if got:
                self.check(got == (f'{r}/odom', f'{r}/base_link'),
                           f'/{r}/odom frames {got} == ({r}/odom, {r}/base_link)')

    def check_tf(self):
        print('TF:')
        self.spin_for(3.0)
        prefixes = tuple(f'{r}/' for r in self.robots)
        for frame, parents in sorted(self.tf_parents.items()):
            self.check(len(parents) == 1, f'{frame} has exactly one parent {sorted(parents)}')
        frames = set(self.tf_parents) | {p for ps in self.tf_parents.values() for p in ps}
        stray = sorted(f for f in frames if f not in SHARED_FRAMES and not f.startswith(prefixes))
        self.check(not stray, f'no unprefixed/colliding frames (stray: {stray})')
        for r in self.robots:
            missing = [f'{r}/{f}' for f in ROBOT_FRAMES if f'{r}/{f}' not in frames]
            self.check(not missing, f'{r} tf tree complete (missing: {missing})')

    def check_independent_drive(self, seconds):
        print('Independent drive:')
        for mover in self.robots:
            self.spin_for(0.5)
            start = dict(self.odom)
            cmd = Twist()
            cmd.linear.x = 0.5
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                self.cmd_pubs[mover].publish(cmd)
                self.spin_for(0.1)
            self.cmd_pubs[mover].publish(Twist())
            self.spin_for(1.0)
            for r in self.robots:
                if r not in start or r not in self.odom:
                    self.check(False, f'odom for {r} available during drive test')
                    continue
                dist = math.dist(start[r], self.odom[r])
                if r == mover:
                    self.check(dist > 0.1, f'cmd to {mover}: {r} moved {dist:.3f} m (> 0.1)')
                else:
                    self.check(dist < 0.02, f'cmd to {mover}: {r} moved {dist:.3f} m (< 0.02)')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--robots', nargs='+', default=['robot1', 'robot2'])
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--drive-seconds', type=float, default=3.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = SmokeCheck(args.robots)
    try:
        node.check_graph(args.timeout)
        node.check_odom(args.timeout)
        node.check_tf()
        node.check_independent_drive(args.drive_seconds)
    finally:
        failures = node.failures
        node.destroy_node()
        rclpy.shutdown()
    print(f'\nSMOKE CHECK {"FAILED" if failures else "PASSED"}'
          f' ({len(failures)} failure(s))')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
