"""Frame handling for the Gazebo adapter.

These are the conversions that decide whether both robots are described in the
SAME frame. Getting them wrong does not crash anything -- it silently shifts
every pose by the spawn offset -- so they are tested directly.

The tests need rclpy message types but never start a node or a simulator.
"""

import math

import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402

from formation_core.geometry import RobotState  # noqa: E402
from formation_gazebo.ros_interface import (  # noqa: E402
    announce,
    odometry_to_state,
    parse_announcement,
    quaternion_from_yaw,
    state_to_odometry,
    yaw_from_quaternion,
)


def _odom(x, y, yaw, v=0.0, w=0.0, frame='robot1/odom'):
    msg = Odometry()
    msg.header.frame_id = frame
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = v
    msg.twist.twist.angular.z = w
    return msg


def _transform(x, y, yaw):
    tf = TransformStamped()
    tf.transform.translation.x = x
    tf.transform.translation.y = y
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


@pytest.mark.parametrize('yaw', [0.0, 0.7, -1.2, math.pi - 0.01, -math.pi + 0.01])
def test_yaw_quaternion_round_trip(yaw):
    assert yaw_from_quaternion(
        state_to_odometry(RobotState(0, 0, yaw), None).pose.pose.orientation) == pytest.approx(yaw)


def test_without_a_transform_the_pose_is_passed_through():
    state = odometry_to_state(_odom(1.0, 2.0, 0.5, v=0.3, w=-0.2))
    assert (state.x, state.y) == (1.0, 2.0)
    assert state.theta == pytest.approx(0.5)
    assert state.v == pytest.approx(0.3) and state.w == pytest.approx(-0.2)


def test_translation_only_transform_shifts_the_pose():
    """Phase 0 spawns robots at an offset with no rotation: the common case."""
    state = odometry_to_state(_odom(1.0, 2.0, 0.5), _transform(0.0, 1.0, 0.0))
    assert (state.x, state.y) == pytest.approx((1.0, 3.0))
    assert state.theta == pytest.approx(0.5)


def test_rotating_transform_rotates_position_and_heading():
    state = odometry_to_state(_odom(1.0, 0.0, 0.0), _transform(0.0, 0.0, math.pi / 2))
    assert state.x == pytest.approx(0.0, abs=1e-9)
    assert state.y == pytest.approx(1.0)
    assert state.theta == pytest.approx(math.pi / 2)


def test_velocities_are_body_frame_so_the_transform_leaves_them_alone():
    state = odometry_to_state(
        _odom(1.0, 2.0, 0.5, v=0.4, w=0.1), _transform(3.0, -2.0, 1.1))
    assert state.v == pytest.approx(0.4)
    assert state.w == pytest.approx(0.1)


def test_two_robots_at_different_spawns_land_in_one_frame():
    """The reason this module exists: relative geometry must be correct."""
    # Both robots report 0 in their own odom frames, but they spawned 2 m apart.
    leader = odometry_to_state(_odom(0.0, 0.0, 0.0, frame='robot1/odom'),
                               _transform(0.0, 1.0, 0.0))
    follower = odometry_to_state(_odom(0.0, 0.0, 0.0, frame='robot2/odom'),
                                 _transform(0.0, -1.0, 0.0))
    assert math.hypot(leader.x - follower.x, leader.y - follower.y) == pytest.approx(2.0)


def test_state_to_odometry_round_trip():
    state = RobotState(1.5, -2.5, 0.75, v=0.4, w=-0.3)
    msg = state_to_odometry(state, None, frame_id='world', child_frame_id='leader_estimate')
    assert msg.header.frame_id == 'world'
    assert msg.child_frame_id == 'leader_estimate'
    back = odometry_to_state(msg)
    assert (back.x, back.y) == pytest.approx((state.x, state.y))
    assert back.theta == pytest.approx(state.theta)
    assert (back.v, back.w) == pytest.approx((state.v, state.w))


def test_gazebo_env_implements_the_shared_interface():
    """Contract check only -- it needs no running simulator to verify."""
    from formation_core.env import FormationEnv
    from formation_gazebo.env import GazeboFormationEnv
    assert issubclass(GazeboFormationEnv, FormationEnv)
    for method in ('reset', 'step', 'close'):
        assert callable(getattr(GazeboFormationEnv, method))


# --------------------------------------------------------------- announcements
# The evaluation node reports the policy and controller it is TOLD about, not
# the ones in its own YAML, because launch-argument overrides never reach it.
# These tests pin the payload both sides agree on.


class _RecordingNode:
    """Just enough Node to capture what :func:`announce` would publish."""

    def __init__(self):
        self.published = []

    def create_publisher(self, msg_type, topic, qos):
        node = self

        class _Pub:
            def publish(self, msg):
                node.published.append((topic, msg))

        return _Pub()


def _announced(component):
    node = _RecordingNode()
    announce(node, '/formation/announced', component)
    (_, msg), = node.published
    return parse_announcement(msg)


@pytest.mark.parametrize('name,params', [
    ('always', {}),
    ('periodic', {'k': 10}),
    ('random', {'p': 0.1}),
    ('event_triggered', {'delta': 0.05}),
    ('analytic', {}),
])
def test_an_announcement_round_trips(name, params):
    from formation_core.config import ComponentConfig
    component = ComponentConfig(name, params)
    assert _announced(component).describe() == component.describe()
    assert _announced(component).params == params


@pytest.mark.parametrize('name,params', [
    ('always', {}),
    ('periodic', {'k': 10}),
    ('random', {'p': 0.1}),
    ('event_triggered', {'delta': 0.05}),
])
def test_an_announced_policy_matches_what_the_fast_twin_writes(name, params):
    """Both backends record the BUILT policy, so the strings are comparable."""
    from formation_core.policies import make_policy
    policy = make_policy(name, **params)
    assert _announced(policy).describe() == policy.describe()


def test_a_defaulted_parameter_is_still_announced():
    """The reason the policy is announced built, not as requested."""
    from formation_core.config import ComponentConfig
    from formation_core.policies import make_policy
    requested = ComponentConfig('random', {})       # no p given anywhere
    assert requested.describe() == 'random'         # the old, lossy string
    assert _announced(make_policy('random')).describe() == 'random(p=0.5)'


def test_an_announced_component_can_replace_the_one_in_a_config():
    """What the evaluation node does before writing metrics.csv/config.yaml."""
    from formation_core.config import EpisodeConfig
    from formation_core.policies import make_policy
    config = EpisodeConfig()                       # policy: always
    assert config.policy.describe() == 'always'
    effective = config.with_overrides(
        policy=_announced(make_policy('periodic', k=10)))
    assert effective.policy.describe() == 'periodic(k=10)'
    assert EpisodeConfig.from_dict(effective.to_dict()).policy.describe() == 'periodic(k=10)'
