"""Spawn ONE namespaced Rover Mini into an already-running Gazebo world.

Per robot <ns> this starts, all inside namespace /<ns>:
  robot_state_publisher  frame_prefix '<ns>/', publishes /<ns>/robot_description
  ros_gz_sim create      Gazebo model named <ns>
  ros_gz_bridge          /<ns>/cmd_vel (ROS->gz), /<ns>/odom, /<ns>/joint_states,
                         /<ns>/scan, /<ns>/imu/data (gz->ROS); DiffDrive tf -> /tf
  static tf              world -> <ns>/odom at the spawn pose

Args: namespace, x, y, z, yaw, use_sim_time.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rover_multi_bringup.description import namespaced_mini_description


def launch_setup(context):
    ns = LaunchConfiguration('namespace').perform(context).strip('/')
    x, y, z, yaw = (LaunchConfiguration(k).perform(context) for k in ('x', 'y', 'z', 'yaw'))
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=ns,
        output='screen',
        parameters=[{
            'robot_description': namespaced_mini_description(ns),
            'frame_prefix': f'{ns}/',
            'use_sim_time': use_sim_time,
        }],
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=ns,
        output='screen',
        arguments=[
            '-name', ns,
            '-topic', f'/{ns}/robot_description',
            '-x', x, '-y', y, '-z', z, '-Y', yaw,
        ],
    )

    # gz-side topic names equal the ROS names (set by the description rewrite),
    # except DiffDrive's tf, which is remapped onto the global /tf.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        namespace=ns,
        name='gz_bridge',
        output='screen',
        arguments=[
            f'/{ns}/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            f'/{ns}/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            f'/{ns}/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            f'/{ns}/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
            f'/{ns}/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            f'/{ns}/imu/data@sensor_msgs/msg/Imu[ignition.msgs.IMU',
        ],
        remappings=[(f'/{ns}/tf', '/tf')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # DiffDrive odometry starts at zero where the robot spawns, so the shared
    # `world` frame sits at the spawn pose relative to each robot's odom.
    world_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace=ns,
        name='world_to_odom',
        arguments=[
            '--x', x, '--y', y, '--z', '0', '--yaw', yaw,
            '--frame-id', 'world', '--child-frame-id', f'{ns}/odom',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return [robot_state_publisher, spawn, bridge, world_to_odom]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='robot1'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.1'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
