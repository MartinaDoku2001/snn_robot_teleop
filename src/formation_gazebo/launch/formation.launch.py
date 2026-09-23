"""Run the formation task in Gazebo.

Brings up the Phase 0 two-robot simulation (spawned on the reference path, in
an empty arena) and the four formation nodes on top of it:

    leader_node          drives /robot1 along the path (pure pursuit + noise)
    comm_interface_node  decides what /robot2 is allowed to know about /robot1
    controller_node      contract observation -> action -> /robot2/cmd_vel
    evaluation_node      records the same metrics as the fast twin, writes CSV

  ros2 launch formation_gazebo formation.launch.py
  ros2 launch formation_gazebo formation.launch.py policy:=event_triggered delta:=0.05
  ros2 launch formation_gazebo formation.launch.py gui:=false duration:=30.0

Set ``sim:=false`` to attach to a simulation that is already running.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SHARE = get_package_share_directory('formation_gazebo')


def launch_setup(context):
    config = LaunchConfiguration('config').perform(context)
    policy = LaunchConfiguration('policy').perform(context)
    duration = float(LaunchConfiguration('duration').perform(context))
    seed = int(LaunchConfiguration('seed').perform(context))

    common = {'use_sim_time': True, 'config': config}
    comm_params = dict(common)
    comm_params.update({
        'policy': policy,
        'k': int(LaunchConfiguration('k').perform(context)),
        'p': float(LaunchConfiguration('p').perform(context)),
        'delta': float(LaunchConfiguration('delta').perform(context)),
    })

    nodes = [
        Node(package='formation_gazebo', executable='leader_node',
             name='formation_leader', output='screen',
             parameters=[dict(common, seed=seed)]),
        Node(package='formation_gazebo', executable='comm_interface_node',
             name='formation_comm_interface', output='screen',
             parameters=[comm_params]),
        Node(package='formation_gazebo', executable='controller_node',
             name='formation_controller', output='screen',
             parameters=[dict(common,
                              controller=LaunchConfiguration('controller').perform(context))]),
        Node(package='formation_gazebo', executable='evaluation_node',
             name='formation_evaluation', output='screen',
             parameters=[dict(common,
                              out=LaunchConfiguration('out').perform(context),
                              duration=duration,
                              label=policy or 'config')]),
    ]
    return nodes


def generate_launch_description():
    default_config = os.path.join(SHARE, 'config', 'gazebo_episode.yaml')
    robots_file = os.path.join(SHARE, 'config', 'formation_robots.yaml')
    world = os.path.join(SHARE, 'worlds', 'formation_arena.sdf')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rover_multi_bringup'),
            'launch', 'multi_mini.launch.py')),
        launch_arguments={
            'gui': LaunchConfiguration('gui'),
            'rviz': LaunchConfiguration('rviz'),
            'world': world,
            'robots_file': robots_file,
        }.items(),
        condition=IfCondition(LaunchConfiguration('sim')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument('sim', default_value='true',
                              description='also start the Phase 0 simulation'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('policy', default_value='',
                              description='override the transmission policy'),
        DeclareLaunchArgument('k', default_value='0'),
        DeclareLaunchArgument('p', default_value='-1.0'),
        DeclareLaunchArgument('delta', default_value='-1.0'),
        DeclareLaunchArgument('controller', default_value=''),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('duration', default_value='-1.0',
                              description='override the episode duration (s)'),
        DeclareLaunchArgument('out', default_value='results/gazebo'),
        sim,
        OpaqueFunction(function=launch_setup),
    ])
