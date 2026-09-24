"""Run the formation task in Gazebo (contract v2.0).

Brings up the Phase 0 two-robot simulation (spawned on the reference path, in
an empty arena) and the formation nodes on top of it:

    leader_node          publishes the reference path (it no longer drives)
    comm_interface_node  ONE PER ROBOT: what the coordinator knows about it
    controller_node      ONE controller: 16-dim observation -> 4-dim action ->
                         /robot1/cmd_vel AND /robot2/cmd_vel
    evaluation_node      records the same metrics as the fast twin, writes CSV

  ros2 launch formation_gazebo formation.launch.py
  ros2 launch formation_gazebo formation.launch.py policy:=event_triggered delta:=0.05
  ros2 launch formation_gazebo formation.launch.py controller:=rl \
      weights:=/ws/results/rl/actor.pt
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

    leader_ns = LaunchConfiguration('leader_namespace').perform(context)
    follower_ns = LaunchConfiguration('follower_namespace').perform(context)

    common = {'use_sim_time': True, 'config': config, 'seed': seed}
    comm_params = {
        'policy': policy,
        'k': int(LaunchConfiguration('k').perform(context)),
        'p': float(LaunchConfiguration('p').perform(context)),
        'delta': float(LaunchConfiguration('delta').perform(context)),
    }

    # One communication interface per robot. Each owns its own policy,
    # predictor and estimate, and draws from its own RNG stream (picked by
    # ``role``), so the two never interfere.
    interfaces = [
        Node(package='formation_gazebo', executable='comm_interface_node',
             name=f'formation_comm_{namespace}', output='screen',
             parameters=[dict(common, **comm_params, robot=namespace, role=role)])
        for namespace, role in ((leader_ns, 'leader'), (follower_ns, 'follower'))
    ]

    return interfaces + [
        Node(package='formation_gazebo', executable='leader_node',
             name='formation_reference_path', output='screen',
             parameters=[dict(common, leader_namespace=leader_ns)]),
        Node(package='formation_gazebo', executable='controller_node',
             name='formation_controller', output='screen',
             parameters=[dict(common,
                              leader_namespace=leader_ns,
                              follower_namespace=follower_ns,
                              controller=LaunchConfiguration('controller').perform(context),
                              weights=LaunchConfiguration('weights').perform(context))]),
        Node(package='formation_gazebo', executable='evaluation_node',
             name='formation_evaluation', output='screen',
             parameters=[dict(common,
                              leader_namespace=leader_ns,
                              follower_namespace=follower_ns,
                              out=LaunchConfiguration('out').perform(context),
                              duration=duration,
                              label=policy or 'config')]),
    ]


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
        DeclareLaunchArgument('leader_namespace', default_value='robot1'),
        DeclareLaunchArgument('follower_namespace', default_value='robot2'),
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
        DeclareLaunchArgument('weights', default_value='',
                              description='learned controller: trained actor path'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('duration', default_value='-1.0',
                              description='override the episode duration (s)'),
        DeclareLaunchArgument('out', default_value='results/gazebo'),
        sim,
        OpaqueFunction(function=launch_setup),
    ])
