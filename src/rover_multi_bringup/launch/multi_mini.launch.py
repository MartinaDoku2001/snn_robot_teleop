"""Gazebo + N namespaced Minis (default: robot1 and robot2) + optional RViz.

  ros2 launch rover_multi_bringup multi_mini.launch.py            # GUI + RViz
  ros2 launch rover_multi_bringup multi_mini.launch.py gui:=false rviz:=false

Robots and spawn poses come from config/robots.yaml (override: robots_file:=...).
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SHARE = get_package_share_directory('rover_multi_bringup')


def include(name, args):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(SHARE, 'launch', name)),
        launch_arguments={
            k: v if isinstance(v, LaunchConfiguration) else str(v) for k, v in args.items()
        }.items())


def spawn_robots(context):
    with open(LaunchConfiguration('robots_file').perform(context)) as f:
        robots = yaml.safe_load(f)['robots']
    names = [r['name'] for r in robots]
    if len(set(names)) != len(names):
        raise RuntimeError(f'robot names must be unique, got {names}')
    return [
        include('spawn_mini.launch.py', {
            'namespace': r['name'],
            'x': r.get('x', 0.0), 'y': r.get('y', 0.0), 'yaw': r.get('yaw', 0.0),
        })
        for r in robots
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUI client'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('world', default_value='arena.sdf'),
        DeclareLaunchArgument(
            'sensors', default_value='auto', description='render sensors: true|false|auto'),
        DeclareLaunchArgument(
            'robots_file', default_value=os.path.join(SHARE, 'config', 'robots.yaml')),
        include('gazebo.launch.py', {
            'world': LaunchConfiguration('world'),
            'gui': LaunchConfiguration('gui'),
            'sensors': LaunchConfiguration('sensors'),
        }),
        OpaqueFunction(function=spawn_robots),
        Node(
            package='rviz2', executable='rviz2', output='screen',
            arguments=['-d', os.path.join(SHARE, 'rviz', 'multi_mini.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
