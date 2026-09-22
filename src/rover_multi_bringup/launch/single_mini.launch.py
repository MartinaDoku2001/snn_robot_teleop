"""Gazebo + ONE namespaced Mini (incremental step before multi_mini)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('rover_multi_bringup')

    def include(name, args):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', name)),
            launch_arguments=args.items())

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='robot1'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('world', default_value='arena.sdf'),
        DeclareLaunchArgument(
            'sensors', default_value='auto', description='render sensors: true|false|auto'),
        include('gazebo.launch.py', {
            'world': LaunchConfiguration('world'),
            'gui': LaunchConfiguration('gui'),
            'sensors': LaunchConfiguration('sensors'),
        }),
        include('spawn_mini.launch.py', {'namespace': LaunchConfiguration('namespace')}),
        Node(
            package='rviz2', executable='rviz2', output='screen',
            arguments=['-d', os.path.join(share, 'rviz', 'multi_mini.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
