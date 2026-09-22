"""Start Gazebo (Fortress) with a world and bridge the sim clock to ROS.

Args:
  world:     SDF file name in rover_multi_bringup/worlds, or an absolute path.
  gui:       true -> server + GUI client; false -> headless server only.
  sensors:   true/false/auto. Rendering sensors (gpu_lidar) need a working
             GL/EGL device; `auto` enables them with the GUI and disables them
             headless, which keeps the headless smoke test hardware-independent.
  verbosity: gz log verbosity (0-4).
"""

import os
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def resolve_world(world):
    if os.path.isabs(world):
        return world
    return os.path.join(get_package_share_directory('rover_multi_bringup'), 'worlds', world)


def without_render_sensors(world):
    """Copy of ``world`` with the Sensors system removed (no rendering at all)."""
    tree = ET.parse(world)
    for world_el in tree.getroot().iter('world'):
        for plugin in list(world_el.findall('plugin')):
            if 'sensors-system' in plugin.get('filename', ''):
                world_el.remove(plugin)
    fd, path = tempfile.mkstemp(prefix='world_nosensors_', suffix='.sdf')
    with os.fdopen(fd, 'wb') as f:
        tree.write(f)
    return path


def launch_setup(context):
    world = resolve_world(LaunchConfiguration('world').perform(context))
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    sensors = LaunchConfiguration('sensors').perform(context).lower()
    if sensors == 'false' or (sensors == 'auto' and not gui):
        world = without_render_sensors(world)
    verbosity = LaunchConfiguration('verbosity').perform(context)

    # -r: start running. Headless: server only; if sensors are forced on,
    # render offscreen through EGL.
    gz_args = f'-r -v {verbosity} {world}'
    if not gui:
        gz_args = f'-s {gz_args}'
        if sensors == 'true':
            gz_args = f'--headless-rendering {gz_args}'

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        # Tear the whole launch down if Gazebo exits (crash or GUI closed).
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )
    return [gz_sim, clock_bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='arena.sdf'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('sensors', default_value='auto'),
        DeclareLaunchArgument('verbosity', default_value='2'),
        OpaqueFunction(function=launch_setup),
    ])
