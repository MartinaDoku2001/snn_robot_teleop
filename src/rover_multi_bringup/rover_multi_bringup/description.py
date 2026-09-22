"""Build a per-robot, namespaced URDF from the unmodified vendor Mini description.

The vendor ``mini.urdf`` configures Gazebo (Fortress) system plugins with
absolute, hard-coded gz-transport topics (``/cmd_vel``, ``/odometry/wheels``,
``/joint_states``, ``scan``, ...) and unprefixed frame ids (``odom``,
``base_link``, ``lidar_link``). Two copies of it in one world would share every
topic and frame. Because the Mini drives through the Gazebo DiffDrive system
(not ros2_control), there is no controller manager to remap; instead we rewrite
the generated URDF before spawning:

* every ``*topic`` tag inside a ``<plugin>`` or ``<sensor>`` is moved under
  ``/<ns>/`` (DiffDrive odometry becomes ``/<ns>/odom``, its tf ``/<ns>/tf``);
* every frame-id tag is prefixed with ``<ns>/`` to match the
  robot_state_publisher ``frame_prefix``;
* world-level systems the vendor attaches to the model (Sensors, Imu) are
  removed -- they must exist once per world, so our world file loads them.
"""

import os
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory

FRAME_TAGS = ('frame_id', 'child_frame_id', 'ignition_frame_id', 'gz_frame_id')
# Tags renamed (not just prefixed) so the ROS-facing names are conventional.
RENAMED_TOPICS = {'odom_topic': 'odom', 'tf_topic': 'tf'}
# Systems that must be loaded once per world, not once per model.
WORLD_LEVEL_SYSTEM_MARKERS = ('sensors-system', 'imu-system')


def vendor_mini_urdf_path():
    return os.path.join(
        get_package_share_directory('roverrobotics_description'), 'urdf', 'mini.urdf')


def _is_world_level_system(plugin):
    filename = plugin.get('filename', '')
    return any(marker in filename for marker in WORLD_LEVEL_SYSTEM_MARKERS)


def _namespace_topic(ns, topic):
    return f'/{ns}/{topic.strip().lstrip("/")}'


def _namespace_block(ns, element):
    for child in element.iter():
        tag = child.tag
        if child.text is None:
            continue
        if tag in RENAMED_TOPICS:
            child.text = _namespace_topic(ns, RENAMED_TOPICS[tag])
        elif tag == 'topic' or tag.endswith('_topic'):
            child.text = _namespace_topic(ns, child.text)
        elif tag in FRAME_TAGS:
            child.text = f'{ns}/{child.text.strip()}'


def namespace_urdf(urdf_xml, ns):
    """Return ``urdf_xml`` with all Gazebo topics/frames scoped to ``ns``."""
    ns = ns.strip('/')
    root = ET.fromstring(urdf_xml)
    for gazebo in root.iter('gazebo'):
        for plugin in list(gazebo.findall('plugin')):
            if _is_world_level_system(plugin):
                gazebo.remove(plugin)
            else:
                _namespace_block(ns, plugin)
        for sensor in gazebo.findall('sensor'):
            _namespace_block(ns, sensor)
    return ET.tostring(root, encoding='unicode')


def namespaced_mini_description(ns):
    """Process the vendor Mini xacro and namespace it for robot ``ns``."""
    doc = xacro.process_file(vendor_mini_urdf_path())
    return namespace_urdf(doc.toxml(), ns)
