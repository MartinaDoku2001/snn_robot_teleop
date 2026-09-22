# syntax=docker/dockerfile:1
#
# Phase 0 simulation image: ROS 2 Humble + Ignition Gazebo Fortress (via ros_gz).
#
# The base is pinned by digest. osrf/ros:humble-desktop-full already ships
# Gazebo Fortress (libignition-gazebo6) and ros_gz 0.244.x, which is exactly
# what roverrobotics_gazebo depends on (ros_gz_sim / ros_gz_bridge). Gazebo
# Classic is NOT installed and must not be added.
ARG BASE_IMAGE=osrf/ros:humble-desktop-full@sha256:1db1e4e941d4f77fab55bcd479158a75273329335e86ec468bff280859e8178c
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    WS=/ws

# Extra tooling not in desktop-full. No `apt-get upgrade`: everything already
# in the pinned base image stays at the pinned versions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-teleop-twist-keyboard \
        mesa-utils \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Resolve workspace dependencies with rosdep from package.xml files only, so
# this layer is cached until a package manifest changes. Sources themselves
# are bind-mounted at runtime (see docker-compose.yml).
COPY src /tmp/deps/src
RUN apt-get update \
    && (rosdep init 2>/dev/null || true) \
    && rosdep update --rosdistro "${ROS_DISTRO}" \
    && rosdep install --from-paths /tmp/deps/src --ignore-src -r -y --rosdistro "${ROS_DISTRO}" \
    && rm -rf /var/lib/apt/lists/* /tmp/deps

# Record exactly what ended up in the image (for cross-machine comparisons).
RUN dpkg-query -W -f='${Package}=${Version}\n' | sort > /opt/image-packages.lock

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/ros-env /usr/local/bin/ros-env
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/ros-env \
    && echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

WORKDIR /ws
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
