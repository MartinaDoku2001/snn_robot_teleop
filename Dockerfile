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
        python3-pip \
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

# Phase 2d: PPO needs PyTorch and Gymnasium, and neither has a rosdep key on
# Humble, so they are installed here rather than through package.xml.
#
# CPU-ONLY, from PyTorch's cpu index. The actor is a 5.5k-parameter MLP and the
# environment is pure Python, so training is bound by the environment rather
# than by matrix multiplies; the CUDA wheels would add several GB to the image
# for no gain. Pinned, because an unpinned torch is the fastest way to make a
# trained policy unreproducible.
#
# This layer roughly doubles the image size. Set TORCH=0 to build without it:
# everything except formation_rl works, and formation_core stays torch-free.
ARG TORCH=1
RUN if [ "${TORCH}" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            "torch==2.2.2" \
        && pip install --no-cache-dir "gymnasium==0.29.1" ; \
    fi

# Record exactly what ended up in the image (for cross-machine comparisons).
RUN dpkg-query -W -f='${Package}=${Version}\n' | sort > /opt/image-packages.lock \
    && (pip freeze 2>/dev/null | sort > /opt/image-pip.lock || true)

# The nvidia-container-toolkit injects libEGL_nvidia but not always its glvnd
# vendor file. Without it, headless (EGL) rendering cannot see the NVIDIA GPU
# and falls back to a DRM node Mesa cannot drive. Harmless on non-NVIDIA
# hosts: glvnd skips a vendor whose library is missing.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/ros-env /usr/local/bin/ros-env
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/ros-env \
    && echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

WORKDIR /ws
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
