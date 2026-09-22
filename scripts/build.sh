#!/usr/bin/env bash
# Build the colcon workspace. Works from the host (runs inside the container
# via docker compose) or directly inside the container.
#   ./scripts/build.sh                           # build everything
#   ./scripts/build.sh --packages-select rover_multi_bringup
set -e
if [ ! -f /.dockerenv ]; then
    cd "$(dirname "$0")/.."
    exec docker compose exec sim /ws/scripts/build.sh "$@"
fi
source /opt/ros/humble/setup.bash
cd /ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release "$@"
