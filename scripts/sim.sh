#!/usr/bin/env bash
# Launch the two-robot simulation (Gazebo GUI + RViz) inside the container.
# Extra launch args pass through, e.g.: ./scripts/sim.sh rviz:=false
set -e
cd "$(dirname "$0")/.."
exec docker compose exec sim ros-env ros2 launch rover_multi_bringup multi_mini.launch.py "$@"
