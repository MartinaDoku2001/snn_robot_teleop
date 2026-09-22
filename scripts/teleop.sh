#!/usr/bin/env bash
# Keyboard teleop for one robot. Opens teleop_twist_keyboard publishing to
# /<robot>/cmd_vel. Run one terminal per robot.
#   ./scripts/teleop.sh robot1
set -e
robot="${1:?usage: teleop.sh <robot namespace, e.g. robot1>}"
if [ ! -f /.dockerenv ]; then
    cd "$(dirname "$0")/.."
    exec docker compose exec sim /ws/scripts/teleop.sh "$robot"
fi
source /opt/ros/humble/setup.bash
exec ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -r "cmd_vel:=/${robot}/cmd_vel" -r "__node:=teleop_${robot}"
