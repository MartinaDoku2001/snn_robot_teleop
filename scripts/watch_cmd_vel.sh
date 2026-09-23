#!/usr/bin/env bash
# Show every command arriving on /<robot>/cmd_vel. Run this in one terminal
# while you press keys in the teleop terminal: if nothing prints, the
# keystrokes are not reaching teleop (wrong window focused, or a key that is
# not bound -- arrow keys are NOT bound, use i / j / k / l / ,).
set -e
robot="${1:?usage: watch_cmd_vel.sh <robot>}"
if [ ! -f /.dockerenv ]; then
    cd "$(dirname "$0")/.."
    exec docker compose exec -T sim /ws/scripts/watch_cmd_vel.sh "$robot"
fi
source /opt/ros/humble/setup.bash
exec ros2 topic echo "/${robot}/cmd_vel" --field linear.x
