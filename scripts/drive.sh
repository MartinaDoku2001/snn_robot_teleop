#!/usr/bin/env bash
# Drive a robot without the keyboard -- useful to check the sim is alive and
# for scripted tests.
#   ./scripts/drive.sh robot1            # forward 0.5 m/s for 3 s
#   ./scripts/drive.sh robot2 0.4 0.5 5  # linear.x angular.z seconds
set -e
robot="${1:?usage: drive.sh <robot> [linear.x] [angular.z] [seconds]}"
lin="${2:-0.5}"; ang="${3:-0.0}"; secs="${4:-3}"
if [ ! -f /.dockerenv ]; then
    cd "$(dirname "$0")/.."
    exec docker compose exec -T sim /ws/scripts/drive.sh "$robot" "$lin" "$ang" "$secs"
fi
source /opt/ros/humble/setup.bash
echo "[drive] /${robot}/cmd_vel  linear.x=$lin angular.z=$ang for ${secs}s"
timeout "$secs" ros2 topic pub -r 10 "/${robot}/cmd_vel" geometry_msgs/msg/Twist \
    "{linear: {x: $lin}, angular: {z: $ang}}" >/dev/null 2>&1 || true
ros2 topic pub -t 3 "/${robot}/cmd_vel" geometry_msgs/msg/Twist '{}' >/dev/null 2>&1
echo "[drive] stopped. odom:"
timeout 8 ros2 topic echo --once "/${robot}/odom" --field pose.pose.position | head -3
