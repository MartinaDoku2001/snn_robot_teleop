#!/usr/bin/env bash
# Headless end-to-end check: launch the two-robot sim with no GUI, run
# `smoke_check` against it, then shut everything down. Exit code 0 == pass.
#   ./scripts/smoke_test.sh               (host: runs inside the container)
#   LAUNCH=single_mini.launch.py ROBOTS=robot1 ./scripts/smoke_test.sh
set -e
if [ ! -f /.dockerenv ]; then
    cd "$(dirname "$0")/.."
    exec docker compose exec -T -e LAUNCH -e ROBOTS sim /ws/scripts/smoke_test.sh "$@"
fi
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash

LAUNCH="${LAUNCH:-multi_mini.launch.py}"
ROBOTS="${ROBOTS:-robot1 robot2}"
LOG=/ws/log/smoke_launch.log

extra=(gui:=false)
[ "$LAUNCH" = multi_mini.launch.py ] && extra+=(rviz:=false)

echo "[smoke] launching $LAUNCH ${extra[*]} (log: $LOG)"
setsid ros2 launch rover_multi_bringup "$LAUNCH" "${extra[@]}" >"$LOG" 2>&1 &
launch_pid=$!
cleanup() {
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$launch_pid" 2>/dev/null || true
    pkill -KILL -f 'ign gazebo' 2>/dev/null || true
}
trap cleanup EXIT

# shellcheck disable=SC2086
timeout 180 ros2 run rover_multi_bringup smoke_check --robots $ROBOTS "$@"
