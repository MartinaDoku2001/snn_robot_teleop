#!/usr/bin/env bash
# Phase 1 Gazebo smoke test: run one formation episode end to end and assert
# the nodes, topics and CSV appear.
#   ./scripts/formation_smoke.sh                       # event_triggered, 20 s
#   DURATION=30 POLICY=periodic K=5 ./scripts/formation_smoke.sh
set -e
if [ ! -f /.dockerenv ]; then
    cd "$(dirname "$0")/.."
    exec docker compose exec -T -e DURATION -e POLICY -e K -e DELTA -e OUT \
        sim /ws/scripts/formation_smoke.sh "$@"
fi
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash

DURATION="${DURATION:-20}"
POLICY="${POLICY:-event_triggered}"
DELTA="${DELTA:-0.05}"
K="${K:-0}"
OUT="${OUT:-/ws/results/gazebo_smoke}"
LOG=/ws/log/formation_smoke.log

cleanup() {
    pkill -f 'ros2 launch' 2>/dev/null || true
    sleep 2
    for pattern in 'ign gazebo' robot_state_publisher static_transform_publisher \
                   parameter_bridge 'formation_gazebo/lib' ; do
        pkill -f "$pattern" 2>/dev/null || true
    done
    sleep 2
}

# A stale simulation from a previous run would publish a SECOND, conflicting
# world->robotN/odom transform and quietly corrupt every pose. Always start clean.
echo '[smoke] clearing any previous simulation'
cleanup
ros2 daemon stop >/dev/null 2>&1 || true
trap cleanup EXIT

rm -rf "$OUT"
echo "[smoke] launching: policy=$POLICY delta=$DELTA k=$K duration=${DURATION}s"
setsid ros2 launch formation_gazebo formation.launch.py \
    gui:=false rviz:=false duration:="$DURATION" \
    policy:="$POLICY" delta:="$DELTA" k:="$K" out:="$OUT" >"$LOG" 2>&1 &

# Gazebo runs below real time, so allow generous wall-clock headroom.
deadline=$(( $(date +%s) + $(printf '%.0f' "$DURATION") * 12 + 120 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    grep -q 'EPISODE COMPLETE' "$LOG" && break
    sleep 5
done

status=0
check() {
    if [ "$2" -gt 0 ] 2>/dev/null; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; status=1; fi
}
echo '[smoke] checks:'
check 'leader, comm, controller and evaluation nodes started' \
      "$(grep -c 'ready\|evaluating' "$LOG")"
check 'estimate/observation/action topics published' \
      "$(grep -c 'formation_comm_interface\|formation_controller' "$LOG")"
check 'unique world->odom transforms (no stale simulation)' \
      "$(grep -c 'resolved static transform' "$LOG")"
check 'episode completed' "$(grep -c 'EPISODE COMPLETE' "$LOG")"
check 'episode.csv written' "$(ls "$OUT"/episode.csv 2>/dev/null | wc -l)"
check 'metrics.csv written' "$(ls "$OUT"/metrics.csv 2>/dev/null | wc -l)"

grep 'EPISODE COMPLETE' "$LOG" | sed 's/.*EPISODE COMPLETE/  /' || true
[ $status -eq 0 ] && echo '[smoke] PASSED' || echo "[smoke] FAILED (see $LOG)"
exit $status
