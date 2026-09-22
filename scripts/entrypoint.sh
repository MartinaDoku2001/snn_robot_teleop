#!/usr/bin/env bash
# Container entrypoint: build the workspace on first start (build/install/log
# live in docker volumes, so this only happens once), then run the command.
set -e
source /opt/ros/humble/setup.bash

if [ ! -f /ws/install/setup.bash ] || [ "${FORCE_BUILD:-0}" = "1" ]; then
    echo "[entrypoint] Building workspace (first start or FORCE_BUILD=1)..."
    /ws/scripts/build.sh
fi

exec ros-env "$@"
