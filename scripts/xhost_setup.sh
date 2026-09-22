#!/usr/bin/env bash
# Allow local containers (running as root) to open windows on your X server.
# Needed once per login session. Undo with: xhost -local:root
set -e
if [ -z "${DISPLAY:-}" ]; then
    echo "DISPLAY is not set. Run this from a terminal inside your graphical session." >&2
    exit 1
fi
if ! command -v xhost >/dev/null; then
    echo "xhost not found. Install it: Ubuntu/Debian: x11-xserver-utils | Arch: xorg-xhost | Fedora: xhost" >&2
    exit 1
fi
xhost +local:root
