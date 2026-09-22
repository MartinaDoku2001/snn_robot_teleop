#!/usr/bin/env bash
# Start the sim container with the right GPU mode.
#   ./scripts/up.sh            # auto-detect: nvidia > dri > software
#   ./scripts/up.sh dri|nvidia|software
# Extra args are passed to `docker compose up` (e.g. --build).
set -e
cd "$(dirname "$0")/.."
mode="${1:-auto}"; [ $# -gt 0 ] && shift

if [ "$mode" = auto ]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
        && docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
        mode=nvidia
    elif [ -e /dev/dri ]; then
        mode=dri
    else
        mode=software
    fi
fi

files=(-f docker-compose.yml)
case "$mode" in
    dri) ;;
    nvidia) files+=(-f docker-compose.nvidia.yml) ;;
    software) files+=(-f docker-compose.software.yml) ;;
    *) echo "unknown mode '$mode' (use auto|dri|nvidia|software)" >&2; exit 1 ;;
esac

echo "[up] GPU mode: $mode"
./scripts/xhost_setup.sh >/dev/null || echo "[up] warning: xhost step failed; GUIs may not open" >&2
docker compose "${files[@]}" up -d "$@"
echo "[up] Container is up. First start builds the workspace; follow with: docker compose logs -f sim"
