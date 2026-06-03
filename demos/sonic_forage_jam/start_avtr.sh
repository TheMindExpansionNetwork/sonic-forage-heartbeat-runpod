#!/usr/bin/env bash
# AVTR-1 interactive WebRTC demo on port 7860 (renderer on 8000).
set -euo pipefail

AVTR_ROOT="${AVTR_ROOT:-/workspace/avtr-1}"
export PATH="${HOME}/.pixi/bin:${PATH}"
export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-${AVTR_ROOT}/artifacts}"
export STREAMER_HOST="${STREAMER_HOST:-0.0.0.0}"
export STREAMER_PORT="${STREAMER_PORT:-7860}"
export RENDERER_PORT="${RENDERER_PORT:-8000}"
export LOAD_BALANCER_URL=disabled

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi not found. Install: curl -fsSL https://pixi.sh/install.sh | bash" >&2
  exit 1
fi

if ss -tlnp 2>/dev/null | grep -q ":${STREAMER_PORT} "; then
  echo "Port ${STREAMER_PORT} already in use." >&2
  exit 1
fi

cd "$AVTR_ROOT"
echo "AVTR1_LOCAL_STORAGE=${AVTR1_LOCAL_STORAGE}"
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "UI:  https://${RUNPOD_POD_ID}-${STREAMER_PORT}.proxy.runpod.net/"
  echo "API: https://${RUNPOD_POD_ID}-${RENDERER_PORT}.proxy.runpod.net/health (internal)"
else
  echo "UI:  http://127.0.0.1:${STREAMER_PORT}/"
fi

exec pixi run -e streamer python scripts/run_local_stream.py