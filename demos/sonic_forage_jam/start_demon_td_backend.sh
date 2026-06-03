#!/usr/bin/env bash
# DEMON backend for TouchDesigner (WebSocket on 1318).
# Does not start Next.js — use the browser UI separately if you want 6660.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export ACESTEP_MODELS_DIR="${ACESTEP_MODELS_DIR:-/workspace/.daydream-scope/models/demon}"

PORT="${DEMON_TD_PORT:-1318}"
HOST="${DEMON_TD_HOST:-0.0.0.0}"
ACCEL="${ACCEL:-compile}"
CHECKPOINT="${CHECKPOINT:-xl}"

cd "$ROOT_DIR"

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "TouchDesigner WebSocket URL:"
  echo "  wss://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/"
fi

echo "Starting DEMON backend (accel=${ACCEL} checkpoint=${CHECKPOINT}) on ${HOST}:${PORT} ..."
exec uv run python -u -m demos.realtime_motion_graph_web \
  --host "$HOST" --port "$PORT" --accel "$ACCEL" --checkpoint "$CHECKPOINT"