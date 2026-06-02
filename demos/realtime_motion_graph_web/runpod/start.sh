#!/usr/bin/env bash
# Start Realtime Motion Graph Web on RunPod (backend + Next.js dev).
# Usage (from anywhere):
#   /workspace/DEMON/demos/realtime_motion_graph_web/runpod/start.sh
# Optional:
#   ACCEL=eager|tensorrt|compile  (default: eager)
#   BACKEND_PORT=1318  WEB_PORT=6660

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

BACKEND_PORT="${BACKEND_PORT:-1318}"
WEB_PORT="${WEB_PORT:-6660}"
ACCEL="${ACCEL:-eager}"

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "RunPod pod: ${RUNPOD_POD_ID}"
  echo "  UI:     https://${RUNPOD_POD_ID}-${WEB_PORT}.proxy.runpod.net/"
  echo "  Engine: https://${RUNPOD_POD_ID}-${BACKEND_PORT}.proxy.runpod.net/"
else
  echo "RUNPOD_POD_ID not set — using local defaults (localhost)."
  echo "  UI: http://127.0.0.1:${WEB_PORT}/"
fi

# Avoid duplicate backends if something is already bound.
if ss -tlnp 2>/dev/null | grep -q ":${BACKEND_PORT} "; then
  echo "Port ${BACKEND_PORT} already in use. Run runpod/stop.sh first." >&2
  exit 1
fi

echo "Starting launcher (accel=${ACCEL})..."
exec uv run python -u -m demos.realtime_motion_graph_web.run \
  -- --host 0.0.0.0 --port "${BACKEND_PORT}" --accel "${ACCEL}"