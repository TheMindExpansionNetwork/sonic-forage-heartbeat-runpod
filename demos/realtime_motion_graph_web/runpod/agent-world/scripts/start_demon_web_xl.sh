#!/usr/bin/env bash
# Full DEMON web UI + engine, XL + compile. Survives shell exit (nohup + pid files).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
LOG="${AGENT_WORLD_LOG_DIR}/demon_web_xl.log"
PIDFILE="${AGENT_WORLD_LOG_DIR}/demon_web_xl.pid"
cd "${DEMON_REPO}"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running (pid $(cat "$PIDFILE")). Log: ${LOG}"
  [[ -n "${DEMON_WEB_URL:-}" ]] && echo "Open: ${DEMON_WEB_URL}"
  exit 0
fi

echo "Starting DEMON XL (nohup) → ${LOG}"
nohup uv run python -u -m demos.realtime_motion_graph_web.run \
  -- --host 0.0.0.0 --port 1318 --accel "${DEMON_ACCEL}" --checkpoint "${DEMON_CHECKPOINT}" \
  >> "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
sleep 4
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Started pid $(cat "$PIDFILE")"
  [[ -n "${DEMON_WEB_URL:-}" ]] && echo "Open: ${DEMON_WEB_URL}"
else
  echo "Failed to start — tail ${LOG}" >&2
  exit 1
fi