#!/usr/bin/env bash
source "$(dirname "$0")/../env.sh"
LOG="${AGENT_WORLD_LOG_DIR}/demon_td.log"
cd "${DEMON_REPO}"
[[ -n "${DEMON_WS_URL:-}" ]] && echo "TouchDesigner URL: ${DEMON_WS_URL}"
exec uv run python -u -m demos.realtime_motion_graph_web \
  --host 0.0.0.0 --port 1318 --accel "${DEMON_ACCEL}" --checkpoint "${DEMON_CHECKPOINT}" \
  2>&1 | tee -a "${LOG}"