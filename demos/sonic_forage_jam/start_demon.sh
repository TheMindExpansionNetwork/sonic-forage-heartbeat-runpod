#!/usr/bin/env bash
# DEMON realtime motion-graph web (6660 + 1318).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export ACESTEP_MODELS_DIR="${ACESTEP_MODELS_DIR:-/workspace/.daydream-scope/models/demon}"

ACCEL="${ACCEL:-compile}"
CHECKPOINT="${CHECKPOINT:-xl}"

cd "$ROOT_DIR"
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "DEMON UI: https://${RUNPOD_POD_ID}-6660.proxy.runpod.net/"
fi

exec uv run python -u -m demos.realtime_motion_graph_web.run \
  -- --host 0.0.0.0 --port 1318 --accel "$ACCEL" --checkpoint "$CHECKPOINT"