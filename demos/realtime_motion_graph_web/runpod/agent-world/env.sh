# Agent World — source on every pod attach:  source /workspace/agent-world-workspace/env.sh
export AGENT_WORLD_ROOT="/workspace/agent-world-workspace"
export WORKSPACE_ROOT="/workspace"

# DEMON models (network volume — never ~/.daydream-scope on overlay)
export ACESTEP_MODELS_DIR="${ACESTEP_MODELS_DIR:-/workspace/.daydream-scope/models/demon}"
export DEMON_REPO="${DEMON_REPO:-/workspace/DEMON}"
export DEMON_TD_REPO="${DEMON_TD_REPO:-/workspace/demonTD}"
export AVTR_ROOT="${AVTR_ROOT:-/workspace/avtr-1}"
export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-/workspace/avtr-1/artifacts}"

# XL defaults for jam sessions
export DEMON_CHECKPOINT="${DEMON_CHECKPOINT:-xl}"
export DEMON_ACCEL="${DEMON_ACCEL:-compile}"

# RunPod proxy helpers
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  export DEMON_WEB_URL="https://${RUNPOD_POD_ID}-6660.proxy.runpod.net/"
  export DEMON_WS_URL="wss://${RUNPOD_POD_ID}-1318.proxy.runpod.net/"
  export AVTR_WEB_URL="https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/"
fi

# Stable model path for agents (created by setup script if missing)
mkdir -p "${AGENT_WORLD_ROOT}/models"
[[ -e "${AGENT_WORLD_ROOT}/models/demon" ]] || ln -sfn "${ACESTEP_MODELS_DIR}" "${AGENT_WORLD_ROOT}/models/demon"

# Keep ~/.daydream-scope on network volume
mkdir -p /workspace/.daydream-scope
[[ -L /root/.daydream-scope ]] || ln -sfn /workspace/.daydream-scope /root/.daydream-scope

# Logs
export AGENT_WORLD_LOG_DIR="${AGENT_WORLD_ROOT}/logs"
mkdir -p "${AGENT_WORLD_LOG_DIR}"

echo "Agent World ready: ACESTEP_MODELS_DIR=${ACESTEP_MODELS_DIR} checkpoint=${DEMON_CHECKPOINT} accel=${DEMON_ACCEL}"
[[ -n "${DEMON_WEB_URL:-}" ]] && echo "  Web: ${DEMON_WEB_URL}"