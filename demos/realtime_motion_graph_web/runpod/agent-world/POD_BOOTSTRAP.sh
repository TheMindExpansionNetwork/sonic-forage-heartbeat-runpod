#!/usr/bin/env bash
# Run once when a pod starts with this volume attached.
set -euo pipefail
AW="/workspace/agent-world-workspace"
if [[ ! -d "$AW" ]]; then
  echo "Mount network volume at /workspace first." >&2
  exit 1
fi
source "${AW}/env.sh"
"${AW}/scripts/verify_xl.sh"
"${AW}/scripts/repair_qwen_encoder.sh"
grep -q 'agent-world-workspace/env.sh' /root/.bashrc 2>/dev/null || \
  echo 'source /workspace/agent-world-workspace/env.sh 2>/dev/null' >> /root/.bashrc
echo "Bootstrap done. Start: ${AW}/scripts/start_demon_web_xl.sh"
echo "Runbook: ${AW}/docs/RUNBOOK.md"