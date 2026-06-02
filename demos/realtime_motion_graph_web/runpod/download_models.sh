#!/usr/bin/env bash
# Download ACE-Step main checkpoints (~10 GB). Required before Play works.
# Run once per fresh RunPod volume:
#   ./demos/realtime_motion_graph_web/runpod/download_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

echo "Downloading main model into ~/.daydream-scope/models/demon/checkpoints ..."
echo "This is ~10 GB and can take 10–30+ minutes depending on network."
echo ""

uv run python -u -c "
from acestep.model_downloader import ensure_main_model, check_main_model_exists
from acestep.paths import checkpoints_dir

cp = checkpoints_dir()
cp.mkdir(parents=True, exist_ok=True)
print('checkpoints_dir:', cp)
if check_main_model_exists(cp):
    print('Main model already present — nothing to do.')
else:
    ok, msg = ensure_main_model(cp, prefer_source='huggingface')
    print(msg)
    if not ok:
        raise SystemExit(1)
print('OK — restart the demo if it is already running.')
"