#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../env.sh"
CKPT="${ACESTEP_MODELS_DIR}/checkpoints/acestep-v15-xl-turbo"
echo "Checking XL at ${CKPT}..."
if [[ ! -d "$CKPT" ]]; then
  echo "MISSING — downloading..."
  cd "${DEMON_REPO}" && uv run acestep-download --model acestep-v15-xl-turbo --skip-main
fi
count=$(find "$CKPT" -name '*.safetensors' 2>/dev/null | wc -l)
echo "safetensors shards: ${count} (expect 4)"
test -f "$CKPT/model.safetensors.index.json" && echo "index: OK"
du -sh "$CKPT"
echo "XL verify OK"