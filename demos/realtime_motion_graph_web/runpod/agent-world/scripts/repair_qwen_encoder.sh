#!/usr/bin/env bash
# Fix "header too small" — re-download empty/corrupt Qwen3-Embedding weights.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
CKPT="${ACESTEP_MODELS_DIR}/checkpoints"
QWEN="${CKPT}/Qwen3-Embedding-0.6B/model.safetensors"
EXPECTED_MIN=$((500*1024*1024))  # ~600MB file

if [[ -f "$QWEN" ]]; then
  sz=$(stat -c%s "$QWEN")
  if [[ "$sz" -ge "$EXPECTED_MIN" ]]; then
    echo "Qwen encoder OK ($(du -h "$QWEN" | cut -f1))"
    exit 0
  fi
  echo "Removing corrupt Qwen weights (${sz} bytes)"
  rm -f "$QWEN"
fi

cd "${DEMON_REPO}"
echo "Downloading Qwen3-Embedding-0.6B from ACE-Step/Ace-Step1.5 ..."
uv run python -u -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
cp = Path('${CKPT}')
path = hf_hub_download(
    'ACE-Step/Ace-Step1.5',
    'Qwen3-Embedding-0.6B/model.safetensors',
    local_dir=cp,
    local_dir_use_symlinks=False,
)
print('Saved:', path)
"

/workspace/DEMON/.venv/bin/python -c "
from safetensors import safe_open
p='${QWEN}'
with safe_open(p, framework='pt') as h:
    print('Verify OK:', len(list(h.keys())), 'tensors')
"
echo "Qwen repair done."