#!/usr/bin/env bash
# Rebuild all AVTR TRT engines on this GPU (required after pod/GPU change).
set -euo pipefail

AVTR_ROOT="${AVTR_ROOT:-/workspace/avtr-1}"
export PATH="${HOME}/.pixi/bin:${PATH}"
export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-${AVTR_ROOT}/artifacts}"

cd "$AVTR_ROOT"
echo "AVTR1_LOCAL_STORAGE=${AVTR1_LOCAL_STORAGE}"
echo "Building avtr1 + hubert + renderer engines (pixi env: renderer)..."

pixi run -e renderer python scripts/build_avtr1_engines.py
pixi run -e renderer python scripts/build_hubert_engine.py
pixi run -e renderer python scripts/build_renderer_engines.py

echo "Done. Start UI: demos/sonic_forage_jam/start_avtr.sh"