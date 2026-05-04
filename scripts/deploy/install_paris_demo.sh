#!/bin/bash
# DEMON paris-demo installer.
#
# Migrates an existing rtmg paris-demo box (set up via the rtmg
# deploy/vast/deploy.sh) over to DEMON. Reuses the ACE-Step model
# checkpoints and TRT engines already on disk under
# ~/.daydream-scope/models/rtmg/, moves the LoRAs into the new
# ~/.daydream-scope/models/demon/loras/ location, copies the ambient
# video into this repo, then launches the demo.
#
# Usage (from the cloned DEMON repo root):
#     bash scripts/deploy/install_paris_demo.sh [path-to-rtmg-checkout]
#
# Defaults rtmg path to ../rtmg (parallel-install layout: clone DEMON
# next to rtmg, cd into DEMON, run this script).
#
# When it finishes, the demo is running on http://<box-ip>:8765/.

set -euo pipefail

PORT="${PORT:-8765}"
RTMG_DIR="${1:-../rtmg}"

if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; CYAN=$'\033[0;36m'; NC=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; CYAN=""; NC=""
fi
die()     { echo "${RED}ERROR:${NC} $*" >&2; exit 1; }
warn()    { echo "${YELLOW}WARN:${NC} $*" >&2; }
ok()      { echo "${GREEN}OK${NC}   $*"; }
heading() { echo; echo "${CYAN}===${NC} $* ${CYAN}===${NC}"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------------- preflight
heading "Pre-flight checks"

[ -f pyproject.toml ] && [ -d acestep ] || die "Run from the DEMON repo root (no pyproject.toml/acestep/ here: $REPO_ROOT)"
ok "DEMON repo root: $REPO_ROOT"

if [ ! -d "$RTMG_DIR" ]; then
    die "rtmg checkout not found at: $RTMG_DIR
       Pass the path explicitly:
         bash scripts/deploy/install_paris_demo.sh /path/to/rtmg"
fi
RTMG_DIR="$(cd "$RTMG_DIR" && pwd)"
ok "rtmg checkout: $RTMG_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. (rtmg ran on this box, so the driver should be present.)"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME"

# -------------------------------------------------------------------- uv
heading "Verifying uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv install failed."
ok "uv $(uv --version | awk '{print $NF}')"

# -------------------------------------------------------------------- Python deps
heading "Installing Python dependencies (DEMON)"
uv sync --frozen || uv sync
uv pip install librosa sounddevice
ok "Python deps installed"

# -------------------------------------------------------------------- model + engines: reuse from rtmg
heading "Reusing model checkpoints + TRT engines from rtmg"
RTMG_MODELS="$HOME/.daydream-scope/models/rtmg"
DEMON_MODELS="$HOME/.daydream-scope/models/demon"
[ -d "$RTMG_MODELS/checkpoints" ] || die "rtmg checkpoints missing at $RTMG_MODELS/checkpoints (was the rtmg deploy completed?)"
[ -d "$RTMG_MODELS/trt_engines" ] || die "rtmg TRT engines missing at $RTMG_MODELS/trt_engines (was the rtmg deploy completed?)"

mkdir -p "$DEMON_MODELS"

if [ ! -e "$DEMON_MODELS/checkpoints" ]; then
    ln -s "$RTMG_MODELS/checkpoints" "$DEMON_MODELS/checkpoints"
    ok "linked checkpoints: $DEMON_MODELS/checkpoints -> $RTMG_MODELS/checkpoints"
else
    ok "checkpoints already in place: $DEMON_MODELS/checkpoints"
fi

if [ ! -e "$DEMON_MODELS/trt_engines" ]; then
    ln -s "$RTMG_MODELS/trt_engines" "$DEMON_MODELS/trt_engines"
    ok "linked trt_engines: $DEMON_MODELS/trt_engines -> $RTMG_MODELS/trt_engines"
else
    ok "trt_engines already in place: $DEMON_MODELS/trt_engines"
fi

# -------------------------------------------------------------------- LoRAs: copy to new demon location (rtmg stays intact as fallback)
heading "Copying LoRAs into demon models dir"
RTMG_LORAS="$RTMG_DIR/demos/realtime_motion_graph_web/loras"
DEMON_LORAS="$DEMON_MODELS/loras"
mkdir -p "$DEMON_LORAS"

if [ -d "$RTMG_LORAS" ]; then
    copied=0
    for f in "$RTMG_LORAS"/*.safetensors; do
        [ -e "$f" ] || continue
        name="$(basename "$f")"
        if [ -e "$DEMON_LORAS/$name" ]; then
            ok "  $name: already in $DEMON_LORAS, skipping"
        else
            cp "$f" "$DEMON_LORAS/$name"
            ok "  $name -> $DEMON_LORAS"
            copied=$((copied+1))
        fi
    done
    if [ "$copied" -eq 0 ]; then
        warn "no LoRAs copied (already migrated, or rtmg dir is empty)"
    fi
else
    warn "rtmg LoRA dir not at $RTMG_LORAS, skipping (existing demon LoRAs preserved)"
fi

# -------------------------------------------------------------------- videos: copy into this repo
heading "Copying ambient video into DEMON repo"
RTMG_VIDEOS="$RTMG_DIR/demos/realtime_motion_graph_web/static/videos"
DEMON_VIDEOS="$REPO_ROOT/demos/realtime_motion_graph_web/static/videos"
mkdir -p "$DEMON_VIDEOS"

if [ -d "$RTMG_VIDEOS" ]; then
    copied=0
    for f in "$RTMG_VIDEOS"/*; do
        [ -e "$f" ] || continue
        case "$(basename "$f" | tr 'A-Z' 'a-z')" in
            *.mp4|*.webm|*.mov) ;;
            *) continue ;;
        esac
        name="$(basename "$f")"
        if [ -e "$DEMON_VIDEOS/$name" ]; then
            ok "  $name: already in $DEMON_VIDEOS, skipping"
        else
            cp "$f" "$DEMON_VIDEOS/$name"
            ok "  $name -> $DEMON_VIDEOS"
            copied=$((copied+1))
        fi
    done
    if [ "$copied" -eq 0 ]; then
        warn "no videos copied (already migrated, or rtmg dir is empty)"
    fi
else
    warn "rtmg video dir not at $RTMG_VIDEOS, skipping"
fi

# -------------------------------------------------------------------- 120s TRT engines (idempotent: skips if already built)
heading "Building 120s TRT engine profile (~15-25 min first run, instant on re-run)"
uv run python -m acestep.engine.trt.build --all --duration 120 --decoder-mixed --decoder-refit
ok "120s engines ready"

# -------------------------------------------------------------------- launch
heading "Starting demo on http://0.0.0.0:$PORT/"
exec uv run python -u -m demos.realtime_motion_graph_web --host 0.0.0.0 --port "$PORT" --mode video
