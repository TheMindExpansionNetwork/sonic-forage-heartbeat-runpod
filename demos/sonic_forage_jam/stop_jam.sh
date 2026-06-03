#!/usr/bin/env bash
set -euo pipefail

echo "Stopping jam stack..."
pkill -f "scripts/run_local_stream.py" 2>/dev/null || true
pkill -f "avtr1_renderer.api.app" 2>/dev/null || true
pkill -f "avaturn_live_streamer.local_stream_cli" 2>/dev/null || true
pkill -f "demos.realtime_motion_graph_web.run" 2>/dev/null || true
pkill -f "demos.realtime_motion_graph_web --host" 2>/dev/null || true
pkill -f "next dev -p 6660" 2>/dev/null || true
sleep 1
ss -tlnp 2>/dev/null | grep -E ':7860|:8000|:6660|:1318' || echo "Ports 7860/8000/6660/1318 free."