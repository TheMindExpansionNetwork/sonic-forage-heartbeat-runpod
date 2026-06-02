#!/usr/bin/env bash
# Stop backend + Next.js dev for the motion graph demo.
set -euo pipefail

echo "Stopping realtime_motion_graph_web processes..."

# Match running Python/Node processes only — not the shell that is *about*
# to exec start.sh (whose argv would also contain "realtime_motion_graph_web.run").
pkill -f "/workspace/DEMON/.venv/bin/python3 -u -m demos.realtime_motion_graph_web" 2>/dev/null || true
pkill -f "python3 -u -m demos.realtime_motion_graph_web --host" 2>/dev/null || true
pkill -f "node_modules/.bin/next dev -p 6660" 2>/dev/null || true

sleep 1

if ss -tlnp 2>/dev/null | grep -qE ':1318 |:6660 '; then
  echo "Some ports still busy (1318/6660). Remaining listeners:"
  ss -tlnp 2>/dev/null | grep -E ':1318 |:6660 ' || true
  echo "Kill manually if needed: fuser -k 1318/tcp 6660/tcp"
else
  echo "Ports 1318 and 6660 are free."
fi