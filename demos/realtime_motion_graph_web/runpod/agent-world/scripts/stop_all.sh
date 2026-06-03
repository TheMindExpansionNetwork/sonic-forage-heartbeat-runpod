#!/usr/bin/env bash
/workspace/DEMON/demos/sonic_forage_jam/stop_jam.sh 2>/dev/null || true
pkill -f build_renderer_engines 2>/dev/null || true
pkill -f build_avtr1_engines 2>/dev/null || true
echo "Stopped DEMON / AVTR build processes."