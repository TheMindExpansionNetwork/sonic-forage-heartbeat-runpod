# RunPod ports (attach this volume)

Expose these TCP ports in your template:

| Port | Use |
|------|-----|
| **6660** | DEMON web UI |
| **1318** | DEMON engine / TouchDesigner WebSocket |
| **7860** | AVTR video (when built later) |
| **8000** | AVTR renderer API (internal) |

After `source env.sh`, URLs print automatically when `RUNPOD_POD_ID` is set.