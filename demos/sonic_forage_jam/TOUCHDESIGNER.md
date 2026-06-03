# TouchDesigner + DEMON (demonTD)

Fun path: **your laptop runs TouchDesigner**, the **GPU pod runs DEMON**, you wire visuals in TD off the live audio CHOP.

## Repos

| Repo | Role |
|------|------|
| [daydreamlive/DEMON](https://github.com/daydreamlive/DEMON) | AI music engine |
| [daydreamlive/demonTD](https://github.com/daydreamlive/demonTD) | TouchDesigner `.tox` operator |
| **Your fork:** [TheMindExpansionNetwork/demonTD](https://github.com/TheMindExpansionNetwork/demonTD) | Same, under your account |

Clone on the pod (already at `/workspace/demonTD`):

```bash
git clone https://github.com/TheMindExpansionNetwork/demonTD.git
```

Prebuilt operator (no TouchDesigner required on the pod):

`/workspace/demonTD/dist/demonTD.tox` — or [release v0.2.11](https://github.com/daydreamlive/demonTD/releases).

## 1. On RunPod — start DEMON for TD (WebSocket only)

TouchDesigner talks to the **Python backend**, not the Next.js UI on 6660.

```bash
cd /workspace/DEMON
export ACESTEP_MODELS_DIR=/workspace/.daydream-scope/models/demon
./demos/sonic_forage_jam/start_demon_td_backend.sh
```

Expose TCP **1318** in RunPod. Note your pod id: `echo $RUNPOD_POD_ID`.

WebSocket URL for TD:

```text
wss://<RUNPOD_POD_ID>-1318.proxy.runpod.net/
```

(Use `ws://127.0.0.1:1318/` only if TD runs on the same machine as the server.)

## 2. On your computer — TouchDesigner

1. Install [TouchDesigner](https://derivative.ca/download) (free non-commercial or license).
2. Copy **`demonTD.tox`** onto your Mac/PC (from the release or scp from the pod).
3. New project → drag **`demonTD.tox`** into the network.
4. **macOS audio (important):** Edit → Preferences → Audio → Audio Device → **None**  
   (so TD doesn’t lock the device PortAudio needs — see demonTD README.)
5. Open the **`demon`** COMP → **Session** page:

| Parameter | Value |
|-----------|--------|
| **Mode** | Direct |
| **Server URL** | `wss://<YOUR_POD_ID>-1318.proxy.runpod.net/` |
| **Source Audio File** | Any WAV/MP3 you want to loop/transform |
| **Python Audio Out** | On (macOS speakers) |

6. Pulse **Connect** → wait for `server ready` in the textport → audio should play.
7. **Prompt+LoRA** → edit prompt → pulse **Send Prompt**.
8. Wire **`demon`** CHOP out → **Analyze** / **Audio Spectrum** / your visuals.

### Vocals vs full track in TD

- **Source file:** use a cappella WAV for “vocals only”, or full mix for “entire track”.
- **On the pod:** DEMON’s web demo can split stems (vocals/instruments) with MelBand-RoFormer; export a WAV with `prepare_avtr_audio.py --stem vocals` and use that as **Source Audio File** in TD.

## 3. Hosted mode (no pod)

If you don’t want to manage RunPod:

- **Mode** → Hosted (Daydream queue)
- Pulse **Paste API Key** from [app.daydream.live/dashboard/api-keys](https://app.daydream.live/dashboard/api-keys)
- Connect

## 4. Python inside TD

```python
d = op('/project1/demon')
d.Connect()
d.SendPrompt('dreamy liquid dnb', key='Am', time_signature='4')
d.SetParams({'denoise': 0.75, 'ch_g0': 1.1})
```

## 5. Jam stack (optional)

| Port | Tool |
|------|------|
| 6660 | DEMON web HUD (browser) |
| 1318 | DEMON engine (TD + API) |
| 7860 | AVTR talking-head video |

See [README.md](./README.md) for AVTR + DEMON together.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| WS failed | Pod running? Port 1318 exposed? URL uses **wss://** on RunPod |
| No audio (Mac) | TD Preferences → Audio Device → **None** |
| Windows | demonTD v0.2 is mac-first; try release anyway, report issues upstream |
| Slow first connect | Models loading on pod; wait 60–90s, watch pod logs |

## Build `.tox` from source (when you edit demonTD)

Requires TouchDesigner on your machine:

1. Open TD → Text DAT → point at `build/build_tox.py` → Run Script  
2. Output: `dist/demonTD.tox`

Log experiments in [NOTES.md](./NOTES.md).