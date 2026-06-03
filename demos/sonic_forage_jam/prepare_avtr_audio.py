#!/usr/bin/env python3
"""Resample audio to 16 kHz mono WAV for AVTR offline or manual feeding.

Optional --stem vocals|instruments runs MelBand-RoFormer via DEMON's stack
(requires acestep + GPU, same as the web demo stem extract).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _resample_to_16k_mono(wave: np.ndarray, sr: int) -> np.ndarray:
    import soxr

    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr == 16_000:
        return wave.astype(np.float32)
    return soxr.resample(wave.astype(np.float32), sr, 16_000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--stem",
        choices=("vocals", "instruments"),
        default=None,
        help="Extract stem with DEMON MelBand-RoFormer before resample",
    )
    args = ap.parse_args()

    if args.stem:
        import torch

        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo))
        from acestep.streaming.stems import extract_upload_stems

        wave, sr = sf.read(args.input, dtype="float32", always_2d=False)
        if wave.ndim == 1:
            t = torch.from_numpy(wave).unsqueeze(0)
        else:
            t = torch.from_numpy(wave.T if wave.shape[1] <= 8 else wave)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        stems = extract_upload_stems(
            waveform=t,
            device=device,
            backend_sample_rate=sr,
        )
        wave = stems[args.stem].detach().cpu().numpy()
        if wave.ndim > 1:
            wave = wave.squeeze(0).mean(axis=0)
        out = _resample_to_16k_mono(wave, sr)
    else:
        wave, sr = sf.read(args.input, dtype="float32", always_2d=False)
        out = _resample_to_16k_mono(wave, sr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, out, 16_000, subtype="PCM_16")
    print(f"Wrote {args.output} ({len(out)} samples @ 16 kHz mono)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())