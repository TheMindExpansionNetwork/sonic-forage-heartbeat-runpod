"""Mel-Band RoFormer helpers for the realtime motion graph backend."""

from __future__ import annotations

import gc
import os
import threading
import time
from collections.abc import Collection
from pathlib import Path

import torch
import torchaudio.functional as TAF

from acestep.model_downloader import resolve_melband_roformer_model_path
from scripts.extract_stems_melbandreformer import (
    SAMPLE_RATE as MELBAND_SAMPLE_RATE,
    MelBandRoformer,
    load_model,
    separate_stems,
)

STEM_SOURCE_MODES = frozenset({"full", "vocals", "instruments"})

_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], MelBandRoformer] = {}


def normalize_stem_source_mode(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in STEM_SOURCE_MODES else None


def resolve_upload_stem_source_mode(
    fixture_name: object,
    requested_mode: str | None,
    *,
    known_fixtures: Collection[str],
) -> str | None:
    """Auto-stem user uploads while keeping built-in fixtures cheap by default."""
    if requested_mode is not None:
        return requested_mode
    if isinstance(fixture_name, str) and fixture_name in known_fixtures:
        return None
    return "full"


def extract_upload_stems(
    *,
    waveform: torch.Tensor,
    device: torch.device | str,
    backend_sample_rate: int,
) -> dict[str, torch.Tensor]:
    """Use Mel-Band RoFormer for vocal and instrumental separation.

    The realtime backend runs sources at 48 kHz, while the RoFormer checkpoint
    is trained for 44.1 kHz. The separator handles the downsample internally;
    we resample its returned stems back to the backend sample rate before
    sending overlays or preparing a selected stem as the inference source.
    """
    torch_device = _coerce_device(device)
    t0 = time.time()
    with _INFER_LOCK:
        model = _get_model(torch_device)
        try:
            vocals_44k, instruments_44k = separate_stems(
                model,
                waveform.detach().cpu().float().unsqueeze(0),
                backend_sample_rate,
                torch_device,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
        finally:
            evicted = _evict_model_from_cache(torch_device)
            model = None
            if evicted:
                _collect_device_cache(torch_device)
    print(f"[Server] Mel-Band RoFormer stems complete in {time.time() - t0:.1f}s")

    vocals = _fit_stem_waveform(
        _resample_stem_to_backend_rate(vocals_44k, backend_sample_rate),
        waveform,
    )
    instruments = _fit_stem_waveform(
        _resample_stem_to_backend_rate(instruments_44k, backend_sample_rate),
        waveform,
    )
    return {
        "vocals": vocals.contiguous(),
        "instruments": instruments.contiguous(),
    }


def _coerce_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _fit_stem_waveform(wf: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Coerce decoded model output to the uploaded waveform's [C, N] shape."""
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf.unsqueeze(0)
    wf = wf.detach().to(dtype=torch.float32, device=target.device)
    if wf.shape[0] == 1 and target.shape[0] == 2:
        wf = wf.repeat(2, 1)
    elif wf.shape[0] > target.shape[0]:
        wf = wf[:target.shape[0]]
    elif wf.shape[0] < target.shape[0]:
        wf = torch.cat(
            [wf, wf[-1:].repeat(target.shape[0] - wf.shape[0], 1)],
            dim=0,
        )
    if wf.shape[-1] > target.shape[-1]:
        wf = wf[:, :target.shape[-1]]
    elif wf.shape[-1] < target.shape[-1]:
        wf = torch.nn.functional.pad(wf, (0, target.shape[-1] - wf.shape[-1]))
    return torch.nan_to_num(wf)


def _resolve_model_path() -> Path:
    explicit_path = os.environ.get("MELBAND_ROFORMER_MODEL_PATH")
    if explicit_path:
        return Path(explicit_path).expanduser()

    return resolve_melband_roformer_model_path()


def _get_model(device: torch.device) -> MelBandRoformer:
    model_path, key = _model_cache_key(device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        print(f"[Server] Loading Mel-Band RoFormer model on {device}...")
        t0 = time.time()
        model = load_model(model_path, device)
        _MODEL_CACHE[key] = model
        print(f"[Server] Mel-Band RoFormer loaded in {time.time() - t0:.1f}s")
        return model


def _model_cache_key(device: torch.device) -> tuple[Path, tuple[str, str]]:
    model_path = _resolve_model_path()
    return model_path, (str(model_path), str(device))


def _evict_model_from_cache(device: torch.device) -> bool:
    """Drop the temporary RoFormer so ACE-Step gets its VRAM back immediately."""
    _, key = _model_cache_key(device)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.pop(key, None)
    if model is None:
        return False

    print(f"[Server] Releasing Mel-Band RoFormer model from {device}...")
    del model
    return True


def _collect_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _resample_stem_to_backend_rate(
    stem: torch.Tensor,
    backend_sample_rate: int,
) -> torch.Tensor:
    stem = stem.detach().cpu().float()
    if MELBAND_SAMPLE_RATE == backend_sample_rate:
        return stem
    return TAF.resample(
        stem,
        orig_freq=MELBAND_SAMPLE_RATE,
        new_freq=backend_sample_rate,
    )
