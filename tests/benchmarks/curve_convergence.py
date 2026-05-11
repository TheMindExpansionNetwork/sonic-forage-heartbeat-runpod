"""Convergence benchmark for shared mutable curves.

Sanity-check that flipping a per-step parameter via the shared-mutable
path lands in 1-2 ticks, while the per-slot path takes the full ring
drain (~``depth`` ticks).  Reports ticks-to-50%/95% of steady-state
delta for each param, A/B between the two paths where applicable.

Usage::

    python tests/benchmarks/curve_convergence.py

Requires the same TRT engines as ``tests/integration/test_shared_sde_curve.py``;
edit the engine paths at the top of this file if yours live elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True  # type: ignore[attr-defined]

from acestep import paths
from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.session import PreparedSource, Session
from acestep.engine.stream import SlotRequest, StreamPipeline
from acestep.fixtures import audio_fixture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")

SAMPLE_RATE = 48000
SEED = 1528
T = 1500
DEPTH = 8
INFER_STEPS = 8

CHECKPOINTS_DIR = paths.checkpoints_dir()
TRT_ENGINE = paths.trt_engine_path("decoder_mixed_refit_b8_240s")
VAE_ENCODE_ENGINE = paths.trt_engine_path("vae_encode_fp16_240s")
VAE_DECODE_ENGINE = paths.trt_engine_path("vae_decode_fp16_240s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_audio(path: Path, duration: float = 60.0):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(
        data.T if data.ndim > 1 else data.reshape(1, -1),
    )
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(duration * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    from acestep.nodes.types import Audio
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


def _convergence_ticks(
    deltas: list[float], thresholds: tuple[float, ...] = (0.5, 0.95),
) -> dict[float, int | None]:
    """Given per-tick deltas (0 = at-A, 1 = at-B), return ticks-to-X%.

    ``None`` when the threshold was never crossed within the window.
    """
    out: dict[float, int | None] = {}
    for thr in thresholds:
        cross: int | None = None
        for i, d in enumerate(deltas):
            if d >= thr:
                cross = i
                break
        out[thr] = cross
    return out


# ---------------------------------------------------------------------------
# Param plug-ins: how to apply each param in per-slot vs shared mode and how
# to compute "0.0 = at-A, 1.0 = at-B" from a completed latent.
# ---------------------------------------------------------------------------


class ParamPlugin:
    name: str
    has_per_slot_path: bool

    def make_request(
        self, base_kwargs: dict, value, mode: str,
    ) -> SlotRequest: ...

    def apply_shared(self, pipe: StreamPipeline, value) -> None: ...

    def apply_per_slot(self, pipe: StreamPipeline, value) -> None:
        """For pipeline-level params, no per-slot override exists."""
        raise NotImplementedError


class SDECurveParam(ParamPlugin):
    name = "sde_denoise_curve"
    has_per_slot_path = True

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B

    def make_request(
        self, base_kwargs: dict, value, mode: str,
    ) -> SlotRequest:
        return SlotRequest(
            **base_kwargs, sde_denoise_curve=value, denoise=0.75,
        )

    def apply_shared(self, pipe: StreamPipeline, value) -> None:
        pipe.set_shared_curve("sde_denoise_curve", value)

    def apply_per_slot(self, pipe: StreamPipeline, value) -> None:
        # New submissions carry the new value; old in-flight slots
        # keep their original curve until they drain.
        pipe.set_shared_curve("sde_denoise_curve", None)


class ChannelGainParam(ParamPlugin):
    """Pipeline-level: no per-slot path. Measures absolute shared latency."""
    name = "channel_gain"
    has_per_slot_path = False

    def __init__(self, A_configs, B_configs):
        self.A = A_configs
        self.B = B_configs

    def make_request(
        self, base_kwargs: dict, value, mode: str,
    ) -> SlotRequest:
        return SlotRequest(**base_kwargs, denoise=0.75)

    def apply_shared(self, pipe: StreamPipeline, value) -> None:
        pipe.set_channel_guidance(value)


class DCWScalerParam(ParamPlugin):
    """Pipeline-level: ditto."""
    name = "dcw_scaler"
    has_per_slot_path = False

    def __init__(self, A: float, B: float):
        self.A = A
        self.B = B

    def make_request(
        self, base_kwargs: dict, value, mode: str,
    ) -> SlotRequest:
        return SlotRequest(**base_kwargs, denoise=0.75)

    def apply_shared(self, pipe: StreamPipeline, value) -> None:
        pipe.set_dcw(enabled=True, mode="double", scaler=value, high_scaler=0.02)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _drift(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized L2 distance, fp32 on host."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    return float(torch.linalg.vector_norm(a_f - b_f) / torch.linalg.vector_norm(b_f).clamp_min(1e-8))


def _run_phase(
    pipe: StreamPipeline, plugin: ParamPlugin, value, base_kwargs: dict,
    n_ticks: int,
) -> list[torch.Tensor]:
    """Submit + tick ``n_ticks`` times; return completions."""
    out: list[torch.Tensor] = []
    for _ in range(n_ticks):
        pipe.submit(plugin.make_request(base_kwargs, value, mode="warmup"))
        r = pipe.tick()
        if r is not None:
            out.append(r.detach().clone())
    return out


def _bench_param(
    plugin: ParamPlugin, base_kwargs: dict, engine, *,
    config: DiffusionConfig, ref_a: torch.Tensor, ref_b: torch.Tensor,
) -> dict:
    """Drive one param: render at-A reference, flip to B, measure convergence."""
    rep: dict = {"name": plugin.name, "modes": {}}

    modes = ["shared"]
    if plugin.has_per_slot_path:
        modes.append("per_slot")

    for mode in modes:
        pipe = StreamPipeline(engine, config, pipeline_depth=DEPTH)
        # Warmup with A
        if hasattr(plugin, "apply_shared") and not isinstance(plugin, (SDECurveParam,)):
            plugin.apply_shared(pipe, plugin.A)
        _ = _run_phase(pipe, plugin, plugin.A, base_kwargs, n_ticks=DEPTH * 2)

        # Switch
        if mode == "shared":
            plugin.apply_shared(pipe, plugin.B)
            new_value = plugin.A
        else:
            plugin.apply_per_slot(pipe, plugin.B)
            new_value = plugin.B

        # Measure convergence
        deltas: list[float] = []
        for _ in range(DEPTH * 2):
            pipe.submit(plugin.make_request(base_kwargs, new_value, mode=mode))
            r = pipe.tick()
            if r is None:
                continue
            d_a = _drift(r, ref_a)
            d_b = _drift(r, ref_b)
            # 0.0 = identical to ref_a, 1.0 = identical to ref_b (linear interp)
            total = d_a + d_b
            score = 0.0 if total <= 1e-8 else d_a / total
            deltas.append(score)

        rep["modes"][mode] = {
            "deltas": deltas,
            "ticks_to": _convergence_ticks(deltas),
        }

    return rep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading session + TRT engines...")
    session = Session(
        project_root=str(CHECKPOINTS_DIR),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines={
            "decoder": str(TRT_ENGINE),
            "vae_encode": str(VAE_ENCODE_ENGINE),
            "vae_decode": str(VAE_DECODE_ENGINE),
        },
    )
    handler = session.handler
    device = handler.device
    dtype = handler.dtype

    audio = _load_audio(SOURCE_AUDIO)
    latent = session.encode_audio(audio)
    context_latent = session.extract_hints(latent)
    source = PreparedSource(latent=latent, context_latent=context_latent)

    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    entry = cond.to_entries()[0]

    ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
    src_lat = source.latent.tensor.to(device=device, dtype=dtype)
    # Use the actual encoded T (varies slightly with the audio). Curves
    # and the context-mask half match it; the global T constant is just
    # a hint for the audio-loader trim.
    T_actual = src_lat.shape[1]
    D_ctx = ctx_lat.shape[2]
    if ctx_lat.shape[1] != T_actual:
        # extract_hints may downsample; pad/truncate to T_actual so the
        # cat below produces a valid [1, T_actual, 2*D_ctx].
        ctx_lat = torch.nn.functional.pad(
            ctx_lat, (0, 0, 0, max(0, T_actual - ctx_lat.shape[1])),
        )[:, :T_actual]
    cm = torch.ones(1, T_actual, D_ctx, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)
    source_latents = src_lat

    base_kwargs = dict(
        encoder_hidden_states=entry.encoder_hidden_states,
        encoder_attention_mask=entry.encoder_attention_mask,
        context_latents=context_latents,
        seed=SEED,
        source_latents=source_latents,
    )

    engine = handler._diffusion_engine
    config = DiffusionConfig(
        infer_steps=INFER_STEPS, shift=3.0, noise_on_cpu=True,
    )

    # ----- Build per-param at-A / at-B references by running each
    #       config to steady state and grabbing a stable completion.
    def _settle_to(plugin: ParamPlugin, value, n: int = DEPTH * 3) -> torch.Tensor:
        pipe = StreamPipeline(engine, config, pipeline_depth=DEPTH)
        if not isinstance(plugin, SDECurveParam):
            plugin.apply_shared(pipe, value)
        outs = _run_phase(pipe, plugin, value, base_kwargs, n_ticks=n)
        return outs[-1]

    # ----- The three params -----
    plugins: list[ParamPlugin] = []

    A_curve = torch.full((1, T_actual, 1), 0.1, device=device, dtype=dtype)
    B_curve = torch.full((1, T_actual, 1), 0.95, device=device, dtype=dtype)
    plugins.append(SDECurveParam(A_curve, B_curve))

    from acestep.nodes.types import ChannelGuidanceEntry
    plugins.append(ChannelGainParam(
        A_configs=[],  # no gain
        B_configs=[ChannelGuidanceEntry(channel_start=8, channel_end=15, scale=2.0)],
    ))
    plugins.append(DCWScalerParam(A=0.05, B=0.5))

    # ----- Run -----
    reports: list[dict] = []
    for plugin in plugins:
        print(f"\n[{plugin.name}] settling A reference...")
        ref_a = _settle_to(plugin, plugin.A)
        print(f"[{plugin.name}] settling B reference...")
        ref_b = _settle_to(plugin, plugin.B)
        print(f"[{plugin.name}] benchmarking...")
        rep = _bench_param(
            plugin, base_kwargs, engine,
            config=config, ref_a=ref_a, ref_b=ref_b,
        )
        reports.append(rep)

    # ----- Report -----
    print("\n" + "=" * 72)
    print(f"{'param':<24} {'mode':<10} {'t->50%':>8} {'t->95%':>8}")
    print("-" * 72)
    for rep in reports:
        for mode, m in rep["modes"].items():
            t50 = m["ticks_to"][0.5]
            t95 = m["ticks_to"][0.95]
            t50s = "n/a" if t50 is None else str(t50)
            t95s = "n/a" if t95 is None else str(t95)
            print(f"{rep['name']:<24} {mode:<10} {t50s:>8} {t95s:>8}")
    print("=" * 72)
    print("Expectation: shared mode <= 2 ticks; per-slot mode ~ DEPTH ticks.")


if __name__ == "__main__":
    main()
