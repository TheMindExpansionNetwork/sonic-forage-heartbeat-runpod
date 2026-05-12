"""Per-Linear activation absmax capture for W8A8 FP8 patching.

Loads the PyTorch XL DiT, hooks every ``nn.Linear`` in
``model.decoder`` with a forward pre-hook that tracks the running
``max(|input|)``, replays the calibration .npz through the model, and
writes a JSON of ``{linear_module_path -> {absmax, weight_shape,
weight_l2_bf16}}`` for the FP8 patch to consume.

Why weight_l2: the dynamo ONNX export anonymizes most Linear weight
initializers as ``val_NNN``. The FP8 patch needs a way to map an ONNX
weight initializer back to its source PyTorch Linear so it can look
up the right activation amax. Shape alone is ambiguous (XL DiT has
hundreds of (2560,1024) MatMuls); shape + L2 norm of the bf16-cast
weight bytes is unique in practice.

Usage::

    uv run python scripts/collect_activation_absmax.py
    uv run python scripts/collect_activation_absmax.py --batch 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.paths import models_dir, checkpoints_dir


def _hash_weight(w: torch.Tensor) -> dict:
    """Stable shape + L2-norm signature for a Linear weight in bf16.

    We compute the L2 norm of the weight tensor AFTER casting to bf16,
    because the ONNX export stores weights as bf16 and the patcher will
    re-load them as bf16. That way the signature here matches what the
    FP8 patch computes on the ONNX side, even though the live PyTorch
    weights are nominally fp32.
    """
    w_bf16 = w.detach().to(torch.bfloat16).contiguous()
    w_fp32 = w_bf16.to(torch.float32)
    return {
        "shape": list(w.shape),
        "l2_bf16": float(w_fp32.pow(2).sum().sqrt().item()),
        # First 4 elements as a tiebreaker; nonzero L2 collisions are
        # vanishingly unlikely but defense in depth costs nothing.
        "head4_bf16": [float(x) for x in w_fp32.flatten()[:4].tolist()],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration .npz (default: "
        "<MODELS_DIR>/calibration/decoder_xl_fp8/calibration.npz)",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="acestep-v15-xl-turbo",
        help="Model checkpoint directory name (default: acestep-v15-xl-turbo)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size to re-shape calibration samples into (default: 4)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <MODELS_DIR>/calibration/"
        "decoder_xl_fp8/activation_absmax.json)",
    )
    args = ap.parse_args()

    cal_path = Path(args.calibration) if args.calibration else (
        models_dir() / "calibration" / "decoder_xl_fp8" / "calibration.npz"
    )
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration .npz not found: {cal_path}")
    out_path = Path(args.output) if args.output else (
        cal_path.parent / "activation_absmax.json"
    )

    print(f"[setup] calibration: {cal_path}")
    print(f"[setup] output:      {out_path}")
    print(f"[setup] checkpoint:  {args.checkpoint}")

    cal = np.load(str(cal_path))
    keys = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
    arrs = {k: cal[k] for k in keys}
    n_samples = arrs["hidden_states"].shape[0]
    if n_samples % args.batch != 0:
        # Drop the tail so reshape works.
        n_use = (n_samples // args.batch) * args.batch
        for k in keys:
            arrs[k] = arrs[k][:n_use]
        n_samples = n_use
    n_batches = n_samples // args.batch
    print(f"[setup] calibration samples: {n_samples} -> {n_batches} batches of {args.batch}")

    handler = ModelContext(
        project_root=str(checkpoints_dir()),
        config_path=args.checkpoint,
        device=args.device,
        use_flash_attention=False,
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    print("[setup] model loaded")

    with handler._load_model_context("model"):
        model = handler.model
        device = handler.device
        dtype = handler.dtype
        print(f"[setup] device={device}  dtype={dtype}")

        # Discover all Linear modules in model.decoder.
        decoder = model.decoder
        linear_modules: dict[str, nn.Linear] = {}
        for name, mod in decoder.named_modules():
            if isinstance(mod, nn.Linear):
                linear_modules[name] = mod
        print(f"[setup] found {len(linear_modules)} nn.Linear modules in model.decoder")

        # Register forward pre-hooks.
        absmax_state: dict[str, float] = {n: 0.0 for n in linear_modules}
        hooks = []

        def _make_hook(linear_name: str):
            def _hook(module, inputs):
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                cur = float(x.detach().abs().max().item())
                if cur > absmax_state[linear_name]:
                    absmax_state[linear_name] = cur
            return _hook

        for name, mod in linear_modules.items():
            hooks.append(mod.register_forward_pre_hook(_make_hook(name)))

        # Replay calibration data through the decoder.
        for k, arr in arrs.items():
            arrs[k] = arr.reshape(n_batches, args.batch, *arr.shape[1:])
        timestep_arr = arrs["timestep"]
        # The exported wrapper uses `timestep` as both timestep and timestep_r,
        # so we replicate the same call shape here.

        print(f"[capture] running {n_batches} batches through model.decoder...")
        for bi in range(n_batches):
            hs = torch.from_numpy(arrs["hidden_states"][bi]).to(device).to(dtype)
            ts = torch.from_numpy(timestep_arr[bi]).to(device).to(dtype)
            enc = torch.from_numpy(arrs["encoder_hidden_states"][bi]).to(device).to(dtype)
            ctx = torch.from_numpy(arrs["context_latents"][bi]).to(device).to(dtype)
            decoder(
                hidden_states=hs,
                timestep=ts,
                timestep_r=ts,
                attention_mask=None,
                encoder_hidden_states=enc,
                encoder_attention_mask=None,
                context_latents=ctx,
                use_cache=False,
                past_key_values=None,
                output_attentions=False,
            )
            if (bi + 1) % 4 == 0:
                print(f"  batch {bi + 1}/{n_batches} done")

        for h in hooks:
            h.remove()

        # Build the output JSON.
        records: dict[str, dict] = {}
        for name, mod in linear_modules.items():
            w_sig = _hash_weight(mod.weight)
            records[name] = {
                "absmax": absmax_state[name],
                "weight_shape": w_sig["shape"],
                "weight_l2_bf16": w_sig["l2_bf16"],
                "weight_head4_bf16": w_sig["head4_bf16"],
            }

        nonzero = sum(1 for r in records.values() if r["absmax"] > 0)
        print(f"[capture] linear modules with nonzero absmax: {nonzero}/{len(records)}")
        amaxes_sorted = sorted(
            [(r["absmax"], n) for n, r in records.items()], reverse=True,
        )
        print("[capture] top 5 amaxes:")
        for v, n in amaxes_sorted[:5]:
            print(f"  {v:9.3f}  {n}")
        print("[capture] bottom 5 amaxes (nonzero):")
        nz = [(v, n) for v, n in amaxes_sorted if v > 0]
        for v, n in nz[-5:]:
            print(f"  {v:9.3e}  {n}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": args.checkpoint,
                "calibration_npz": str(cal_path),
                "batch": args.batch,
                "n_batches": n_batches,
                "n_samples": n_samples,
                "linears": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
