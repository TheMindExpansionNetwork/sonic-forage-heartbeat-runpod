"""Standalone driver: re-run the FP8 patch on the bf16 XL decoder ONNX.

Useful for iterating on the patch logic without re-invoking the full
build CLI (which would also rebuild the 8 GB engine).

Pass ``--w8a8`` to run the activation-quantized variant using the
per-Linear absmax JSON from ``scripts/collect_activation_absmax.py``.
"""
import argparse
from pathlib import Path

from acestep.engine.trt.fp8_onnx import patch_bf16_onnx_to_fp8

SRC = Path(
    r"C:\Users\ryanf\.daydream-scope\models\demon\trt_engines"
    r"\_onnx_acestep-v15-xl-turbo\decoder_refit\decoder_refit_dynbatch.onnx"
)
AMAX_JSON = Path(
    r"C:\Users\ryanf\.daydream-scope\models\demon\calibration"
    r"\decoder_xl_fp8\activation_absmax.json"
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w8a8", action="store_true",
                    help="Use activation absmax JSON to insert activation "
                         "Q->DQ alongside the weight DQ (W8A8 mode).")
    args = ap.parse_args()

    amax = AMAX_JSON if args.w8a8 else None
    out = patch_bf16_onnx_to_fp8(
        SRC,
        activation_absmax_json_path=amax,
        force=True,
    )
    print(f"Patched ONNX: {out}")
