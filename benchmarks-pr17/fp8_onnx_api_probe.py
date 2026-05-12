"""Probe modelopt.onnx.quantization for FP8 on a pre-existing ONNX.

We want to confirm:
  1. The quantize_static (or equivalent) entry point exists and takes FP8.
  2. Calibration is fed as a sequence of input dicts / data readers.
  3. The output ONNX has FP8 QDQ nodes on MatMul / Gemm inputs.
  4. Per-tensor input-name exclusions work (we'll need to skip the
     AdaLN, time_embed, RMSNorm, proj_out paths by name).
"""
from __future__ import annotations
import warnings, os, sys
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

print("=== modelopt.onnx.quantization surface ===")
import modelopt.onnx.quantization as moq
for a in sorted(dir(moq)):
    if not a.startswith("_"):
        print(" ", a)

print()
print("=== quantize signature ===")
import inspect
if hasattr(moq, "quantize"):
    print(inspect.signature(moq.quantize))
    print(inspect.getdoc(moq.quantize) or "(no doc)")

print()
print("=== quantize_static / quantize_fp8 alternatives ===")
for name in ("quantize_static", "quantize_fp8", "fp8_quantize", "calibrate"):
    if hasattr(moq, name):
        print(f"  {name}: {inspect.signature(getattr(moq, name))}")

print()
print("=== submodules ===")
import pkgutil
for m in pkgutil.iter_modules(moq.__path__, prefix="modelopt.onnx.quantization."):
    print(" ", m.name)
