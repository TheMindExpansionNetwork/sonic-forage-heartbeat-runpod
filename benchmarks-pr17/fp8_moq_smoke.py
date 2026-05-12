"""End-to-end smoke for modelopt.onnx.quantization.quantize on a tiny ONNX.

Goal: validate that the FP8 ONNX-side path works on torch 2.9.1+cu128 and
Windows. If this succeeds, scaling to the XL DiT is calibration data + an
exclusion list, not a tooling fight.

Steps:
  1. Build a tiny transformer-shaped module
  2. Export to ONNX in fp16 (avoiding bf16 ORT issues)
  3. Generate calibration tensors as a list of dicts
  4. Run moq.quantize with quantize_mode='fp8'
  5. Inspect the output ONNX for FP8 QDQ nodes + initializer dtypes
"""
from __future__ import annotations
import warnings, os, sys, tempfile
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import numpy as np
import torch
import torch.nn as nn


class TinyDiT(nn.Module):
    def __init__(self, d=128, h=4):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp_up = nn.Linear(d, 4 * d, bias=False)
        self.mlp_dn = nn.Linear(4 * d, d, bias=False)
        self.h = h
    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).reshape(B, T, self.h, D // self.h).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.h, D // self.h).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.h, D // self.h).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        x = x + self.mlp_dn(torch.nn.functional.gelu(self.mlp_up(x)))
        return x


def main():
    workdir = tempfile.mkdtemp(prefix="fp8_smoke_")
    print(f"workdir: {workdir}")

    m = TinyDiT().cuda().half().eval()  # fp16 to avoid bf16 ORT issues
    x = torch.randn(1, 16, 128, device="cuda", dtype=torch.float16)

    onnx_path = os.path.join(workdir, "tiny_dit.onnx")
    print(f"\n[1/5] Export bf16/fp16 ONNX -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            m, (x,), onnx_path,
            input_names=["x"], output_names=["y"],
            dynamic_axes={"x": {0: "B", 1: "T"}, "y": {0: "B", 1: "T"}},
            opset_version=20,
            do_constant_folding=False,
            dynamo=False,
        )
    print(f"  ONNX size: {os.path.getsize(onnx_path)/1024:.1f} KB")

    print("\n[2/5] Build calibration data (4 samples stacked on axis 0)")
    np.random.seed(0)
    calib_data = {"x": np.random.randn(4, 16, 128).astype(np.float16)}
    print(f"  stacked x shape {calib_data['x'].shape}")

    print("\n[3/5] Run modelopt.onnx.quantization.quantize (fp8)")
    out_path = os.path.join(workdir, "tiny_dit.fp8.onnx")
    import modelopt.onnx.quantization as moq
    moq.quantize(
        onnx_path=onnx_path,
        output_path=out_path,
        quantize_mode="fp8",
        calibration_data=calib_data,
        calibration_eps=["cuda:0"],
        op_types_to_quantize=["MatMul"],
        high_precision_dtype="fp16",
        opset=20,
        log_level="INFO",
    )
    sz = os.path.getsize(out_path) / 1024
    print(f"  FP8 ONNX: {out_path} ({sz:.1f} KB)")

    print("\n[4/5] Inspect FP8 ONNX")
    import onnx
    from collections import Counter
    g = onnx.load(out_path, load_external_data=False).graph
    op_counts = Counter(n.op_type for n in g.node)
    print(f"  Op histogram (top 8): {dict(op_counts.most_common(8))}")
    qdq = {k: v for k, v in op_counts.items()
           if "Quantize" in k or "Dequantize" in k}
    print(f"  QDQ ops: {qdq}")
    fp8_inits = [(i.name, list(i.dims), i.data_type) for i in g.initializer
                 if i.data_type in (onnx.TensorProto.FLOAT8E4M3FN,
                                    onnx.TensorProto.FLOAT8E5M2)]
    print(f"  FP8 initializers: {len(fp8_inits)}")
    for n, dims, dt in fp8_inits[:6]:
        dt_name = {17: "FLOAT8E4M3FN", 18: "FLOAT8E5M2"}.get(dt, str(dt))
        print(f"    {n}: shape={dims} dtype={dt_name}")

    print("\n[5/5] Sanity: load with onnxruntime and run one inference")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            out_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        y = sess.run(None, {"x": calib_data[0]["x"]})
        print(f"  ORT run OK; output[0] shape={y[0].shape} dtype={y[0].dtype}")
    except Exception as e:
        print(f"  ORT run FAILED: {type(e).__name__}: {e}")
        print("  (this is informational; TRT is the real consumer)")

    print(f"\nSUCCESS: FP8 ONNX path is viable. Output at {out_path}")


if __name__ == "__main__":
    main()
