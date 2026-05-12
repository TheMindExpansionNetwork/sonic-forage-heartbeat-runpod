"""Smoke-test ModelOpt FP8 PTQ with the project's torch 2.9.1+cu128.

Goal: confirm that FP8_PER_CHANNEL_PER_TOKEN_CFG actually works end-to-end
on a tiny transformer-shaped module with bf16 base dtype, including the
ONNX export step that TRT will consume.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import modelopt.torch.quantization as mtq


class TinyDiTBlock(nn.Module):
    """Mimics one DiT-ish layer: QKV proj, output proj, MLP up/down."""
    def __init__(self, d=128, h=4):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.mlp_up = nn.Linear(d, 4 * d)
        self.mlp_dn = nn.Linear(4 * d, d)
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
    m = TinyDiTBlock().cuda().bfloat16().eval()

    def cal_loop(model):
        for _ in range(4):
            x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
            model(x)

    cfg_name = "FP8_PER_CHANNEL_PER_TOKEN_CFG"
    cfg = getattr(mtq, cfg_name)
    print(f"Quantizing with {cfg_name}")
    print(f"  cfg top keys: {list(cfg.keys())}")
    print(f"  quant_cfg sample: {list(cfg['quant_cfg'].keys())[:5]}")

    mq = mtq.quantize(m, cfg, cal_loop)
    print("Quantize OK; forward...")
    with torch.no_grad():
        y = mq(torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16))
    print(f"Forward OK: {y.shape} {y.dtype}")

    qmods = {type(c).__name__ for c in mq.modules()}
    print(f"Quantized module types present: {sorted(qmods)}")

    # Confirm we can read the per-channel weight amax / scale tensors that
    # we'll need to refit alongside FP8 weight bytes.
    print()
    print("Probing quantizer attrs on m.q (QKV proj):")
    qmod = mq.q
    for name in ("weight_quantizer", "input_quantizer", "output_quantizer"):
        if hasattr(qmod, name):
            qz = getattr(qmod, name)
            print(f"  {name}: {type(qz).__name__}")
            for attr in ("amax", "_amax", "num_bits", "axis"):
                if hasattr(qz, attr):
                    v = getattr(qz, attr)
                    if torch.is_tensor(v):
                        print(f"    .{attr}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
                    else:
                        print(f"    .{attr}: {v}")

    # Now the big test: can we export this to ONNX with QDQ nodes?
    print()
    print("Attempting ONNX export with QDQ...")
    import tempfile, os
    onnx_path = os.path.join(tempfile.gettempdir(), "fp8_smoke.onnx")
    x = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        try:
            torch.onnx.export(
                mq, (x,), onnx_path,
                input_names=["x"], output_names=["y"],
                dynamic_axes={"x": {0: "B", 1: "T"}, "y": {0: "B", 1: "T"}},
                opset_version=20,
                do_constant_folding=False,
                dynamo=False,
            )
            sz = os.path.getsize(onnx_path) / 1024
            print(f"ONNX export OK: {onnx_path} ({sz:.1f} KB)")
        except Exception as e:
            print(f"torchscript ONNX export FAILED: {type(e).__name__}: {e}")
            print("Trying dynamo path...")
            from torch.export import Dim
            torch.onnx.export(
                mq, (x,), onnx_path,
                input_names=["x"], output_names=["y"],
                dynamic_shapes={"x": {0: Dim("B", min=1, max=8),
                                       1: Dim("T", min=1, max=64)}},
                dynamo=True,
            )
            sz = os.path.getsize(onnx_path) / 1024
            print(f"Dynamo ONNX export OK: {onnx_path} ({sz:.1f} KB)")

    # Inspect the ONNX graph for QDQ nodes
    import onnx
    g = onnx.load(onnx_path, load_external_data=False).graph
    op_counts = {}
    for n in g.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    qdq_ops = {k: v for k, v in op_counts.items() if "Quantize" in k or "DequantizeLinear" in k}
    print(f"ONNX op summary (top): {dict(sorted(op_counts.items(), key=lambda kv: -kv[1])[:8])}")
    print(f"QDQ ops: {qdq_ops}")
    print(f"Initializers: {len(g.initializer)}")
    fp8_inits = []
    for init in g.initializer:
        if init.data_type in (onnx.TensorProto.FLOAT8E4M3FN,
                              onnx.TensorProto.FLOAT8E5M2):
            fp8_inits.append((init.name, init.dims, init.data_type))
    print(f"FP8 initializers: {len(fp8_inits)}")
    for name, dims, dtype in fp8_inits[:5]:
        print(f"  {name}: shape={list(dims)} dtype={dtype}")


if __name__ == "__main__":
    main()
