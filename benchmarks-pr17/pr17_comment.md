Thanks for getting the 2B path onto 10.16. Verified on a 5090 against fresh 10.13 baselines, both runs at depth=4, steps=8, vae_window=3.0s, denoise=0.7, same fixture, `--no-fast-vae` so VAE comparison uses the canonical `vae_decode_fp16_3to30s` engine on both sides:

| Metric | 10.13 | 10.16 | Δ |
|---|---:|---:|---:|
| tick mean (DiT decoder TRT) | 44.0 ms | 42.2 ms | -4.1% |
| decode mean (VAE TRT) | 8.3 ms | 4.6 ms | -44.6% |
| tick+decode mean | 52.3 ms | 46.7 ms | -10.7% |
| Wall (28 gens) | 2960 ms | 2680 ms | -9.5% |
| CUDA peak alloc / reserved | 4.04 / 4.05 GiB | 4.04 / 4.05 GiB | unchanged |

Eager controls matched between branches within run-to-run noise; PR-side changes don't leak into the eager path. The 2B engine upgrade is solid.

The XL TRT path I'd ask you to drop from this PR. Two reasons:

**1. b1 at depth=4 defeats the ring buffer.** The whole point of the StreamDiffusion-style streaming pipeline is one batched forward pass per tick across the active slots. With a b1 engine and depth=4, `_trt_forward` micro-batches into 4 sequential single-row TRT calls per tick. That's 4× the kernel launch overhead and zero intra-batch parallelism. Measured impact: XL TRT vs eager is only -24% tick on the 5090, instead of the ~3× speedup the b4/b8 version of the same engine should produce.

**2. The b1 choice isn't actually just a VRAM call.** Tried rebuilding XL at `--batch-max 4 --workspace-gb 20` on the 5090. The engine compiled fine (84s, 7.98 GB on disk, same as b1) but the first tick at B=2 fails immediately:

```
[E] IExecutionContext::inferShapes: IShuffleLayer node_view: reshaping failed
    for tensor: linear_2
    RESHAPE input dims{2, 15360} reshape dims{1, 6, 2560}
    reshape would change volume 30720 to 15360
RuntimeError: infer_shapes error code: -1
```

The dynamo exporter bakes a literal `1` into at least one reshape op during the batch=1 example-input trace, despite `dynamic_shapes={"hidden_states": {0: Dim("batch", min=1, max=8), ...}}` being specified in `export.py:867`. Known dynamo-exporter limitation with reshape/view ops. Even with 80 GB of headroom, this ONNX cannot be served at B>1.

I already worked through the dynamic-batch ONNX patch + b8 XL build separately on `arch/xl` (walks the ONNX, replaces `[1, ...]` shape constants on Reshape nodes with `[-1, ...]`, then rebuilds), with bench data attached on that branch. I'd suggest:

- Keep all the 2B 10.16 work, the runtime per-tensor dtype fix, build metadata sidecars, and the docs.
- Drop the XL profile registration, XL build recipes, and the streaming micro-batch dispatch path. The micro-batch path is only there to support b1 XL, and b1 XL shouldn't ship.

Happy to merge once XL is removed.
