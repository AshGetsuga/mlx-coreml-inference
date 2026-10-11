# MLX → CoreML: On-Device Transformer Inference on Apple Silicon

Train a small GPT in **MLX**, carry the weights through **PyTorch** into **CoreML**, run it on Apple Silicon (CPU / GPU / Neural Engine), and check at every hop that it is still the same model and how fast it runs.

```
 MLX (train) ──.npz──► PyTorch (mirror model) ──jit.trace──► CoreML .mlpackage ──► CPU / GPU / ANE
                              │                                      │
                              └──────── greedy-decode parity ────────┘
```

## Results (so far)

| | |
|---|---|
| Model | Pre-norm decoder-only GPT · 4 layers · d_model 128 · 4 heads · context 256 · char vocab 65 · **840,704 params** |
| Training (MLX, M4 Pro) | 5,000 steps, ~37 ms/step · **train loss 1.497 / val loss 1.696** ([log](results/shakespeare_train.txt)) |
| Export parity | CoreML reproduces PyTorch's greedy decode **token-for-token over 242 generated tokens** |
| Latency, full 256-token forward pass | **CoreML 0.82 ms** vs **PyTorch MPS 2.13 ms** |

**Caveats, stated up front:**
- CoreML (`mlprogram`, macOS 13 target) computes in **fp16 by default**; PyTorch ran in fp32. Part of the latency gap is precision, not just graph compilation. A precision-matched comparison is in progress.
- Which ops actually land on the Neural Engine has not been verified yet (`MLComputePlan` check in progress). At 840K params, dispatch overhead dominates and CPU-only CoreML is competitive.
- Parity is one prompt, greedy decoding. Logit-level error metrics (max abs diff, KL) are next.

## Hardware notes: what actually mattered

- **The causal mask's fill value matters for export.** A `float('-inf')` mask works in PyTorch but produced NaN logits after CoreML conversion; a finite `-1e4` fixes it. Root cause analysis is in progress; a write-up will follow.
- **Timers lie on Apple Silicon unless you synchronize.** MLX is lazy (`mx.eval()` forces compute) and PyTorch MPS is async (`torch.mps.synchronize()`); without them you time the dispatch, not the work.
- **Fused kernels decide the winner, not the framework.** See [`experiments/mlx_vs_pytorch`](experiments/mlx_vs_pytorch/log.md): PyTorch won with a fused interpolate kernel, MLX won once both used `ConvTranspose2d`.
- **MLX and PyTorch share weight layout for `Linear`** (`(out, in)`), so MLX `.npz` weights load into PyTorch by name with no transposes.

## Run it

```bash
pip install -r requirements.txt
# all commands from the repo root
python -m train.train_mlx          # train in MLX → weights/gpt_shakespeare.npz (data downloads on first run)
python -m train.generate_mlx       # sample: greedy / temperature / top-k
python -m export.export_coreml     # → artifacts/gpt_shakespeare.mlpackage
python -m export.generate_coreml   # sample from the CoreML model

python -m parity.pred_pytorch && python -m parity.pred_coreml && python -m parity.compare
python -m bench.bench_coreml && python -m bench.bench_pytorch
```

## Layout

```
config.py, tokenizer.py   shared settings + char tokenizer
train/                    MLX model, training, sampling
export/                   PyTorch mirror model, CoreML export, CoreML sampling
parity/                   PyTorch vs CoreML greedy-decode comparison
bench/                    latency benchmarks
weights/, results/        trained weights, training log
experiments/              side experiments: MLX vs PyTorch kernels, KV cache (WIP),
                          UNet in MLX, batching study, speculative-decoding design notes
```

## Roadmap

- [ ] Root-cause the `-inf` mask NaN and publish the write-up
- [ ] Precision-matched (fp32 vs fp16) latency matrix across PyTorch CPU/MPS and every CoreML compute unit
- [ ] Verify per-op device placement with `MLComputePlan`
- [ ] Scale up: a real small LLM (Qwen2.5-0.5B) on MLX and CoreML with stateful KV cache and int8/int4 quantization: tokens/sec, memory, perplexity
