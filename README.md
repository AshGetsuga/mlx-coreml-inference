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
| Export parity | All four models (PyTorch / CoreML × fp32 / fp16) produce **identical greedy tokens over 242 generated tokens**; CoreML fp16 logit error is the same order of magnitude as PyTorch fp16's ([details](#parity-is-it-still-the-same-model)) |
| Latency, full 256-token forward pass | **0.45 ms** on CoreML `CPU_AND_NE` (fp16), down from 0.89 ms after restructuring the mask ([details](#coreml-benchmarks-ml_bench)) |

**Caveats, stated up front:**
- Parity is one prompt, greedy decoding, and reports raw differences only (no pass/fail tolerances). Repeating it across several prompts is next.
- PyTorch latency comes from a separate script and run (`bench/bench_pytorch.py`), not from `ML_Bench`, so compare it with the CoreML numbers loosely.

## Parity: is it still the same model?

Four models from the same trained weights, compared pairwise (`parity/compare_raw.py`, raw output in [`results/parity_raw.txt`](results/parity_raw.txt)):

| model | runs on |
|---|---|
| PyTorch fp32 | CPU (the reference) |
| PyTorch fp16 | MPS |
| CoreML fp32 | `ALL` |
| CoreML fp16 | `CPU_AND_NE` (input padded to 256 so the Neural Engine actually runs; see [hardware notes](#hardware-notes-what-actually-mattered)) |

Tokens are each model's own greedy decode. Logits are teacher-forced: every model gets PyTorch fp32's decoded sequence as input, so every position is comparable even if the decodes had diverged.

| comparison | what changes | tokens (242) | argmax | logit \|diff\| mean | std |
|---|---|---|---|---|---|
| PyTorch fp32 vs CoreML fp32 | framework | identical | identical | 1.1e-6 | 1.1e-6 |
| PyTorch fp32 vs PyTorch fp16 | precision | identical | identical | 2.4e-3 | 2.5e-3 |
| CoreML fp32 vs CoreML fp16 | precision | identical | identical | 4.0e-3 | 3.9e-3 |
| PyTorch fp32 vs CoreML fp16 | precision + framework | identical | identical | 4.0e-3 | 3.9e-3 |
| PyTorch fp16 vs CoreML fp16 | framework (both fp16) | identical | identical | 4.2e-3 | 4.6e-3 |
| PyTorch fp16 vs CoreML fp32 | precision + framework | identical | identical | 2.4e-3 | 2.5e-3 |

**What it says:**
- **The fp32 conversion is close to exact.** The two fp32 models differ by a mean of 1.1e-6: different kernels and summation order, nothing more. That is the noise floor.
- **fp16 adds about 2000×,** in line with the precision gap between the formats (2^-11 vs 2^-24 relative rounding error, a factor of 2^13 ≈ 8000).
- **No evidence the conversion adds error beyond fp16 itself.** CoreML fp16 on the Neural Engine is 4.0e-3 from the reference, PyTorch fp16 is 2.4e-3: the same order of magnitude.
- **The two fp16 models round independently.** They differ from each other (4.2e-3) by more than either differs from fp32. This is what independent rounding errors look like; a conversion bug would more likely show up as one model far off, or a shared systematic error.

## CoreML benchmarks (`ML_Bench`)

Every CoreML compute unit at fp16 and fp32, plus the per-op device placement from `MLComputePlan`. Raw results (every iteration, plus per-op plans) are in [`ML_Bench/results`](ML_Bench/results); results from before the mask change are kept in [`ML_Bench/results_before_deterministic_mask`](ML_Bench/results_before_deterministic_mask).

**Latency, ms (100 timed iterations after 10 warmup; one full forward pass, input `(1, 256)`):**

| compute units | fp16 mean | fp16 p50 | fp16 std | fp32 mean | fp32 p50 | fp32 std |
|---|---|---|---|---|---|---|
| `CPU_AND_NE` | **0.45** | **0.45** | 0.03 | — | — | — |
| `CPU_ONLY` | 0.67 | 0.67 | 0.01 | 1.16 | 1.16 | 0.02 |
| `CPU_AND_GPU` | 0.97 | 0.77 | 0.40 | 0.98 | 0.79 | 0.36 |
| `ALL` | 0.96 | 0.77 | 0.41 | 0.88 | 0.86 | 0.10 |

`CPU_AND_NE` is not run at fp32: the Neural Engine is fp16-only, so it would just fall back to the CPU.

**Building the mask once halved Neural Engine latency.** `MLComputePlan` showed the ANE can run 98% of the model's estimated cost. The exceptions are the causal-mask ops (`band_part`, `sub`, `range_1d`), which run on the CPU. Before, every decoder block rebuilt its own mask, so execution switched devices 9 times per forward pass (CPU → ANE → CPU → …). Building the mask once in `GPT.forward` and passing it to every block leaves a single CPU segment at the start, then the whole model on the ANE.

| fp16 `CPU_AND_NE` | device switches | mean | std |
|---|---|---|---|
| mask built in every block | 9 | 0.89 ms | 0.05 |
| mask built once per forward pass | **1** | **0.45 ms** | 0.03 |

Parity was re-checked after the change and still matches.

**What the rest of the numbers say:**
- **`ALL` is not the fastest choice.** Under `ALL`, CoreML places all 109 ops on the GPU, both before and after the mask change, even though the ANE is 2× faster here. Pick compute units explicitly and measure; don't rely on the default.
- **Precision matters on the CPU, not on the GPU.** fp32 is 1.7× slower than fp16 on the CPU, but GPU latency is about the same at either precision (~0.8 ms p50). At this model size the GPU looks overhead-bound, not compute-bound. fp32 does just enough extra work per op for the GPU to beat the CPU; fp16 doesn't.
- **The GPU has slow streaks.** The GPU configurations run ~0.77 ms at the median but have bursts of up to ~11 consecutive iterations above 1 ms, which drive their std to ~0.4. The bursts show up in every configuration that uses the GPU and vary from one run to the next (fp32 `CPU_AND_GPU` std was 0.10 in one run and 0.36 in another). They never appear on the CPU or ANE, which stay within ±0.03 ms. The cause is not yet confirmed (likely GPU clock or power-state changes; the GPU also drives the display).
- **The ANE costs more to load:** ~440 ms to load and compile versus ~75 ms for the CPU. That's a one-time startup cost on device.

**Conditions:** M4 Pro, macOS 26.6.2, coremltools 9.0. Timed with `time.perf_counter()` around Python `predict()`, so binding overhead is included. Power source and background load not yet recorded. Numbers vary by about ±0.05 ms between runs.

## PyTorch benchmarks (`bench/bench_pytorch.py`)

Same model and input `(1, 256)`, 100 timed iterations after 10 warmup, MPS synchronized before stopping the clock. Mean ± std, ms ([raw](results)):

| device | fp32 | fp16 |
|---|---|---|
| CPU | 3.03 ± 0.20 | 14.05 ± 0.48 |
| MPS | 1.98 ± 0.48 | 2.11 ± 0.68 |

- **fp16 speed depends on the framework's kernels, not just the hardware.** On the same CPU, fp16 is 1.7× *faster* than fp32 in CoreML but 4.6× *slower* in PyTorch, which suggests PyTorch's CPU fp16 path isn't optimized for this chip.
- **On the GPU, precision is within noise again** (1.98 vs 2.11 ms, std ~0.5–0.7), matching the CoreML GPU result.
- These runs were not on an idle machine, so treat the gap to CoreML as indicative only.

## Hardware notes: what actually mattered

- **A `-inf` causal mask turns into NaN on device.** With a flexible sequence length, CoreML builds the mask at runtime in fp16 and implements `triu` as `x − band_part(x)`, so every kept cell becomes `-inf − (-inf) = NaN`. `-1e9`/`-1e14` fail the same way (they round to `-inf` in fp16); a finite `-1e4` fixes it. A fixed shape or fp32 hides the bug. Full write-up and repro: [`experiments/mask_nan`](experiments/mask_nan/README.md).
- **With a flexible input length, the Neural Engine only runs at the default shape.** The model accepts lengths 1–256 (`RangeDim`, default 256). Under `CPU_AND_NE`, any length other than 256 silently falls back to the CPU: output bit-identical to `CPU_ONLY`, same latency, no warning. The parity script pads every input to 256 for this reason. (`EnumeratedShapes` is the likely fix; not yet tested.)
- **Timers lie on Apple Silicon unless you synchronize.** MLX is lazy (`mx.eval()` forces compute) and PyTorch MPS is async (`torch.mps.synchronize()`); without them you time the dispatch, not the work.
- **Fused kernels decide the winner, not the framework.** See [`experiments/mlx_vs_pytorch`](experiments/mlx_vs_pytorch/log.md): PyTorch won with a fused interpolate kernel, MLX won once both used `ConvTranspose2d`.
- **MLX and PyTorch share weight layout for `Linear`** (`(out, in)`), so MLX `.npz` weights load into PyTorch by name with no transposes.

## Run it

```bash
pip install -r requirements.txt
# all commands from the repo root
python -m train.train_mlx          # train in MLX → weights/gpt_shakespeare.npz (data downloads on first run)
python -m train.generate_mlx       # sample: greedy / temperature / top-k
python -m export.export_coreml     # → artifacts/gpt_shakespeare.mlpackage (add --precision fp32 for the fp32 model)
python -m export.generate_coreml   # sample from the CoreML model

python -m parity.pred_pytorch && python -m parity.pred_coreml && python -m parity.compare_raw
python -m bench.bench_coreml
python -m bench.bench_pytorch --device mps --precision half    # --device mps|cpu, --precision full|half

python -m ML_Bench.plan --precision fp16 --units CPU_AND_NE   # per-op device placement (MLComputePlan)
python -m ML_Bench.bench                                      # latency: every compute unit × fp16/fp32
```

## Layout

```
config.py, tokenizer.py   shared settings + char tokenizer
train/                    MLX model, training, sampling
export/                   PyTorch mirror model, CoreML export, CoreML sampling
parity/                   PyTorch vs CoreML greedy-decode comparison
bench/                    latency benchmarks (CoreML vs PyTorch)
ML_Bench/                 CoreML compute-unit × precision benchmarks + MLComputePlan op placement
weights/, results/        trained weights, training log
experiments/              side experiments: MLX vs PyTorch kernels, KV cache (WIP),
                          UNet in MLX, batching study, speculative-decoding design notes,
                          mask_nan/ (CoreML -inf mask NaN root cause + repro)
```

## Roadmap

- [x] Root-cause the `-inf` mask NaN ([write-up](experiments/mask_nan/README.md))
- [x] Verify per-op device placement with `MLComputePlan` ([results](#coreml-benchmarks-ml_bench))
- [x] fp16 vs fp32 latency across every CoreML compute unit
- [ ] Add PyTorch CPU/MPS to the same benchmark at matched precision
- [ ] Precompute the mask as a registered buffer: does removing the last CPU segment help?
- [ ] Repeat runs with recorded conditions (power source, idle) to confirm the GPU slow streaks
- [ ] Scale up: a real small LLM (Qwen2.5-0.5B) on MLX and CoreML with stateful KV cache and int8/int4 quantization: tokens/sec, memory, perplexity
