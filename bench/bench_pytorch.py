"""PyTorch forward-pass latency at full context. Run: python -m bench.bench_pytorch"""
import time

import numpy as np
import torch

import config
from export.model_torch import GPT, load_mlx_weights

WARMUP     = 10
ITERATIONS = 100
# Edit to compare devices: "mps" or "cpu" (CLI flags come in W3)
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"


def sync():
    # MPS dispatches async — flush before stopping the clock
    if DEVICE == "mps":
        torch.mps.synchronize()


# ── Load model + weights ───────────────────────────────────────────────────────
model = GPT()
load_mlx_weights(model)
model = model.to(DEVICE).eval()

# ── Input: full context window of zeros (same as CoreML benchmark) ─────────────
x = torch.zeros(1, config.CONTEXT_LEN, dtype=torch.long, device=DEVICE)

# ── Warmup ─────────────────────────────────────────────────────────────────────
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(x)
        sync()

# ── Benchmark ──────────────────────────────────────────────────────────────────
times = []
with torch.no_grad():
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        _ = model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)

times = np.array(times)
print(f"\nPyTorch ({DEVICE.upper()})  —  input shape (1, {config.CONTEXT_LEN})  —  {ITERATIONS} iters")
print(f"  Mean : {times.mean():.2f} ms")
print(f"  Std  : {times.std():.2f} ms")
print(f"  Min  : {times.min():.2f} ms")
print(f"  Max  : {times.max():.2f} ms")
