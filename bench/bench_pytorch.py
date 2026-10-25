"""PyTorch forward-pass latency at full context.

Run:      python -m bench.bench_pytorch [--precision full|half] [--device mps|cpu]
Outputs:  results/bench_pytorch_<device>_<precision>.txt
"""
import argparse
import time

import numpy as np
import torch

import config
from export.model_torch import GPT, load_mlx_weights

WARMUP     = 10
ITERATIONS = 100

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["full", "half"], default="full", help="full = fp32, half = fp16")
parser.add_argument("--device", choices=["mps", "cpu"], default="mps" if torch.backends.mps.is_available() else "cpu")
args = parser.parse_args()
PRECISION, DEVICE = args.precision, args.device


def sync():
    # MPS dispatches async — flush before stopping the clock
    if DEVICE == "mps":
        torch.mps.synchronize()


# ── Load model + weights ───────────────────────────────────────────────────────
model = GPT()
load_mlx_weights(model)
if PRECISION == "half":
    model = model.half() # Halfing precision
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
text = (f"PyTorch ({DEVICE.upper()}, {PRECISION} precision)  —  input shape (1, {config.CONTEXT_LEN})  —  {ITERATIONS} iters\n"
        f"  Mean : {times.mean():.2f} ms\n"
        f"  Std  : {times.std():.2f} ms\n"
        f"  Min  : {times.min():.2f} ms\n"
        f"  Max  : {times.max():.2f} ms\n")
print(f"\n{text}", end="")

out_path = config.ROOT / "results" / f"bench_pytorch_{DEVICE}_{PRECISION}.txt"
out_path.write_text(text)
print(f"Saved → {out_path.relative_to(config.ROOT)}")
