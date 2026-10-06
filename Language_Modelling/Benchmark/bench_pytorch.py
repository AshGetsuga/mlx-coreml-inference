import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../graph"))

import torch
import numpy as np
import time
from GPT import GPT

WEIGHTS_PATH = "../gpt_shakespeare.npz"
CONTEXT_LEN  = 256
WARMUP       = 10
ITERATIONS   = 100
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE       = "cpu"

# ── Load model + weights ───────────────────────────────────────────────────────
model = GPT().to(DEVICE)
model.eval()

arrays = np.load(WEIGHTS_PATH, allow_pickle=True)
sd     = model.state_dict()
model.load_state_dict({k: torch.from_numpy(np.array(arrays[k])).to(DEVICE) for k in sd})

# ── Input: full context window of zeros (same as CoreML benchmark) ─────────────
x = torch.zeros(1, CONTEXT_LEN, dtype=torch.long, device=DEVICE)

# ── Warmup ─────────────────────────────────────────────────────────────────────
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(x)
        torch.mps.synchronize()  # MPS dispatches async — flush before timing

# ── Benchmark ──────────────────────────────────────────────────────────────────
times = []
with torch.no_grad():
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        _ = model(x)
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

times = np.array(times)
print(f"\nPyTorch ({DEVICE.upper()})  —  input shape (1, {CONTEXT_LEN})  —  {ITERATIONS} iters")
print(f"  Mean : {times.mean():.2f} ms")
print(f"  Std  : {times.std():.2f} ms")
print(f"  Min  : {times.min():.2f} ms")
print(f"  Max  : {times.max():.2f} ms")
