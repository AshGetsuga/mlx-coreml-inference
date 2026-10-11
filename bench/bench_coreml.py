"""CoreML forward-pass latency at full context. Run: python -m bench.bench_coreml"""
import time

import numpy as np
import coremltools as ct

import config

WARMUP     = 10
ITERATIONS = 100
# Edit to compare compute units: ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE (CLI flags come in W3)
COMPUTE_UNITS = ct.ComputeUnit.ALL

# ── Load CoreML model ──────────────────────────────────────────────────────────
model = ct.models.MLModel(str(config.MLPACKAGE_PATH), compute_units=COMPUTE_UNITS)

# ── Input: full context window of zeros ───────────────────────────────────────
x = {"token_ids": np.zeros((1, config.CONTEXT_LEN), dtype=np.int32)}

# ── Warmup ─────────────────────────────────────────────────────────────────────
for _ in range(WARMUP):
    _ = model.predict(x)

# ── Benchmark ──────────────────────────────────────────────────────────────────
# model.predict() is synchronous — no extra sync needed
times = []
for _ in range(ITERATIONS):
    t0 = time.perf_counter()
    _ = model.predict(x)
    times.append((time.perf_counter() - t0) * 1000)

times = np.array(times)
print(f"\nCoreML  —  input shape (1, {config.CONTEXT_LEN})  —  {ITERATIONS} iters  —  Compute units : {COMPUTE_UNITS.name}")
print(f"  Mean : {times.mean():.2f} ms")
print(f"  Std  : {times.std():.2f} ms")
print(f"  Min  : {times.min():.2f} ms")
print(f"  Max  : {times.max():.2f} ms")
