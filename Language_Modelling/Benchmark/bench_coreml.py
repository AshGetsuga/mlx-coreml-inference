import numpy as np
import coremltools as ct
import time

MODEL_PATH  = "../graph/gpt_shakespeare.mlpackage"
CONTEXT_LEN = 256
WARMUP      = 10
ITERATIONS  = 100

# ── Load CoreML model ──────────────────────────────────────────────────────────
model = ct.models.MLModel(MODEL_PATH)

# ── Input: full context window of zeros ───────────────────────────────────────
x = {"token_ids": np.zeros((1, CONTEXT_LEN), dtype=np.int32)}

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
print(f"\nCoreML  —  input shape (1, {CONTEXT_LEN})  —  {ITERATIONS} iters  —  Device : CPU")
print(f"  Mean : {times.mean():.2f} ms")
print(f"  Std  : {times.std():.2f} ms")
print(f"  Min  : {times.min():.2f} ms")
print(f"  Max  : {times.max():.2f} ms")
