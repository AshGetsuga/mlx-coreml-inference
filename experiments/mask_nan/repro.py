"""Minimal repro: why a float('-inf') causal mask turns into NaN after CoreML conversion.

What it does:
  1. Builds ONLY the causal mask, exactly like export/model_torch.py:
         mask = torch.triu(torch.full((N, N), fill), diagonal=1)
     in two versions:
         - FixedMask:   N is a Python constant (128)
         - DynamicMask: N = x.shape[1], read from the input (what the real GPT does)
  2. Converts each version to CoreML for every fill value (-inf, -1e14, -1e9, -1e4)
     and both compute precisions (fp16 = CoreML default, fp32).
  3. Runs the CoreML model on a zero input, so the output is just the mask itself.
  4. Prints, for each case, how many cells came out NaN / -inf.

A correct 128x128 causal mask has 0 NaN and 8,128 masked (-inf or very negative) cells.

Run from the repo root: python -m experiments.mask_nan.repro
"""
import io
import logging
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)   # hides "torch version not tested"


@contextmanager
def quiet():
    """Capture everything coremltools prints (progress bars, overflow warnings) in a buffer and drop it."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield


with quiet():
    import coremltools as ct

N = 128
FILLS = [float("-inf"), -1e14, -1e9, -1e4]
PRECISIONS = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}


# ── The two mask models ────────────────────────────────────────────────────────
class FixedMask(nn.Module):
    """N is a constant, so coremltools can compute the mask once, during conversion."""
    def __init__(self, fill):
        super().__init__()
        self.fill = fill

    def forward(self, x):
        return x + torch.triu(torch.full((N, N), self.fill), diagonal=1)


class DynamicMask(FixedMask):
    """N comes from the input shape, so CoreML must build the mask on device, at runtime."""
    def forward(self, x):
        n = x.shape[1]
        return x + torch.triu(torch.full((n, n), self.fill), diagonal=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def convert(model, flexible, precision):
    """PyTorch → torch.jit.trace → CoreML. flexible=True uses RangeDim, like export/export_coreml.py."""
    traced = torch.jit.trace(model.eval(), torch.zeros(1, N, dtype=torch.int32))
    seq_len = ct.RangeDim(lower_bound=1, upper_bound=256, default=256) if flexible else N
    with quiet():
        return ct.convert(
            traced,
            inputs=[ct.TensorType(name="token_ids", shape=ct.Shape((1, seq_len)), dtype=np.int32)],
            outputs=[ct.TensorType(name="mask", dtype=np.float32)],
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS13,
        )


def run(mlmodel):
    """Zero input → output is the mask alone."""
    with quiet():
        return mlmodel.predict({"token_ids": np.zeros((1, N), dtype=np.int32)})["mask"]


def runtime_ops(mlmodel):
    """The MIL ops CoreML executes on device (constants left out)."""
    return [op.op_type for op in mlmodel._mil_program.functions["main"].operations if op.op_type != "const"]


# ── Experiment ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Causal mask {N}x{N}. Correct result: NaN = 0, masked cells = {N * (N - 1) // 2}\n")

    for title, cls, flexible in [("FIXED length    (N = 128 constant)", FixedMask, False),
                                 ("FLEXIBLE length (N = x.shape[1], RangeDim) <- the real GPT", DynamicMask, True)]:
        print(f"== {title}")
        print(f"   ops run on device: {' -> '.join(runtime_ops(convert(cls(float('-inf')), flexible, ct.precision.FLOAT16)))}")
        print(f"   {'fill':>6} | {'precision':9} | {'NaN':>5} | {'-inf':>5} | result")
        for fill in FILLS:
            for name, precision in PRECISIONS.items():
                out = run(convert(cls(fill), flexible, precision))
                nan, neg_inf = int(np.isnan(out).sum()), int(np.isneginf(out).sum())
                print(f"   {fill:>6.0e} | {name:9} | {nan:5d} | {neg_inf:5d} | {'BROKEN' if nan else 'ok'}")
        print()

    # Why fp32 survives: CoreML stores the -inf fill as the most negative finite float32, so x - x = 0
    out = run(convert(DynamicMask(float("-inf")), True, ct.precision.FLOAT32))
    print(f"fp32, fill=-inf: masked cells hold {out.min():.4g} (not -inf), so the subtraction gives 0, not NaN")
