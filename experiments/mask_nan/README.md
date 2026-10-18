# Why a `-inf` causal mask returns NaN after CoreML conversion

**Symptom.** The PyTorch model is correct. The same model converted to CoreML returns NaN logits on the first forward pass. Greedy decoding hides this, because `argmax` over all-NaN logits returns token 0. Top-k sampling exposes it.

**Root cause (one line).** With a flexible sequence length, CoreML builds the causal mask on the device at runtime in fp16. It implements `triu` as `x − band_part(x)`, so with an `-inf` fill, every cell that should be `0` becomes `-inf − (-inf) = NaN`.

**Fix.** Use a finite fill that fits in fp16 (`-1e4`, now used in [`export/model_torch.py`](../../export/model_torch.py)), or precompute the mask as a registered buffer so the device never builds it.

## What CoreML actually runs

`torch.triu(torch.full((N, N), -inf), diagonal=1)` with `N = x.shape[1]` and `RangeDim(1, 256)` lowers to this MIL. It was also confirmed in the real GPT's `.mlpackage` in Xcode:

```
shape → gather → concat           # N = x.shape[1]
fill(shape=[N,N], value=-inf)     # torch.full, fp16
band_part(x, lower=-1, upper=0)   # lower triangle + diagonal (= tril)
sub(x, band_part)                 # x − tril(x)  =  CoreML's triu
```

| cell | `fill` | `band_part` | `sub` |
|---|---|---|---|
| above diagonal (masked) | -inf | 0 | -inf − 0 = **-inf** ✓ |
| diagonal and below (kept) | -inf | -inf | -inf − (-inf) = **NaN** ✗ |

## Evidence

From [`repro.py`](repro.py) (`python -m experiments.mask_nan.repro`; raw output in [`results/mask_nan.txt`](../../results/mask_nan.txt)). It shows a 128×128 mask. A correct mask has 0 NaN and 8,128 masked cells.

| fill | fixed N, fp16 | **flexible N, fp16** | flexible N, fp32 |
|---|---|---|---|
| `-inf` | 0 NaN | **8,256 NaN** | 0 NaN |
| `-1e14` | 0 NaN | **8,256 NaN** | 0 NaN |
| `-1e9` | 0 NaN | **8,256 NaN** | 0 NaN |
| `-1e4` | 0 NaN | 0 NaN | 0 NaN |

8,256 = 128·129/2, exactly the diagonal plus the lower triangle: the cells `triu` should set to zero.

**How to read the table:**
- **Infinity is the trigger, not "large values".** `-1e14` and `-1e9` fail only because they exceed fp16's range (±65,504) and round to `-inf` before the subtraction. `-1e4` is finite, and `-1e4 − (-1e4) = 0`.
- **A fixed N hides the bug.** When `N` is a constant, coremltools computes the mask during conversion and stores a clean constant. The runtime graph is just `cast → add → cast`, and no `triu` runs on device. The bug appears only when the shape is flexible.
- **fp32 hides it differently.** In fp32 CoreML stores the `-inf` fill as `-3.4e38` (the most negative finite fp32 value), and `-3.4e38 − (-3.4e38) = 0`. Converting to fp32 isn't a real fix, because the Neural Engine runs fp16 only.
- **The MLX model's `-1e9` would break too** if it were exported as-is. It's safe in MLX because MLX trains in fp32.

## Dead ends, kept on purpose

- **The conversion-time `RuntimeWarning: overflow encountered in cast` is unrelated.** It fires for the working `-1e4` mask too, and for `-inf` without `triu`, which gives no NaN. With a fixed N, `-1e14` triggers no warning at all. The warning also seemed "inconsistent" at first, but that was Python showing each warning only once per code location; `warnings.simplefilter("always")` shows it every time.
- **Softmax is not the cause.** A correct causal mask always leaves the diagonal finite, so every softmax row has at least one real value. A row of all `-inf` gives NaN in any framework.
- **The weights are clean.** PyTorch matches token-for-token with the same `.npz`.
- **Corrected explanation.** My first public write-up said "-inf overflows on the fp16 cast". That was wrong: `-inf` is representable in fp16, and the NaN comes from the subtraction.

## Still open

- When `N` is fixed, coremltools precomputes the mask without producing NaN. I haven't checked which code path it uses for that.

Exploration log: [`../notebooks/float.ipynb`](../notebooks/float.ipynb).
Environment: coremltools 9.0, torch 2.11.0, macOS 26.6, M4 Pro.
