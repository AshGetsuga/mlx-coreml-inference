# MLX vs PyTorch Benchmarking Log

## Environment
- Platform: Apple Silicon (MacOS Darwin 25.3.0)
- MLX env: conda `mlx`
- PyTorch env: conda `workspace`
- PyTorch device: MPS (Metal Performance Shaders)

---

## Experiment 1: Single Linear Layer + ReLU

**Architecture**
- Linear layer: 256 → 512
- Activation: ReLU
- Batch size: 32
- Iterations: 100 (1 warmup pass excluded)

**Key implementation notes**
- MLX uses lazy evaluation — `mx.eval()` must be called after every forward pass or timing measures near-zero no-ops.
- PyTorch MPS is async — `torch.mps.synchronize()` must be called before stopping the clock, equivalent to `mx.eval()`.

**Results**

| Framework      | Device          | Avg forward pass |
|----------------|-----------------|------------------|
| MLX            | Metal           | 0.2207 ms        |
| PyTorch        | MPS             | 0.2182 ms        |

**Conclusion**
Essentially identical. At this scale the bottleneck is kernel launch overhead, not computation. No meaningful signal.

---

## Experiment 2: Bottleneck Segmentation Model

### Architecture Design

Stride-3 convolutions were considered for downsampling but rejected due to non-integer spatial dimension cascades:

```
256 → 85 → 28 → 9   (stride 3, no padding)
```

Upsampling back to 256 from 9 is not a clean integer factor at any stage, complicating the decoder.

**Chosen approach:** same-padding Conv2d + MaxPool2d(2), giving clean 2× symmetry:

```
Stage        | Operation                  | HxW      | Channels
-------------|----------------------------|----------|----------
Input        | —                          | 256x256  |    3
Enc1         | Conv 3x3 (same) + ReLU     | 256x256  |   32
             | MaxPool 2x2                | 128x128  |   32
Enc2         | Conv 3x3 (same) + ReLU     | 128x128  |   64
             | MaxPool 2x2                |  64x64   |   64
Enc3         | Conv 3x3 (same) + ReLU     |  64x64   |  128
             | MaxPool 2x2                |  32x32   |  128  ← bottleneck
Dec1         | Upsample 2x (nearest)      |  64x64   |  128
             | Conv 3x3 (same) + ReLU     |  64x64   |   64
Dec2         | Upsample 2x (nearest)      | 128x128  |   64
             | Conv 3x3 (same) + ReLU     | 128x128  |   32
Dec3         | Upsample 2x (nearest)      | 256x256  |   32
             | Conv 3x3 (same)            | 256x256  |   num_classes
```

**Parameters**
- Batch size: 4
- num_classes: 10
- Iterations: 100 (1 warmup pass excluded)
- Activation: ReLU throughout

**Implementation notes**
- MLX is channels-last (NHWC). Input shape: `(N, H, W, C)`.
- PyTorch is channels-first (NCHW). Input shape: `(N, C, H, W)`.
- MLX upsample implemented via `mx.repeat` on spatial axes (no fused kernel).
- PyTorch upsample uses `F.interpolate(..., mode='nearest')` (fused MPS kernel).

**Results**

| Framework      | Device | Avg forward pass |
|----------------|--------|------------------|
| MLX            | Metal  | 6.27 ms          |
| PyTorch        | MPS    | 4.21 ms          |

PyTorch is ~33% faster.

**Root cause**
The gap comes from the decoder upsampling. MLX's `mx.repeat` (applied twice per upsample — once on H axis, once on W axis) is not a fused op, while PyTorch's `F.interpolate` on MPS is a single optimized Metal kernel. Encoder convolutions are likely comparable between the two.

---

## Experiment 3: Bottleneck Segnet — ConvTranspose2d Decoder

**Hypothesis:** `mx.repeat` (two unfused scatter ops) is the root cause of MLX's decoder overhead. Replacing the upsample+Conv2d pairs with `ConvTranspose2d` — a single fused kernel that expands spatial dims and applies the learned filter simultaneously — should close or reverse the gap.

### Architecture change (decoder only)

`kernel_size=4, stride=2, padding=1` gives exact 2× expansion: `(H-1)*2 - 2 + 4 = 2H`.

```
Stage        | Operation                       | HxW      | Channels
-------------|----------------------------------|----------|----------
             | (encoder unchanged)              |          |
Dec1         | ConvTranspose2d 4x4 s=2 + ReLU  |  64x64   |   64
Dec2         | ConvTranspose2d 4x4 s=2 + ReLU  | 128x128  |   32
Dec3         | ConvTranspose2d 4x4 s=2         | 256x256  |   num_classes
```

Same change applied to both MLX and PyTorch for a fair comparison.

**Results**

| Framework      | Device | Avg forward pass |
|----------------|--------|------------------|
| MLX            | Metal  | 4.63 ms          |
| PyTorch        | MPS    | 5.82 ms          |

**MLX is now ~20% faster.**

**Analysis**

The results confirm the hypothesis. Switching to `ConvTranspose2d`:
- MLX improved by ~1.6 ms (6.27 → 4.63 ms) — the fused Metal kernel eliminated the two-pass `mx.repeat` scatter bottleneck entirely.
- PyTorch slowed by ~1.6 ms (4.21 → 5.82 ms) — MPS's `ConvTranspose2d` kernel appears less optimized than its `F.interpolate` + `Conv2d` path. PyTorch benefited more from the fused interpolation + separate conv than from a single transposed conv.

The crossover reveals that the bottleneck in Experiment 2 was not the decoder operation in principle, but specifically MLX's lack of a fused nearest-neighbor upsample kernel. MLX's `ConvTranspose2d` implementation on Metal is competitive and likely benefits from the same unified memory layout (NHWC) that its regular convolutions use.

---

## MLX Architectural Advantage: Unified Memory

The benchmark above is pure inference with no mid-loop value reads — this is the workload where PyTorch MPS is competitive due to more mature kernel implementations.

MLX's actual advantage surfaces in **training loops** where values are read back from the compute device:

```python
# PyTorch — every .item() call forces a full device sync + CPU transfer
loss = criterion(out, target)
loss.backward()
print(loss.item())            # stalls CPU until GPU finishes, then copies
writer.add_scalar(loss.item()) # same
if loss.item() < best:         # same again
```

On a discrete GPU (NVIDIA), `.to("cuda")` / `.to("cpu")` are real PCIe memcpys — data physically moves between DRAM and VRAM. On Apple Silicon, PyTorch MPS still maintains this abstraction (requiring explicit syncs) even though the underlying hardware uses a single unified DRAM.

MLX has no device abstraction. Arrays live in unified memory and are accessible from both CPU and GPU cores at all times — no sync, no copy, no stall. In a training loop that logs metrics every step, this difference compounds significantly.

**The right benchmark to expose MLX's advantage:** a training loop with per-step loss logging, not pure inference.

---

## Files

| File               | Description                        |
|--------------------|------------------------------------|
| `mlx_test.py`      | MLX bottleneck segmentation model  |
| `pytorch_test.py`  | PyTorch bottleneck segmentation model |
| `log.md`           | This file                          |
