"""
U-Net skip connection study on CIFAR-10 using MLX.

Architecture: Bottleneck encoder-decoder (2 pooling layers for 32x32 input)
  Encoder : Conv+ReLU → MaxPool × 2   (32→16→8 spatial, 3→32→64 channels)
  Bottleneck: Conv+ReLU               (8×8, 64→128 channels)
  Decoder : ConvTranspose+ReLU × 2   (8→16→32 spatial)
    skip=True  — concat encoder features before each decoder conv
    skip=False — straight upsampling, no lateral connections

Usage:
  python unet_mlx.py                  # train both, compare
  python unet_mlx.py --skip true      # train only with skip connections
  python unet_mlx.py --skip false     # train only without
  python unet_mlx.py --epochs 50 --batch_size 128
"""

import argparse
import os
import pickle
import tarfile
import time
import urllib.request

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def _download_cifar10(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    extracted = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
        print("Downloading CIFAR-10 (~163 MB)...")
        urllib.request.urlretrieve(CIFAR10_URL, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir)
        print("Download complete.")
    return extracted


def _load_batch(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # (N, 3072) uint8 → (N, 32, 32, 3) float32 NHWC, normalised to [0, 1]
    data = d[b"data"].astype(np.float32) / 255.0
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(d[b"labels"], dtype=np.int32)
    return data, labels


def load_cifar10(data_dir: str = "./cifar10_data"):
    base = _download_cifar10(data_dir)
    train_x, train_y = [], []
    for i in range(1, 6):
        x, y = _load_batch(os.path.join(base, f"data_batch_{i}"))
        train_x.append(x)
        train_y.append(y)
    train_x = np.concatenate(train_x)   # (50000, 32, 32, 3)
    train_y = np.concatenate(train_y)   # (50000,)
    test_x, test_y = _load_batch(os.path.join(base, "test_batch"))
    return train_x, train_y, test_x, test_y


def get_class_samples(test_x: np.ndarray, test_y: np.ndarray):
    """Return one representative image per CIFAR-10 class (10 total)."""
    images, labels = [], []
    for cls in range(10):
        idx = int(np.where(test_y == cls)[0][0])
        images.append(test_x[idx])
        labels.append(cls)
    return np.stack(images), labels   # (10, 32, 32, 3)


# ─────────────────────────────────────────────
# Augmentations  (applied on CPU via NumPy)
# ─────────────────────────────────────────────
def _augment_single(img: np.ndarray) -> np.ndarray:
    # Random horizontal flip
    if np.random.random() > 0.5:
        img = img[:, ::-1, :].copy()
    # Gaussian noise (σ = 0.02)
    img = img + np.random.normal(0.0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def augment_batch(batch: np.ndarray) -> np.ndarray:
    return np.stack([_augment_single(img) for img in batch])


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class UNet(nn.Module):
    """
    Bottleneck U-Net for image reconstruction (input = target).

    Spatial sizes (32×32 input, 2 pooling layers):
      enc1 → (N, 32, 32, 32)
      pool → (N, 16, 16, 32)
      enc2 → (N, 16, 16, 64)
      pool → (N,  8,  8, 64)
      btn  → (N,  8,  8, 128)
      up1  → (N, 16, 16, 64)   [+ cat enc2 if skip → 128ch]
      dec1 → (N, 16, 16, 64)
      up2  → (N, 32, 32, 32)   [+ cat enc1 if skip →  64ch]
      dec2 → (N, 32, 32, 32)
      out  → (N, 32, 32,  3)
    """

    def __init__(self, skip: bool = True):
        super().__init__()
        self.skip = skip

        # Encoder
        self.enc1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.btn = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Decoder upsamplers (ConvTranspose: k=4, s=2, p=1 → exact 2× spatial)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64,  32, kernel_size=4, stride=2, padding=1)

        # Decoder refinement convs — channel-in doubles when skip concat is applied
        dec1_in = 128 if skip else 64   # 64 (up1 out) + 64 (enc2) = 128
        dec2_in = 64  if skip else 32   # 32 (up2 out) + 32 (enc1) = 64
        self.dec1 = nn.Conv2d(dec1_in, 64, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(dec2_in, 32, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

    def __call__(self, x):
        # ── Encoder ──────────────────────────────────────────
        e1 = nn.relu(self.enc1(x))        # (N, 32, 32, 32)
        e2 = nn.relu(self.enc2(self.pool(e1)))  # (N, 16, 16, 64)
        b  = nn.relu(self.btn(self.pool(e2)))   # (N,  8,  8, 128)

        # ── Decoder ──────────────────────────────────────────
        u1 = nn.relu(self.up1(b))          # (N, 16, 16, 64)
        if self.skip:
            u1 = mx.concatenate([u1, e2], axis=-1)   # (N, 16, 16, 128) — channel axis last
        d1 = nn.relu(self.dec1(u1))        # (N, 16, 16, 64)

        u2 = nn.relu(self.up2(d1))         # (N, 32, 32, 32)
        if self.skip:
            u2 = mx.concatenate([u2, e1], axis=-1)   # (N, 32, 32, 64)
        d2 = nn.relu(self.dec2(u2))        # (N, 32, 32, 32)

        return mx.sigmoid(self.out_conv(d2))          # (N, 32, 32,  3)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def _mse_loss(model, x):
    pred = model(x)
    return mx.mean((pred - x) ** 2)


def train_model(
    skip: bool,
    train_x: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
):
    label = "skip=True " if skip else "skip=False"
    print(f"\n{'─'*50}")
    print(f"  Training  [{label}]  |  epochs={epochs}  bs={batch_size}  lr={lr}")
    print(f"{'─'*50}")

    model = UNet(skip=skip)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _mse_loss)

    n = len(train_x)
    steps_per_epoch = n // batch_size

    epoch_losses = []
    all_step_ms = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_loss_sum = 0.0
        epoch_step_ms = []

        for step in range(steps_per_epoch):
            batch_idx = indices[step * batch_size : (step + 1) * batch_size]
            batch = augment_batch(train_x[batch_idx])
            x = mx.array(batch)

            t0 = time.perf_counter()
            loss, grads = loss_and_grad(model, x)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), loss)   # force Metal execution
            t1 = time.perf_counter()

            step_ms = (t1 - t0) * 1000
            epoch_step_ms.append(step_ms)
            all_step_ms.append(step_ms)
            epoch_loss_sum += loss.item()

        avg_loss = epoch_loss_sum / steps_per_epoch
        avg_ms   = float(np.mean(epoch_step_ms))
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1:3d}/{epochs}  |  loss={avg_loss:.5f}  |  step={avg_ms:.1f} ms")

    avg_step_ms = float(np.mean(all_step_ms))
    print(f"\n  → [{label}]  avg step: {avg_step_ms:.2f} ms  |  final loss: {epoch_losses[-1]:.5f}")

    return model, epoch_losses, avg_step_ms


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
def plot_loss_curves(results: dict, save_path: str = "loss_curve.png"):
    """
    results = {
        'skip':    list of per-epoch losses  (optional),
        'no_skip': list of per-epoch losses  (optional),
    }
    """
    plt.figure(figsize=(9, 5))
    colors = {"skip": "steelblue", "no_skip": "tomato"}
    labels = {"skip": "With skip connections", "no_skip": "Without skip connections"}

    for key in ("skip", "no_skip"):
        if results.get(key) is not None:
            epochs = range(1, len(results[key]) + 1)
            plt.plot(epochs, results[key], label=labels[key], color=colors[key], linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Reconstruction Loss — Skip vs No-Skip", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nLoss curve saved → {save_path}")


def plot_reconstructions(
    vis_samples: np.ndarray,
    vis_labels: list,
    model_skip=None,
    model_noskip=None,
    save_path: str = "reconstruction.png",
):
    """
    Rows:
      Row 0 : original inputs
      Row 1 : skip=True  reconstructions  (if model provided)
      Row 2 : skip=False reconstructions  (if model provided)
    """
    row_data   = [("Input", vis_samples)]
    row_labels = [CIFAR10_CLASSES[l] for l in vis_labels]

    for model, tag in [(model_skip, "Skip = True"), (model_noskip, "Skip = False")]:
        if model is None:
            continue
        x = mx.array(vis_samples)
        out = model(x)
        mx.eval(out)
        row_data.append((tag, np.clip(np.array(out), 0.0, 1.0)))

    n_rows = len(row_data)
    n_cols = len(vis_samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))

    for r, (row_title, imgs) in enumerate(row_data):
        for c in range(n_cols):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            ax.imshow(imgs[c])
            ax.axis("off")
            if r == 0:
                ax.set_title(row_labels[c], fontsize=7, pad=3)
        # Row label on the leftmost cell
        left_ax = axes[r, 0] if n_rows > 1 else axes[0]
        left_ax.set_ylabel(row_title, fontsize=9, rotation=0, labelpad=55, va="center")

    plt.suptitle("CIFAR-10 Reconstruction — one image per class", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Reconstruction grid saved → {save_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--skip",
        choices=["true", "false", "both"],
        default="both",
        help='Which model(s) to train: "true", "false", or "both" (default: both)',
    )
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data_dir",   type=str,   default="./cifar10_data")
    args = parser.parse_args()

    run_skip   = args.skip in ("true",  "both")
    run_noskip = args.skip in ("false", "both")

    # ── Load data ─────────────────────────────
    print("Loading CIFAR-10...")
    train_x, train_y, test_x, test_y = load_cifar10(args.data_dir)
    print(f"  train: {train_x.shape}  test: {test_x.shape}")

    vis_samples, vis_labels = get_class_samples(test_x, test_y)

    # ── Train ─────────────────────────────────
    model_skip   = None
    model_noskip = None
    loss_results = {}

    if run_skip:
        model_skip, losses_skip, ms_skip = train_model(
            skip=True,
            train_x=train_x,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        loss_results["skip"] = losses_skip

    if run_noskip:
        model_noskip, losses_noskip, ms_noskip = train_model(
            skip=False,
            train_x=train_x,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        loss_results["no_skip"] = losses_noskip

    # ── Summary ───────────────────────────────
    print(f"\n{'═'*50}")
    print("  Summary")
    print(f"{'═'*50}")
    if run_skip:
        print(f"  skip=True   | final loss={losses_skip[-1]:.5f} | avg step={ms_skip:.2f} ms")
    if run_noskip:
        print(f"  skip=False  | final loss={losses_noskip[-1]:.5f} | avg step={ms_noskip:.2f} ms")
    print(f"{'═'*50}")

    # ── Plots ─────────────────────────────────
    if loss_results:
        suffix = {"true": "skip", "false": "noskip", "both": "comparison"}[args.skip]
        plot_loss_curves(loss_results, save_path=f"loss_curve_{suffix}.png")

    if args.skip == "both":
        plot_reconstructions(
            vis_samples, vis_labels,
            model_skip=model_skip,
            model_noskip=model_noskip,
            save_path="reconstruction_comparison.png",
        )
    elif args.skip == "true":
        plot_reconstructions(
            vis_samples, vis_labels,
            model_skip=model_skip,
            save_path="reconstruction_skip.png",
        )
    elif args.skip == "false":
        plot_reconstructions(
            vis_samples, vis_labels,
            model_noskip=model_noskip,
            save_path="reconstruction_noskip.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
