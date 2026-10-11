import time
import mlx.core as mx
import mlx.nn as nn


class BottleneckSegNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3,   32,  kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Decoder — ConvTranspose2d fuses 2x upsample + conv in one kernel
        # kernel=4, stride=2, padding=1 → exact 2x spatial expansion
        self.dec1 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(32,  num_classes, kernel_size=4, stride=2, padding=1)

    def __call__(self, x):
        # Encoder: 256→128→64→32
        x = nn.relu(self.enc1(x)); x = self.pool(x)
        x = nn.relu(self.enc2(x)); x = self.pool(x)
        x = nn.relu(self.enc3(x)); x = self.pool(x)
        # Decoder: 32→64→128→256 (fused upsample+conv)
        x = nn.relu(self.dec1(x))
        x = nn.relu(self.dec2(x))
        x = self.dec3(x)
        return x


BATCH_SIZE  = 4
NUM_CLASSES = 10
NUM_ITERS   = 100

model = BottleneckSegNet(num_classes=NUM_CLASSES)
mx.eval(model.parameters())

# MLX conv2d: channels-last (N, H, W, C)
x = mx.random.normal(shape=(BATCH_SIZE, 256, 256, 3))
mx.eval(x)

# Warmup — lets Metal JIT compile the graph before timing
out = model(x)
mx.eval(out)

times = []
for _ in range(NUM_ITERS):
    start = time.perf_counter()
    out = model(x)
    mx.eval(out)   # flush lazy graph — mandatory for accurate timing
    end = time.perf_counter()
    times.append(end - start)

avg_ms = (sum(times) / NUM_ITERS) * 1000
print(f"MLX  | avg forward pass: {avg_ms:.4f} ms over {NUM_ITERS} iterations")
print(f"      output shape: {out.shape}  (expected: ({BATCH_SIZE}, 256, 256, {NUM_CLASSES}))")
