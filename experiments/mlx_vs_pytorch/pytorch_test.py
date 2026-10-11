import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        # Encoder: 256→128→64→32
        x = F.relu(self.enc1(x)); x = self.pool(x)
        x = F.relu(self.enc2(x)); x = self.pool(x)
        x = F.relu(self.enc3(x)); x = self.pool(x)
        # Decoder: 32→64→128→256 (fused upsample+conv)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x


BATCH_SIZE  = 4
NUM_CLASSES = 10
NUM_ITERS   = 100

device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

model = BottleneckSegNet(num_classes=NUM_CLASSES).to(device)
model.eval()

# PyTorch conv2d: channels-first (N, C, H, W)
x = torch.randn(BATCH_SIZE, 3, 256, 256, device=device)

# Warmup
with torch.no_grad():
    out = model(x)
if device == "mps":  torch.mps.synchronize()
if device == "cuda": torch.cuda.synchronize()

times = []
with torch.no_grad():
    for _ in range(NUM_ITERS):
        start = time.perf_counter()
        out = model(x)
        if device == "mps":  torch.mps.synchronize()   # MPS is async
        if device == "cuda": torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

avg_ms = (sum(times) / NUM_ITERS) * 1000
print(f"PyTorch ({device}) | avg forward pass: {avg_ms:.4f} ms over {NUM_ITERS} iterations")
print(f"      output shape: {tuple(out.shape)}  (expected: ({BATCH_SIZE}, {NUM_CLASSES}, 256, 256))")
