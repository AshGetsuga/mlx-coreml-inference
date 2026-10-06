import time
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

# ── Hyperparameters ───────────────────────────────────────────────────────────
CONTEXT_LEN = 256
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 4
D_FF        = 512
BATCH_SIZE  = 32
DROPOUT     = 0.2
LR          = 3e-4
NUM_ITERS   = 5000
EVAL_EVERY  = 500
EVAL_ITERS  = 50
WEIGHTS_PATH = "gpt_shakespeare.npz"

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"

try:
    open(DATA_PATH)
except FileNotFoundError:
    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH) as f:
    text = f.read()

chars      = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for c, i in stoi.items()}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda ids: "".join(itos[i] for i in ids)

print(f"Vocab size : {VOCAB_SIZE}")
print(f"Dataset    : {len(text):,} characters")

data       = np.array(encode(text), dtype=np.int32)
split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]


def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - CONTEXT_LEN, size=(BATCH_SIZE,))
    x  = np.stack([d[i     : i + CONTEXT_LEN    ] for i in ix])
    y  = np.stack([d[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    return mx.array(x), mx.array(y)


# ── Model ─────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo      = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=DROPOUT)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.Wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.Wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.Wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # (B, n_heads, T, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)

        # Causal mask: upper triangle = -inf so softmax zeroes them out
        mask   = mx.triu(mx.full((T, T), -1e9), k=1)
        scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out     = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.dropout(self.Wo(out))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=DROPOUT)

    def __call__(self, x):
        return self.dropout(self.fc2(nn.relu(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, d_ff)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))  # pre-norm → attention → residual
        x = x + self.ffn(self.ln2(x))   # pre-norm → FFN      → residual
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, D_MODEL)
        self.blocks   = [DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        self.ln_f     = nn.LayerNorm(D_MODEL)
        self.head     = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.drop_emb = nn.Dropout(p=DROPOUT)

    def __call__(self, x):
        B, T = x.shape
        x    = self.drop_emb(self.tok_emb(x) + self.pos_emb(mx.arange(T)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))                         # (B, T, VOCAB_SIZE)


# ── Training ──────────────────────────────────────────────────────────────────

model     = GPT()
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LR)

n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
print(f"Parameters : {n_params:,}\n")


def loss_fn(model, x, y):
    logits = model(x)                           # (B, T, V)
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T),
        reduction="mean",
    )
    return loss


loss_and_grad = nn.value_and_grad(model, loss_fn)


def train_step(x, y):
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss


def estimate_loss():
    model.eval()   # disable dropout for evaluation
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            loss = loss_fn(model, x, y)
            mx.eval(loss)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()  # re-enable dropout after evaluation
    return out


model.train()
print(f"{'iter':>6}  {'train loss':>10}  {'val loss':>10}  {'ms/iter':>8}")
print("-" * 46)

t0 = time.perf_counter()
for i in range(1, NUM_ITERS + 1):
    x, y = get_batch("train")
    loss = train_step(x, y)

    if i % EVAL_EVERY == 0 or i == 1:
        est    = estimate_loss()
        t1     = time.perf_counter()
        ms_per = (t1 - t0) / (EVAL_EVERY if i > 1 else 1) * 1000
        print(f"{i:>6}  {est['train']:>10.4f}  {est['val']:>10.4f}  {ms_per:>7.1f}ms")
        t0 = time.perf_counter()

model.eval()
model.save_weights(WEIGHTS_PATH)
print(f"\nWeights saved to {WEIGHTS_PATH}")
print("\nGenerating sample:\n")

# ── Generate ──────────────────────────────────────────────────────────────────
def generate(prompt, max_new_tokens=200):
    ids = mx.array([encode(prompt)], dtype=mx.int32)
    for _ in range(max_new_tokens):
        ctx     = ids[:, -CONTEXT_LEN:]
        logits  = model(ctx)
        next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        ids     = mx.concatenate([ids, next_id], axis=1)
    mx.eval(ids)
    return decode(ids[0].tolist())

print(generate("ROMEO:", max_new_tokens=300))
