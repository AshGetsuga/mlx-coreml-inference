"""Train the Shakespeare GPT in MLX. Run from repo root: python -m train.train_mlx"""
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import config
from tokenizer import CharTokenizer, load_text
from train.model_mlx import GPT

# ── Data ──────────────────────────────────────────────────────────────────────
text = load_text()
tok  = CharTokenizer(text)

print(f"Vocab size : {tok.vocab_size}")
print(f"Dataset    : {len(text):,} characters")

data       = np.array(tok.encode(text), dtype=np.int32)
split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]


def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - config.CONTEXT_LEN, size=(config.BATCH_SIZE,))
    x  = np.stack([d[i     : i + config.CONTEXT_LEN    ] for i in ix])
    y  = np.stack([d[i + 1 : i + config.CONTEXT_LEN + 1] for i in ix])
    return mx.array(x), mx.array(y)


# ── Training ──────────────────────────────────────────────────────────────────

model     = GPT(tok.vocab_size)
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=config.LR)

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
        for _ in range(config.EVAL_ITERS):
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
for i in range(1, config.NUM_ITERS + 1):
    x, y = get_batch("train")
    loss = train_step(x, y)

    if i % config.EVAL_EVERY == 0 or i == 1:
        est    = estimate_loss()
        t1     = time.perf_counter()
        ms_per = (t1 - t0) / (config.EVAL_EVERY if i > 1 else 1) * 1000
        print(f"{i:>6}  {est['train']:>10.4f}  {est['val']:>10.4f}  {ms_per:>7.1f}ms")
        t0 = time.perf_counter()

model.eval()
config.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save_weights(str(config.WEIGHTS_PATH))
print(f"\nWeights saved to {config.WEIGHTS_PATH}")
print("\nGenerating sample:\n")

# ── Generate ──────────────────────────────────────────────────────────────────
def generate(prompt, max_new_tokens=200):
    ids = mx.array([tok.encode(prompt)], dtype=mx.int32)
    for _ in range(max_new_tokens):
        ctx     = ids[:, -config.CONTEXT_LEN:]
        logits  = model(ctx)
        next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        ids     = mx.concatenate([ids, next_id], axis=1)
    mx.eval(ids)
    return tok.decode(ids[0].tolist())

print(generate("ROMEO:", max_new_tokens=300))
