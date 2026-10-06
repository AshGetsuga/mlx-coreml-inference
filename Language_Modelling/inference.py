import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ── Must match train.py exactly ───────────────────────────────────────────────
CONTEXT_LEN  = 256
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
D_FF         = 512
DROPOUT      = 0.2
WEIGHTS_PATH = "gpt_shakespeare.npz"
DATA_PATH    = "shakespeare.txt"
MAX_TOKENS   = 300

# ── Tokenizer ─────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    text = f.read()

chars      = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for c, i in stoi.items()}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda ids: "".join(itos[i] for i in ids)


# ── Model (identical to train.py) ─────────────────────────────────────────────

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
        scores  = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)
        mask    = mx.triu(mx.full((T, T), -1e9), k=1)
        weights = mx.softmax(scores + mask, axis=-1)
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
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb  = nn.Embedding(CONTEXT_LEN, D_MODEL)
        self.blocks   = [DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        self.ln_f     = nn.LayerNorm(D_MODEL)
        self.head     = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.drop_emb = nn.Dropout(p=DROPOUT)

    def __call__(self, x):
        B, T = x.shape
        x = self.drop_emb(self.tok_emb(x) + self.pos_emb(mx.arange(T)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ── Load weights ──────────────────────────────────────────────────────────────
model = GPT()
model.load_weights(WEIGHTS_PATH)
model.eval()   # disables dropout — deterministic from here
mx.eval(model.parameters())


# ── Decoding strategies ───────────────────────────────────────────────────────

def decode_greedy(logits):
    """Always pick the single highest-probability token."""
    return mx.argmax(logits, axis=-1)


def decode_temperature(logits, temperature=1.0):
    """
    Divide logits by T before softmax.
    T < 1 → sharper (more repetitive); T > 1 → flatter (more random).
    """
    return mx.random.categorical(logits / temperature)


def decode_top_k(logits, k=40, temperature=1.0):
    """
    Keep only the top-k candidates, zero out the rest, then apply temperature
    and sample. Prevents the model from ever picking very low-prob tokens.
    """
    logits_np = np.array(logits)
    # Indices of the k largest values
    top_k_idx = np.argpartition(logits_np, -k)[-k:]
    # Build a -inf mask, preserve only top-k values
    masked = np.full_like(logits_np, -1e9)
    masked[top_k_idx] = logits_np[top_k_idx]
    return mx.random.categorical(mx.array(masked) / temperature)


# ── Generation loop ───────────────────────────────────────────────────────────

def generate(prompt, mode="greedy", temperature=1.0, top_k=40, max_new_tokens=MAX_TOKENS):
    ids = mx.array([encode(prompt)], dtype=mx.int32)
    for _ in range(max_new_tokens):
        ctx    = ids[:, -CONTEXT_LEN:]
        logits = model(ctx)[0, -1, :]   # (vocab_size,) — last position only

        if mode == "greedy":
            next_id = decode_greedy(logits)
        elif mode == "temperature":
            next_id = decode_temperature(logits, temperature)
        elif mode == "top_k":
            next_id = decode_top_k(logits, top_k, temperature)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ids = mx.concatenate([ids, next_id.reshape(1, 1)], axis=1)

    mx.eval(ids)
    return decode(ids[0].tolist())


# ── Run all three strategies on the same prompt ───────────────────────────────

PROMPT = "ROMEO:"

configs = [
    ("Greedy",                dict(mode="greedy")),
    ("Temperature  (T=0.8)",  dict(mode="temperature", temperature=0.8)),
    ("Temperature  (T=1.2)",  dict(mode="temperature", temperature=1.2)),
    ("Top-k=40, T=0.8",       dict(mode="top_k", top_k=40, temperature=0.8)),
    ("Top-k=10, T=0.8",       dict(mode="top_k", top_k=10, temperature=0.8)),
]

for label, kwargs in configs:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(generate(PROMPT, **kwargs))
