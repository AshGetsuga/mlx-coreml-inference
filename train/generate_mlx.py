"""Sample from the trained MLX model with several decoding strategies. Run: python -m train.generate_mlx"""
import mlx.core as mx
import numpy as np

import config
from tokenizer import CharTokenizer
from train.model_mlx import GPT

MAX_TOKENS = 300
CONTEXT_LEN = config.CONTEXT_LEN

tok    = CharTokenizer()
encode = tok.encode
decode = tok.decode


# ── Load weights ──────────────────────────────────────────────────────────────
model = GPT(tok.vocab_size)
model.load_weights(str(config.WEIGHTS_PATH))
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
