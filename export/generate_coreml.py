"""Sample text from the exported CoreML model (top-k). Run: python -m export.generate_coreml"""
import numpy as np
import coremltools as ct

import config
from tokenizer import CharTokenizer

MODEL_PATH     = str(config.MLPACKAGE_PATH)
CONTEXT_LEN    = config.CONTEXT_LEN
MAX_NEW_TOKENS = 300

# The .mlpackage stores weights + compute graph only — not the tokenizer.
tok    = CharTokenizer()
encode = tok.encode
decode = tok.decode

# ── Load CoreML model ─────────────────────────────────────────────────────────
# MLModel is a Python handle to the compiled hardware graph.
# On first load, CoreML compiles the MIL ops to device-specific bytecode
# (Metal shaders for GPU, ANE kernels for Neural Engine).
model = ct.models.MLModel(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}\n")

# ── Sampling ──────────────────────────────────────────────────────────────────
def sample_top_k(logits: np.ndarray, k: int, temperature: float) -> int:
    # Keep only the top-k logit positions, mask the rest to -inf
    top_k_idx        = np.argpartition(logits, -k)[-k:]
    masked           = np.full_like(logits, -1e9)
    masked[top_k_idx] = logits[top_k_idx]
    # Scale by temperature, then softmax, then draw one sample
    masked  -= masked.max()                 # numerical stability
    probs    = np.exp(masked / temperature)
    probs   /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

# ── Autoregressive generation ─────────────────────────────────────────────────
def generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
             k: int = 40, temperature: float = 0.8) -> str:
    ids = encode(prompt)

    for _ in range(max_new_tokens):
        context = ids[-CONTEXT_LEN:]
        seq_len = len(context)

        # The causal mask was baked at (256, 256) during JIT tracing, so we
        # must always feed exactly CONTEXT_LEN tokens. Right-pad with zeros.
        x = np.zeros((1, CONTEXT_LEN), dtype=np.int32)
        x[0, :seq_len] = context

        out = model.predict({"token_ids": x})

        # logits shape: (1, CONTEXT_LEN, VOCAB_SIZE)
        # Read at the last real position, not the last padded position.
        logits      = out["logits"]              # (1, CONTEXT_LEN, VOCAB_SIZE)
        last_logits = logits[0, seq_len - 1, :]  # (VOCAB_SIZE,)

        next_id = sample_top_k(last_logits, k=k, temperature=temperature)
        ids.append(next_id)

    return decode(ids)

# ── Run ───────────────────────────────────────────────────────────────────────
prompt = "ROMEO:"
print(f"Prompt : {prompt}")
print("-" * 50)
print(generate(prompt))
