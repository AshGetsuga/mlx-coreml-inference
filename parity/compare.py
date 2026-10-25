"""Raw differences between the 4 models (PyTorch / CoreML × fp32 / fp16), every pair, no tolerances.

Run (after parity.pred_pytorch and parity.pred_coreml):  python -m parity.compare_raw
Outputs: results/parity_raw.txt

Positions are sequence positions (the prompt occupies 0 .. n_prompt-1). Logit row i predicts the token at
position n_prompt + i; all logits are teacher-forced on the PyTorch fp32 sequence, so every model sees the same input.
"""
import io
from contextlib import redirect_stdout
from itertools import combinations

import numpy as np

import config

MODELS = ["torch_fp32", "torch_fp16", "coreml_fp32", "coreml_fp16"]
n_prompt = len(config.PARITY_PROMPT)


def load(model):
    framework, precision = model.split("_")
    tokens, logits = (config.PARITY_DIR / f"{framework}_{kind}_{precision}.npy" for kind in ("tokens", "logits"))
    assert tokens.exists() and logits.exists(), f"{model} outputs not found. Run parity.pred_pytorch and parity.pred_coreml first"
    return np.load(tokens), np.load(logits).astype(np.float32)


def compare(a, b):
    (tok_a, log_a), (tok_b, log_b) = a, b

    # Tokens: free-running greedy decode, so after the first divergence the sequences are unrelated
    diverged = np.flatnonzero(tok_a != tok_b)
    first = int(diverged[0]) if len(diverged) else None

    # Logits: same input for both, so every position is comparable
    err = np.abs(log_a - log_b)
    argmax_diff = np.flatnonzero(log_a.argmax(-1) != log_b.argmax(-1)) + n_prompt

    print(f"  tokens : first divergence at position {first}" if first is not None else "  tokens : identical")
    print(f"  logits : |diff| mean {err.mean():.3e} · std {err.std():.3e}")
    print(f"           argmax differs at positions {argmax_diff.tolist()}" if len(argmax_diff) else "           argmax identical at every position")


if __name__ == "__main__":
    outputs = {m: load(m) for m in MODELS}

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"Prompt {config.PARITY_PROMPT!r} ({n_prompt} tokens) · {len(outputs[MODELS[0]][0]) - n_prompt} generated tokens\n")
        for a, b in combinations(MODELS, 2):
            print(f"{a} vs {b}")
            compare(outputs[a], outputs[b])
            print()
    text = buf.getvalue()
    print(text, end="")

    out_path = config.ROOT / "results" / "parity_raw.txt"
    out_path.write_text(text)
    print(f"Saved → {out_path.relative_to(config.ROOT)}")
