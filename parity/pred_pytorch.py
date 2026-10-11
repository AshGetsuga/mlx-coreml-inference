"""Greedy-decode the parity prompt with PyTorch (fp32, CPU). Run: python -m parity.pred_pytorch"""
import numpy as np
import torch

import config
from export.model_torch import GPT, load_mlx_weights
from tokenizer import CharTokenizer

tok = CharTokenizer()

# ── Load model + weights (trained in MLX) ─────────────────────────────────────
model = GPT()
load_mlx_weights(model)
model.eval()

tokens = tok.encode(config.PARITY_PROMPT)

# Unpadded, growing context: one forward pass per new token
while len(tokens) != config.CONTEXT_LEN:
    x = torch.tensor(tokens, dtype=torch.long).reshape(1, len(tokens))
    with torch.no_grad():
        y = model(x)                                   # [B, Len, Vocab]
    tokens.append(int(torch.argmax(y[0, -1, :])))      # greedy pick at the last position

print(f"Output: {tok.decode(tokens)}")

config.PARITY_DIR.mkdir(parents=True, exist_ok=True)
np.save(config.PARITY_DIR / "torch_tokens.npy", np.array(tokens))
