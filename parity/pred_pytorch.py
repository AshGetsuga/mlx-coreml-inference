"""Greedy-decode the parity prompt with PyTorch fp32 (CPU, the reference) and fp16 (MPS). Run: python -m parity.pred_pytorch"""
import numpy as np
import torch

import config
from export.model_torch import GPT, load_mlx_weights
from tokenizer import CharTokenizer

tok = CharTokenizer()

# fp32 on CPU is the reference every other run is compared against.
# fp16 runs on MPS: PyTorch's CPU fp16 kernels are slow and not representative (see bench.bench_pytorch)
DEVICES = {"fp32": "cpu", "fp16": "mps"}


def load_model(precision):
    """GPT with the MLX-trained weights, at the given precision, on its device."""
    model = GPT()
    load_mlx_weights(model)
    if precision == "fp16":
        model = model.half()
    return model.to(DEVICES[precision]).eval()


def forward(model, tokens):
    """Logits [len(tokens), VOCAB_SIZE], returned as fp32 numpy so later diffs aren't computed in fp16."""
    device = next(model.parameters()).device
    x = torch.tensor(tokens, dtype=torch.long, device=device).reshape(1, len(tokens))
    with torch.no_grad():
        return model(x)[0].float().cpu().numpy()


def greedy_decode(model):
    tokens = tok.encode(config.PARITY_PROMPT)
    # Unpadded, growing context: one forward pass per new token
    while len(tokens) != config.CONTEXT_LEN:
        tokens.append(int(np.argmax(forward(model, tokens)[-1])))   # greedy pick at the last position
    return tokens


if __name__ == "__main__":
    config.PARITY_DIR.mkdir(parents=True, exist_ok=True)
    n_prompt = len(config.PARITY_PROMPT)

    ref_tokens = None
    for precision in DEVICES:   # fp32 first: its decoded sequence is the input every model's logits are taken on
        model = load_model(precision)
        tokens = greedy_decode(model)
        if ref_tokens is None:
            ref_tokens = tokens

        print(f"── {precision} ({DEVICES[precision]})\n{tok.decode(tokens)}\n")

        np.save(config.PARITY_DIR / f"torch_tokens_{precision}.npy", np.array(tokens))
        # Teacher-forced on the fp32 sequence (minus the last token), rows from the last prompt position onward
        np.save(config.PARITY_DIR / f"torch_logits_{precision}.npy", forward(model, ref_tokens[:-1])[n_prompt - 1:])
