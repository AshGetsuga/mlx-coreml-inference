"""Compare PyTorch vs CoreML greedy-decoded token sequences. Run: python -m parity.compare"""
import numpy as np

import config

torch_out  = np.load(config.PARITY_DIR / "torch_tokens.npy")
coreml_out = np.load(config.PARITY_DIR / "coreml_tokens.npy")

n_prompt = len(config.PARITY_PROMPT)
match    = np.array_equal(torch_out, coreml_out)
print(f"Greedy-decode parity : {match}  ({len(torch_out) - n_prompt} generated tokens)")
if not match:
    diff = np.where(torch_out != coreml_out)[0]
    print(f"First divergence at position {diff[0]} (generated token {diff[0] - n_prompt})")
