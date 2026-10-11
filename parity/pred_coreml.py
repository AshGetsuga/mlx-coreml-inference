"""Greedy-decode the parity prompt with the CoreML model. Run: python -m parity.pred_coreml"""
import numpy as np
import coremltools as ct

import config
from tokenizer import CharTokenizer

tok   = CharTokenizer()
model = ct.models.MLModel(str(config.MLPACKAGE_PATH))

tokens = tok.encode(config.PARITY_PROMPT)

while len(tokens) != config.CONTEXT_LEN:
    seq_len = len(tokens)

    # The traced graph expects CONTEXT_LEN tokens — right-pad with zeros
    x = np.zeros((1, config.CONTEXT_LEN), dtype=np.int32)
    x[0, :seq_len] = tokens

    out = model.predict({"token_ids": x})

    # Read logits at the last real position, not the last padded position
    logits = out["logits"][0, seq_len - 1, :]   # (VOCAB_SIZE,)
    tokens.append(int(np.argmax(logits)))

print(f"Output: {tok.decode(tokens)}")

config.PARITY_DIR.mkdir(parents=True, exist_ok=True)
np.save(config.PARITY_DIR / "coreml_tokens.npy", np.array(tokens))
