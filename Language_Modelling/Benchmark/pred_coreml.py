# Script for Bit Exactness checking — CoreML side
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../graph"))

import numpy as np
import coremltools as ct
from utils import Tokenizer

# Model Parameters
MODEL_PATH  = "../graph/gpt_shakespeare.mlpackage"
CONTEXT_LEN = 256

# ── Load CoreML model ─────────────────────────────────────────────────────────
model = ct.models.MLModel(MODEL_PATH)

# ── Load tokens ───────────────────────────────────────────────────────────────
tokens = Tokenizer().encode("ADAM: Good Day")

while len(tokens) != CONTEXT_LEN:

    seq_len = len(tokens)

    # CoreML requires exactly CONTEXT_LEN tokens — right-pad with zeros
    x = np.zeros((1, CONTEXT_LEN), dtype=np.int32)
    x[0, :seq_len] = tokens

    out = model.predict({"token_ids": x})

    # Read logits at the last real position, not the last padded position
    logits = out["logits"][0, seq_len - 1, :]   # (VOCAB_SIZE,)
    token  = int(np.argmax(logits))

    tokens.append(token)

# Printing Prediction
print(f'Output: {"".join(Tokenizer().decode(tokens))}')

try:
    np.save("CoreML_Values.npy", np.array(tokens))
except Exception as e:
    print(f"Error: {e}\n")
