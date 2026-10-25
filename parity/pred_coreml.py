"""Greedy-decode the parity prompt with the fp16 and fp32 CoreML models. Run (after parity.pred_pytorch): python -m parity.pred_coreml"""
import numpy as np
import coremltools as ct

import config
from tokenizer import CharTokenizer

tok = CharTokenizer()

# Each precision runs on the compute units it is benchmarked on (see ML_Bench):
# fp16 on the Neural Engine (fastest config), fp32 on ALL (the ANE is fp16-only, so this lands on GPU/CPU)
COMPUTE_UNITS = {"fp16": ct.ComputeUnit.CPU_AND_NE, "fp32": ct.ComputeUnit.ALL}


def forward(model, tokens):
    """Logits for the real positions only: [len(tokens), VOCAB_SIZE].

    Always pads to CONTEXT_LEN: with the flexible (RangeDim) input, the Neural Engine only runs at the default
    length (256); any other length silently falls back to the CPU. Padded positions sit after the real tokens,
    so the causal mask keeps them from affecting real positions; their logits are dropped.
    """
    x = np.zeros((1, config.CONTEXT_LEN), dtype=np.int32)
    x[0, :len(tokens)] = tokens
    return model.predict({"token_ids": x})["logits"][0, :len(tokens)]


def greedy_decode(model):
    tokens = tok.encode(config.PARITY_PROMPT)
    while len(tokens) != config.CONTEXT_LEN:
        tokens.append(int(np.argmax(forward(model, tokens)[-1])))   # greedy pick at the last real position
    return tokens


if __name__ == "__main__":
    config.PARITY_DIR.mkdir(parents=True, exist_ok=True)
    n_prompt = len(config.PARITY_PROMPT)

    # Logits are compared on the SAME input as PyTorch (its decoded sequence, minus the last token), so a
    # divergence in greedy tokens doesn't make the logit comparison meaningless
    ref_tokens = np.load(config.PARITY_DIR / "torch_tokens_fp32.npy")[:-1].tolist()

    for precision, units in COMPUTE_UNITS.items():
        model = ct.models.MLModel(str(config.MLPACKAGE_PATHS[precision]), compute_units=units)
        tokens = greedy_decode(model)

        print(f"── {precision} ({units.name})\n{tok.decode(tokens)}\n")

        np.save(config.PARITY_DIR / f"coreml_tokens_{precision}.npy", np.array(tokens))
        # Rows from the last prompt position onward: the logits that predict each generated token
        np.save(config.PARITY_DIR / f"coreml_logits_{precision}.npy", forward(model, ref_tokens)[n_prompt - 1:])
