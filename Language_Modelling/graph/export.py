import numpy as np
import torch
import coremltools as ct
from GPT import GPT

# ── Hyperparameters (must match train.py) ─────────────────────────────────────
CONTEXT_LEN  = 256
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
D_FF         = 512
DROPOUT      = 0.2
WEIGHTS_PATH = "../gpt_shakespeare.npz"
DATA_PATH    = "../shakespeare.txt"
OUTPUT_PATH  = "gpt_shakespeare.mlpackage"

# ── Vocab size (derived from corpus, must match train.py) ─────────────────────
with open(DATA_PATH) as f:
    text = f.read()
VOCAB_SIZE = len(sorted(set(text)))




# ── Load MLX weights into PyTorch model ───────────────────────────────────────

def load_weights(model: "GPT", npz_path: str) -> None:
    """
    Maps MLX parameter names → PyTorch state_dict keys and loads weights.

    MLX key examples:
        tok_emb.weight               → tok_emb.weight
        pos_emb.weight               → pos_emb.weight
        blocks.0.attn.Wq.weight      → blocks.0.attn.Wq.weight
        blocks.0.ffn.fc1.weight      → blocks.0.ffn.fc1.weight
        ln_f.weight / ln_f.bias      → ln_f.weight / ln_f.bias
        head.weight                  → head.weight

    MLX Linear stores weights as (out, in) — same as PyTorch, no transpose needed.
    MLX LayerNorm stores weight + bias — same as PyTorch.
    """
    arrays   = np.load(npz_path, allow_pickle=True)
    sd_torch = model.state_dict()
    sd_new   = {}

    for k in sd_torch:
        if k not in arrays:
            raise KeyError(f"Weight '{k}' expected in .npz but not found.\n"
                           f"Available keys: {list(arrays.keys())}")
        sd_new[k] = torch.from_numpy(np.array(arrays[k]))

    model.load_state_dict(sd_new)
    print(f"Loaded {len(sd_new)} weight tensors from {npz_path}")


# ── JIT trace + CoreML export ─────────────────────────────────────────────────

def export():
    model = GPT()                  # TODO: defined above
    model.eval()
    load_weights(model, WEIGHTS_PATH)

    # Dummy input for tracing: (batch=1, seq_len=CONTEXT_LEN)
    # torch.jit.trace records ops for this exact shape;
    # RangeDim below tells CoreML the seq dimension is actually flexible.
    dummy = torch.zeros(1, CONTEXT_LEN, dtype=torch.long)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    print("JIT trace complete.")

    # Flexible sequence length: 1 token minimum, CONTEXT_LEN maximum
    seq_dim   = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LEN, default=CONTEXT_LEN)
    input_shape = ct.Shape(shape=(1, seq_dim))

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="token_ids", shape=input_shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,   # ANE + GPU + CPU — runtime picks best
    )

    mlmodel.save(OUTPUT_PATH)
    print(f"CoreML model saved to {OUTPUT_PATH}")
    print(f"Input  : token_ids  — shape (1, T),              dtype int32")
    print(f"Output : logits     — shape (1, T, {VOCAB_SIZE}), dtype float32")


if __name__ == "__main__":
    export()
