"""MLX weights → PyTorch → torch.jit.trace → CoreML .mlpackage. Run: python -m export.export_coreml"""
import numpy as np
import torch
import coremltools as ct

import config
from export.model_torch import GPT, VOCAB_SIZE, load_mlx_weights


def export():
    model = GPT()
    model.eval()
    load_mlx_weights(model)
    print(f"Loaded weights from {config.WEIGHTS_PATH}")

    # Dummy input for tracing: (batch=1, seq_len=CONTEXT_LEN)
    # torch.jit.trace records ops for this exact shape;
    # RangeDim below tells CoreML the seq dimension is actually flexible.
    dummy = torch.zeros(1, config.CONTEXT_LEN, dtype=torch.long)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    print("JIT trace complete.")

    # Flexible sequence length: 1 token minimum, CONTEXT_LEN maximum
    seq_dim     = ct.RangeDim(lower_bound=1, upper_bound=config.CONTEXT_LEN, default=config.CONTEXT_LEN)
    input_shape = ct.Shape(shape=(1, seq_dim))

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="token_ids", shape=input_shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,   # ANE + GPU + CPU — runtime picks best
    )

    config.MLPACKAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(config.MLPACKAGE_PATH))
    print(f"CoreML model saved to {config.MLPACKAGE_PATH}")
    print(f"Input  : token_ids  — shape (1, T),              dtype int32")
    print(f"Output : logits     — shape (1, T, {VOCAB_SIZE}), dtype float32")


if __name__ == "__main__":
    export()
