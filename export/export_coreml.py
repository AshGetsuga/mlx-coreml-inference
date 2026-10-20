"""MLX weights → PyTorch → torch.jit.trace → CoreML .mlpackage. Run: python -m export.export_coreml [--precision fp16|fp32]"""
import argparse

import numpy as np
import torch
import coremltools as ct

import config
from export.model_torch import GPT, VOCAB_SIZE, load_mlx_weights

PRECISIONS = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}


def export(precision="fp16"):
    model = GPT()
    model.eval()
    load_mlx_weights(model)
    print(f"Loaded weights from {config.WEIGHTS_PATH}")

    # Dummy input for tracing: (batch=1, seq_len=CONTEXT_LEN)
    # torch.jit.trace records ops for this exact shape;
    # RangeDim below tells CoreML the seq dimension is actually flexible.
    dummy = torch.zeros(1, config.CONTEXT_LEN, dtype=torch.long)

    with torch.no_grad():
        _ = model(dummy)
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
        compute_precision=PRECISIONS[precision],  # internal op precision; I/O dtypes above are unaffected
    )

    out_path = config.MLPACKAGE_PATHS[precision]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"CoreML model ({precision}) saved to {out_path}")
    print(f"Input  : token_ids  — shape (1, T),              dtype int32")
    print(f"Output : logits     — shape (1, T, {VOCAB_SIZE}), dtype float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=PRECISIONS, default="fp16")
    export(parser.parse_args().precision)
