"""Shared settings for the Shakespeare GPT pipeline (train → export → bench → parity)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Model ─────────────────────────────────────────────────────────────────────
CONTEXT_LEN = 256
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 4
D_FF        = 512
DROPOUT     = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR         = 3e-4
NUM_ITERS  = 5000
EVAL_EVERY = 500
EVAL_ITERS = 50

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_URL       = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH      = ROOT / "data" / "shakespeare.txt"
WEIGHTS_PATH   = ROOT / "weights" / "gpt_shakespeare.npz"
MLPACKAGE_PATH = ROOT / "artifacts" / "gpt_shakespeare.mlpackage"
MLPACKAGE_FP32_PATH = ROOT / "artifacts" / "gpt_shakespeare_fp32.mlpackage"
MLPACKAGE_PATHS = {"fp16": MLPACKAGE_PATH, "fp32": MLPACKAGE_FP32_PATH}  # keyed by compute precision
PARITY_DIR     = ROOT / "artifacts" / "parity"

# ── Parity check ──────────────────────────────────────────────────────────────
PARITY_PROMPT = "ADAM: Good Day"
