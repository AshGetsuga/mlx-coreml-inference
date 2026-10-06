"""
Cornell Movie Dialogues — Coherence vs Random batching experiment.

Both modes train for the same number of gradient steps so the comparison is fair
regardless of how many full batches each movie can contribute.

Usage:
    python train_cornell.py --mode coherent   # same movie per batch
    python train_cornell.py --mode random     # shuffled across all movies
"""

import os, random, math, argparse, time, itertools
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import tiktoken

# ── Config ────────────────────────────────────────────────────────────────────
CONTEXT_LEN  = 256
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
D_FF         = 512
BATCH_SIZE   = 32
NUM_STEPS    = 3000
EVAL_EVERY   = 300
EVAL_ITERS   = 50       # val batches to average for each evaluation
LR           = 3e-4
VAL_FRACTION = 0.15
DATA_ROOT    = os.path.join(os.path.dirname(__file__), "../LSTMvsTransformer/archive")
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

enc        = tiktoken.get_encoding("r50k_base")
VOCAB_SIZE = enc.n_vocab  # 50257

# ── Data loading ──────────────────────────────────────────────────────────────

def load_cornell(root: str) -> dict:
    """Returns {movie_title: concatenated_dialogue_text}."""
    movies = {}
    with open(os.path.join(root, "movie_titles_metadata.txt"), encoding="latin-1") as f:
        for line in f:
            parts = line.split("+++$+++")
            if len(parts) >= 2:
                movies[parts[0].strip()] = parts[1].strip()

    dialogues = defaultdict(str)
    with open(os.path.join(root, "movie_lines.txt"), encoding="latin-1") as f:
        for line in f:
            parts = line.split("+++$+++")
            if len(parts) < 5:
                continue
            tag      = parts[-3].strip()
            speaker  = parts[-2].strip()
            dialogue = parts[-1].strip()
            if tag in movies:
                dialogues[movies[tag]] += f"{speaker} : {dialogue}\n"

    return dict(dialogues)


def make_chunks(token_ids: list, context_len: int) -> list:
    """Non-overlapping windows of length context_len+1."""
    chunks = []
    for i in range(0, len(token_ids) - context_len, context_len):
        chunks.append(torch.tensor(token_ids[i : i + context_len + 1], dtype=torch.long))
    return chunks

# ── Datasets ──────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    def __init__(self, chunks: list):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        return c[:-1], c[1:]   # x, y — each shape (CONTEXT_LEN,)


class MovieBatchSampler(Sampler):
    """
    Yields batches where every index belongs to the same movie.
    Reshuffles movie order and within-movie order on each __iter__ call
    so cycling over the DataLoader gives a fresh shuffle each pass.
    """
    def __init__(self, movie_index_groups: list, batch_size: int):
        self.groups     = movie_index_groups
        self.batch_size = batch_size

    def _build(self):
        batches = []
        for group in self.groups:
            indices = group.copy()
            random.shuffle(indices)
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batches.append(indices[i : i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._build())

    def __len__(self):
        return sum(len(g) // self.batch_size for g in self.groups)

# ── Model ─────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask   = torch.triu(torch.full((N, N), -1e4), diagonal=1).to(x.device)
        attn   = torch.softmax(scores + mask, dim=-1)
        return self.Wo((attn @ V).transpose(1, 2).reshape(B, N, -1))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, D_MODEL)
        self.blocks  = nn.ModuleList([DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(D_MODEL)
        self.head    = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        x    = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(model, val_loader, device, n_iters: int) -> tuple:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    val_iter = iter(val_loader)
    for _ in range(n_iters):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y    = x.to(device), y.to(device)
        B, T, V = (logits := model(x)).shape
        loss    = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        total_loss   += loss.item() * B * T
        total_tokens += B * T
    model.train()
    mean = total_loss / total_tokens
    return mean, math.exp(mean)

# ── Generate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt: str, max_new: int = 200, temperature: float = 0.8, device="cpu") -> str:
    model.eval()
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx     = ids[:, -CONTEXT_LEN:]
        logits  = model(ctx)[:, -1, :] / temperature
        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)
    return enc.decode(ids[0].tolist())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["coherent", "random"], default="coherent")
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Mode   : {args.mode}\n")
    print("Loading Cornell data...")

    dialogues  = load_cornell(DATA_ROOT)
    token_map  = {title: enc.encode(text) for title, text in dialogues.items()}

    movie_chunks = {
        title: make_chunks(ids, CONTEXT_LEN)
        for title, ids in token_map.items()
        if len(ids) > CONTEXT_LEN
    }

    all_titles = sorted(movie_chunks.keys())
    rng = random.Random(42)
    rng.shuffle(all_titles)
    n_val        = max(1, int(len(all_titles) * VAL_FRACTION))
    val_titles   = all_titles[:n_val]
    train_titles = all_titles[n_val:]

    print(f"Movies total  : {len(all_titles)}")
    print(f"Train movies  : {len(train_titles)}")
    print(f"Val movies    : {len(val_titles)}")

    train_chunks = [c for t in train_titles for c in movie_chunks[t]]
    val_chunks   = [c for t in val_titles   for c in movie_chunks[t]]

    print(f"Train chunks  : {len(train_chunks):,}")
    print(f"Val chunks    : {len(val_chunks):,}\n")

    train_dataset = ChunkDataset(train_chunks)
    val_dataset   = ChunkDataset(val_chunks)

    if args.mode == "coherent":
        offset, groups = 0, []
        for title in train_titles:
            n = len(movie_chunks[title])
            groups.append(list(range(offset, offset + n)))
            offset += n
        train_loader = DataLoader(train_dataset, batch_sampler=MovieBatchSampler(groups, BATCH_SIZE))
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = GPT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}\n")

    header = f"{'Step':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'ms/step':>8}"
    print(header)
    print("─" * len(header))

    best_val_loss = float("inf")
    checkpoint    = f"cornell_{args.mode}_best.pt"

    train_iter     = itertools.cycle(train_loader)
    running_loss   = 0.0
    running_tokens = 0
    t0             = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss   += loss.item() * B * T
        running_tokens += B * T

        if step % EVAL_EVERY == 0 or step == 1:
            tr_loss = running_loss / running_tokens
            tr_ppl  = math.exp(tr_loss)
            va_loss, va_ppl = estimate_val_loss(model, val_loader, DEVICE, EVAL_ITERS)
            elapsed_ms = (time.perf_counter() - t0) / (EVAL_EVERY if step > 1 else 1) * 1000

            print(f"{step:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}  {elapsed_ms:>6.1f}ms")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save(model.state_dict(), checkpoint)

            running_loss = running_tokens = 0
            t0 = time.perf_counter()
            model.train()

    print(f"\nBest val loss : {best_val_loss:.4f}  (PPL {math.exp(best_val_loss):.2f})")
    print(f"Saved         : {checkpoint}\n")

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    prompt = "CAMERON : Hey, where are you going?\nBIANCA :"
    print(f"Prompt:\n{prompt}\n")
    print("Generated:")
    print(generate(model, prompt, max_new=200, temperature=0.8, device=DEVICE))


if __name__ == "__main__":
    main()
