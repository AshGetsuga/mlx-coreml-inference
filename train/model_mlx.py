"""Pre-norm decoder-only GPT in MLX. Parameter names match export/model_torch.py so weights load by name."""
import mlx.core as mx
import mlx.nn as nn

import config


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo      = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=config.DROPOUT)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.Wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.Wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.Wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # (B, n_heads, T, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)

        # Causal mask: large negative upper triangle so softmax zeroes future positions
        mask   = mx.triu(mx.full((T, T), -1e9), k=1)
        scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out     = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.dropout(self.Wo(out))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=config.DROPOUT)

    def __call__(self, x):
        return self.dropout(self.fc2(nn.relu(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, d_ff)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))  # pre-norm → attention → residual
        x = x + self.ffn(self.ln2(x))   # pre-norm → FFN      → residual
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab_size, config.D_MODEL)
        self.pos_emb  = nn.Embedding(config.CONTEXT_LEN, config.D_MODEL)
        self.blocks   = [DecoderBlock(config.D_MODEL, config.N_HEADS, config.D_FF) for _ in range(config.N_LAYERS)]
        self.ln_f     = nn.LayerNorm(config.D_MODEL)
        self.head     = nn.Linear(config.D_MODEL, vocab_size, bias=False)
        self.drop_emb = nn.Dropout(p=config.DROPOUT)

    def __call__(self, x):
        B, T = x.shape
        x    = self.drop_emb(self.tok_emb(x) + self.pos_emb(mx.arange(T)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))                         # (B, T, VOCAB_SIZE)
