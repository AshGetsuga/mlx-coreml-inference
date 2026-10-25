"""PyTorch mirror of train/model_mlx.py, used for JIT tracing → CoreML. Parameter names match the MLX model."""
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

import config
from config import CONTEXT_LEN, D_MODEL, N_HEADS, N_LAYERS, D_FF
from tokenizer import CharTokenizer

VOCAB_SIZE = CharTokenizer().vocab_size

# Attention -- Multi-Head

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads

        assert self.d_model % self.n_heads == 0, "Model Dim must be divisible by Num Heads"
        self.d_head = self.d_model // self.n_heads

        self.Wq = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.Wo = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, N, k_dim = x.shape  # [B, N, D]
        assert k_dim == self.d_model, "Model and output Dims do not match"

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        # Step 0: Split into heads → [B, H, N, d_head]
        Q = Q.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Step 1: Attention weights → [B, H, N, N]
        attention_weights = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Step 2: Causal mask + softmax 
        attention_weights = torch.softmax(attention_weights + mask, dim=-1)

        # Step 3: Weighted values → [B, H, N, d_head]
        attention_output = attention_weights @ V

        # Step 4: Merge heads + output projection → [B, N, D]
        attention_output = self.Wo(attention_output.transpose(1, 2).reshape(B, N, k_dim))

        return attention_output
    
# Transformer Block -- Only Decoder

class FFN(nn.Module):
    """
    Custom Feedforward Layer
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Dim -> Expand -> Dim
        self.fc1 = nn.Linear(in_features = d_model, out_features = d_ff)
        self.fc2 = nn.Linear(in_features = d_ff, out_features = d_model)

    def forward(self, x):
        """
        Forward Pass
        """
        # Simple Forward Pass with ReLU
        return self.fc2(F.relu(self.fc1(x)))

class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        self.d_model = d_model
        # Layer Norm 1
        self.ln1 = nn.LayerNorm(normalized_shape = d_model)
        # The Attention Layer
        self.attn = MultiHeadAttention(d_model = d_model, n_heads = n_heads)
        # FFN
        self.ffn = FFN(d_model = d_model, d_ff = d_ff)
        # Layer Norm 2
        self.ln2 = nn.LayerNorm(normalized_shape = d_model)
    
    def forward(self, x, mask):
        """
        """
        # Verify shape
        assert x.shape[-1] == self.d_model, "Input shape doesnt match Model"

        x = x + self.attn(self.ln1(x), mask) # pre-norm -> attn -> residual

        x = x + self.ffn(self.ln2(x)) # pre-norm -> ffn -> residual

        return x


# GPT 2
class GPT(nn.Module):
    def __init__(self):
        """
        GPT: Tokenizer -> Decoder Layers -> LM Head
        """
        super().__init__()
        # Token and Pos Embedding
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, D_MODEL)
        
        # Decoder Blocks
        self.blocks = nn.ModuleList([DecoderBlock(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF) for _ in range(N_LAYERS)])
        
        # Layer Norm -> LM Head
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias = False)
    
    def forward(self, x):
        B, T = x.shape
        # Positional Encoding
        positions = torch.arange(T, device=x.device)
        # Creating deterministic masked attention mask
        mask = torch.full((T, T), fill_value=-1e4)
        # Match the weights' dtype: an fp32 mask would promote fp16 attention scores (and softmax) back to fp32
        mask = torch.triu(mask, diagonal = 1).to(device=x.device, dtype=self.tok_emb.weight.dtype)

        x = self.tok_emb(x) + self.pos_emb(positions)

        # Forward Pass
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

        

def load_mlx_weights(model: GPT, npz_path=config.WEIGHTS_PATH) -> None:
    """Load MLX .npz weights by name. MLX Linear stores (out, in), same as PyTorch, so no transposes."""
    arrays = np.load(npz_path, allow_pickle=True)
    sd     = model.state_dict()
    missing = [k for k in sd if k not in arrays]
    if missing:
        raise KeyError(f"Weights missing from {npz_path}: {missing}")
    model.load_state_dict({k: torch.from_numpy(np.array(arrays[k])) for k in sd})
