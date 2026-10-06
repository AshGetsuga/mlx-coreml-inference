import torch.nn as nn
import torch.nn.functional as F
import torch
import urllib


class Decoder(nn.Module):

    """
    Components:
        input and output dim
        ffn dim in case of expansion
        context len
        Cache K and V Matrices for attention
        FFN Layers
            Q, K and V FFN
            W0 FFN
            Feature Combination FFN
        Layer Norm for Attn and FFN
        Attention Function
    """
    def __init__(self, input_dim: int, ffn_dim: int, context_len: int, n_heads: int):

        # Get parent class features
        super().__init__()

        # Instance Variables required for processing
        self.input_dim, self.ffn_dim, self.context_len= input_dim, ffn_dim, context_len

        assert self.input_dim % n_heads == 0, "Dim not divisible by num of heads"
        self.head_dim = int(self.input_dim / n_heads)
        self.n_heads = n_heads

        # Cache Variables
        self.K_cache, self.V_cache = None, None
        self.cache = False

        # Q, K and V projection
        self.K = nn.Linear(in_features = input_dim, out_features = input_dim, bias = False)
        self.V = nn.Linear(in_features = input_dim, out_features = input_dim, bias = False)
        self.Q = nn.Linear(in_features = input_dim, out_features = input_dim, bias = False)

        # Output Projection for Multi-Head
        self.W0 = nn.Linear(in_features = input_dim, out_features = input_dim)

        # Layer Norm
        self.layer_norm_attn = nn.LayerNorm(self.input_dim)
        self.layer_norm_ffn = nn.LayerNorm(self.input_dim)

        # Feedforward with expansion
        self.FFN = nn.Sequential(
            nn.Linear(in_features = self.input_dim, out_features = self.ffn_dim),
            nn.ReLU(),
            nn.Linear(in_features = self.ffn_dim, out_features = self.input_dim),
        )

    def clear_cache(self):
        """
        Helper Function to clear cache
        """
        self.K_cache, self.V_cache = None, None
        self.cache = False
        print(f"Cleared Cache")
        return

    def inference(self, input):
        """
        Cache-ing for efficiency
        """

        Q, K, V = self.Q(input), self.K(input), self.V(input)
        # Add cache for K and V
        if self.cache:
            K, V = torch.cat([self.K_cache, K], dim = 1), torch.cat([self.V_cache, V], dim = 1)


        # Check context length, remove first token
        if K.shape[1] >= self.context_len:
            K, V = K[:, 1:, :], V[:, 1:, :]

        output = self.MultiHeadAttention(Q, K, V)

        # Update cache
        self.K_cache, self.V_cache = K, V
        self.cache = True

        # Return output
        return output

    

    def forward(self, input):
        """
        Forward Pass
        """
        # Check if eval mode
        if not self.training:
            # Residual for attention
            output =  input + self.inference(self.layer_norm_attn(input))
            
        # Default train behavior
        else:
            # Create Q, K and V matrix
            normed = self.layer_norm_attn(input)
            Q, K, V = self.Q(normed), self.K(normed), self.V(normed)
            # Run Multi-Head Attention
            # Residual for attention
            output = input + self.MultiHeadAttention(Q, K, V)
        
        # Residual for Feedforward layer
        output = output + self.FFN(self.layer_norm_ffn(output))

        return output
            
            
    
    def MultiHeadAttention(self, Q, K, V):
        """
        MultiHeadAttention Calculation
        """
        # Step 0: Get Dims
        B, Q_N, _ = Q.shape
        _, K_N, _ = K.shape
        _, V_N, _ = V.shape
        
        # Step 1: Split into heads
        Q, K, V = Q.reshape(B, Q_N, self.n_heads, self.head_dim).transpose(1, 2), K.reshape(B, K_N, self.n_heads, self.head_dim).transpose(1, 2), V.reshape(B, V_N, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 2: Calculate Attention Score
        Attention_Score = (Q @ K.transpose(-2, -1)) / (self.head_dim)**0.5 # [B, H, N, N]

        # Step 3: Mask Upper Triangular Matrix 
        # In Case of Q_N = 1 (happens in KV cache-ing), no masking required
        if Q_N != 1:
            mask = torch.triu(torch.full(size = (Q_N, K_N), fill_value = -1e4), diagonal = 1).to(Q.device)
            Attention_Score += mask

        # Step 3: Scale attention score and flatten with softmax
        Attention_Score = Attention_Score.softmax(dim = -1) # [B, H, N, N]


        # Step 4: Reweigh V matric and get embeddings for each N
        output_tensor = Attention_Score @ V # [B, H, N, head_dim]

        # Step 5: Reshape and pass though output projector
        output_tensor = self.W0(output_tensor.transpose(1, 2).reshape(B, Q_N, self.input_dim))

        return output_tensor

