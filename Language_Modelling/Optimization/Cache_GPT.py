from Cache_Decoder import *


# GPT is basically a collection of decoder blocks followed by a language head

class GPT(nn.Module):

    def __init__(self, input_dim: int, vocab_size: int, context_len: int, ffn_dim: int, n_heads: int, blocks: int):
        # Inherit from parent
        super().__init__()

        # Tok Embedding
        self.tok_embedding = nn.Linear(in_features = vocab_size, out_features = input_dim)

        # Create Decoder Blocks
        self.decoder_blocks = nn.ModuleList(
            [Decoder(input_dim = input_dim, ffn_dim = ffn_dim, context_len = context_len, n_heads = n_heads) 
             for _ in range(blocks)]
        )

        # Language Head
        self.lm_head = nn.Linear(in_features = input_dim, out_features = vocab_size)
    
    def forward(self, x):
        """
        Simple forward pass
        """
        



        