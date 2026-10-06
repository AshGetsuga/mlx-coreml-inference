# Script for Bit Exactness checking
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../graph"))

import torch
import numpy as np
from GPT import GPT
from utils import Tokenizer

# Model Parameters
WEIGHTS_PATH = "../gpt_shakespeare.npz"
CONTEXT_LEN  = 256
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE       = "cpu"

# ── Load model + weights ───────────────────────────────────────────────────────
model = GPT().to(DEVICE)
model.eval()

# Manual loading as trained on mlx
arrays = np.load(WEIGHTS_PATH, allow_pickle=True)
sd     = model.state_dict()
model.load_state_dict({k: torch.from_numpy(np.array(arrays[k])).to(DEVICE) for k in sd})

# Loading content
tokens = Tokenizer().encode("ADAM: Good Day")
x = torch.tensor(tokens, dtype = torch.long, device = DEVICE).reshape(1, len(tokens))

while len(tokens) != CONTEXT_LEN:
    
    with torch.no_grad(): # Single Forward Pass
        y = model(x) # Shape: [B, Len, Vocab]
    
    # Extract max over Vocab and last token
    prediction = torch.argmax(input = y, dim = -1)[:, -1]

    # Convert to Token
    token = prediction.numpy().item()

    # New Context
    tokens.append(token)
    x = torch.tensor(tokens, dtype = torch.long, device = DEVICE).reshape(1, len(tokens))

# Printing Prediction
print(f'Output: {"".join(Tokenizer().decode(tokens))}')

try: 
    np.save("Torch_Values.npy", np.array(tokens))
except Exception as e:
    print(f"Error: {e} \n")
    

    

        
