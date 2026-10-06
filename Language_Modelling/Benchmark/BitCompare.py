import numpy as np

# Load the 2 outputs
torch_out  = np.load("Torch_Values.npy")
coreml_out = np.load("CoreML_Values.npy")
# Compare all Tokens
print(f"Match Status: ", np.array_equal(torch_out, coreml_out))         # True = identical sequence
# In case of not matching, where it goes wrong
print(f"Token Mismatch (if valid): ",np.where(torch_out != coreml_out))              # first divergence point if not
