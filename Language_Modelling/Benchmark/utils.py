import urllib
import os

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "../shakespeare.txt"


class Tokenizer:

    def __init__(self, PATH: str = DATA_PATH):
        # Check if Path exists
        assert os.path.exists(PATH), "INVALID DATASET PATH!"

        # Open Dataset
        with open(DATA_PATH) as f:
            self.text = f.read()
        
        # Create Decoder - encoder
        self.chars = sorted(set(self.text))
        # Map 1: Char -> idx
        self.stoi = {c : i for i, c in enumerate(self.chars)}
        # Map 2: idx - > Char
        self.iots = {i : c for i, c in enumerate(self.chars)}

    def encode(self, string: str):
        """
        Encode Input String
        """
        return [self.stoi[c] for c in string]
    
    def decode(self, tokens: list):
        """
        Decode Input Tokens
        """
        return [self.iots[i] for i in tokens]

    def vocab_size(self):
        """
        Return Vocab Size
        """
        return len(self.chars)

        


