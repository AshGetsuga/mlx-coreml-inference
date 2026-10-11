"""Character-level tokenizer for Tiny Shakespeare (65-char vocab), shared by every script."""
import urllib.request

import config


def load_text() -> str:
    """Return the corpus, downloading it on first use."""
    if not config.DATA_PATH.exists():
        print("Downloading Shakespeare dataset...")
        config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.DATA_URL, config.DATA_PATH)
    return config.DATA_PATH.read_text()


class CharTokenizer:
    def __init__(self, text: str | None = None):
        text = text if text is not None else load_text()
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)
