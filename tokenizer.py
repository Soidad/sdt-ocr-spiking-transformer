import string

class SimpleTokenizer:
    def __init__(self):
        chars = string.ascii_lowercase + string.digits + " .,;:!?'-àâäéèêëîïôöùûüç"
        self.blank_token = 0

        self.char2id = {c: i+1 for i, c in enumerate(chars)}
        self.id2char = {i: c for c, i in self.char2id.items()}

        self.vocab_size = len(self.char2id) + 1  # + blank

    def encode(self, text):
        text = text.lower()
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids):
        return "".join(
            [self.id2char[i] for i in ids if i in self.id2char]
        )
