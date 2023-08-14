class CharTokenizer:
  def __init__(self):
    f = open('names.txt', 'r')
    names = f.read().splitlines()
    self.vocab = ['<pad>', '<eos>', '<sos>'] + sorted(set(''.join(names)))
    self.stoi = {c:i for i, c in enumerate(self.vocab)}
    self.itos = {i:c for i, c in enumerate(self.vocab)}
    self.vocab_size = len(self.vocab)

  def encode(self, word):
    return [self.stoi[c] for c in word]

  def decode(self, tokens):
    return ''.join([self.itos[t] for t in tokens])
  
  def get_vocab(self):
    return self.vocab


# Just for testing...
"""
tokenizer = CharTokenizer()
print(tokenizer.vocab_size) 
encoded = tokenizer.encode('john')
decoded = tokenizer.decode(encoded)
print(encoded)
print(decoded)
"""
