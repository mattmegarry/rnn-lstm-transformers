class CharTokenizer:
  def __init__(self, words):
    self.vocab = ['<pad>', '<eos>', '<sos>'] + sorted(set(''.join(words)))
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
names = ['john', 'jane', 'doe']
tokenizer = CharTokenizer(names)
print(tokenizer.vocab_size) 
encoded = tokenizer.encode('john')
decoded = tokenizer.decode(encoded)
print(encoded)
print(decoded)
"""