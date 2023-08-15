import torch
from char_tokenizer import CharTokenizer

class CharDataset(torch.utils.data.Dataset):
  def __init__(self):
    f = open('names.txt', 'r')
    self.names = f.read().split('\n')
    self.tokenizer = CharTokenizer()
    f.close()

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name = self.names[idx]
    return torch.tensor(self.tokenizer.encode(name))