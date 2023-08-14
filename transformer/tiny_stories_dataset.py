import torch

class CharDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer):
    f = open('TinyStories-train.txt', 'r')
    self.stories = f.read().split('\n')
    self.tokenizer = tokenizer
    f.close()

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name = self.names[idx]
    return torch.tensor(self.tokenizer.encode(name))