import torch

class TinyStoriesDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer):
    f = open('TinyStories-10k.txt', 'r')
    self.story_lines = f.read().split('\n')
    self.tokenizer = tokenizer
    f.close()

  def __len__(self):
    return len(self.story_lines)

  def __getitem__(self, idx):
    story_line = self.story_lines[idx]
    return torch.tensor(self.tokenizer.encode(story_line))