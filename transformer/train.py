#%%
import random
import torch
import math
import matplotlib.pyplot as plt

from sentencepeice_tokenizer import SentencePieceTokenizer
from tiny_stories_dataset import TinyStoriesDataset
from model import DecoderModel

torch.manual_seed(42)
max_seq_len = 2000
epochs = 10

tokenizer = SentencePieceTokenizer()
dataset = TinyStoriesDataset(tokenizer)
vocab_len = tokenizer.get_vocab_size()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
model = DecoderModel(max_seq_len, vocab_len, 128)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
  for idx, batch in enumerate(dataloader):

    sos = torch.tensor([1])
    eos = torch.tensor([2])

    x = batch[0]

    # Hack to make it work with the current dataset
    if x.numel() == 0: break

    x = torch.cat([sos, x])
    y = torch.cat([x[1:], eos])

    probabilities = model(x)
    loss = torch.nn.functional.cross_entropy(probabilities, y)
    if idx % 1000 == 0: print("Loss:", loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#%%