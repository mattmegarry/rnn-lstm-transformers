#%%
import random
import torch
import math
import matplotlib.pyplot as plt

from char_dataset import CharDataset
from model import DecoderModel

torch.manual_seed(42)
max_seq_len = 26
epochs = 10

dataset = CharDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
model = DecoderModel(max_seq_len, 29, 7)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
  for idx, batch in enumerate(dataloader):

    sos = torch.tensor([2])
    eos = torch.tensor([1])

    x = batch[0]
    x = torch.cat([sos, x])
    y = torch.cat([x[1:], eos])

    probabilities = model(x)
    loss = torch.nn.functional.cross_entropy(probabilities, y)
    if idx % 1000 == 0: print("Loss:", loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#%%