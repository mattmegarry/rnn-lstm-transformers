#%%
import torch
import matplotlib.pyplot as plt

from sentencepeice_tokenizer import SentencePieceTokenizer
from tiny_stories_dataset import TinyStoriesDataset, pad
from model import DecoderModel

torch.manual_seed(42)
max_seq_len = 2200
epochs = 100

tokenizer = SentencePieceTokenizer()
dataset = TinyStoriesDataset(tokenizer)
vocab_len = tokenizer.get_vocab_size()
batch_size = 32

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad)

model = DecoderModel(max_seq_len, vocab_len, 32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
  print("Epoch:", epoch)
  for idx, batch in enumerate(dataloader):
    """ if epoch == 1:
      print(idx)
      print(batch.shape)
      x = batch[idx]
      print(tokenizer.decode(x.tolist())) """
    #if idx == batch_size: break
    sos = torch.tensor([1])
    eos = torch.tensor([2])

    x = batch
    y = batch

    print("y:")
    print(y.shape)

    """ x = torch.cat([sos, x])
    y = torch.cat([x[1:], eos]) """

    probabilities = model(x)
    print("probabilities:")
    print(probabilities.shape)
    loss = torch.nn.functional.cross_entropy(probabilities.view(-1, 2000), y.view(-1))
    if idx % 1000 == 0: 
      print("Loss:", loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#%%
def generate_from_string(string):
      x = torch.cat([sos, torch.tensor(tokenizer.encode(string))])
      while True:
        p = model(x)
        p = torch.nn.functional.softmax(p, dim=1)
        p = torch.argmax(p, dim=1)
        x = torch.cat([x, p[-1].unsqueeze(0)])
        if p[-1] == 2 or len(p.tolist()) == 17: break
      print("Generate:", tokenizer.decode(x.tolist()))

generate_from_string("The man")
generate_from_string("The woman")
generate_from_string("In the beginning")
generate_from_string("Once upon a time")
# %%
"""
100 Epochs
Generate: The man said, "I know, "I know".
Generate: The woman smiled and said, "I know," said.
Generate: In the beginningedde.
Generate: Once upon a time there was a time there was a time there was a time

1 Epoch
Generate: The man...............
Generate: The woman...............
Generate: In the beginning..........
Generate: Once upon a time............

"""