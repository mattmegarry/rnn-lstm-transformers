#%%
import torch
import matplotlib.pyplot as plt
import wandb

from sentencepeice_tokenizer import SentencePieceTokenizer
from tiny_stories_dataset import TinyStoriesDataset, pad
from model import DecoderModel

torch.manual_seed(42)

learning_rate = 0.001
max_seq_len = 2200
epochs = 100
batch_size = 32
tokenizer = SentencePieceTokenizer()
vocab_len = tokenizer.get_vocab_size()
dataset = TinyStoriesDataset(tokenizer)

wandb.init(
    project="tiny-stories-decoder",
    config={
    "learning_rate": learning_rate,
    "max_seq_len": max_seq_len,
    "vocab_len": vocab_len,
    "batch_size": batch_size,
    "dataset": dataset.get_data_filename(),
    "epochs": epochs,
    "architecture": "decoder-only"
    }
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad)
model = DecoderModel(max_seq_len, vocab_len, 32)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  print("Epoch:", epoch)
  for idx, batch in enumerate(dataloader):
    sos = torch.full((batch.shape[0], 1), 1)
    eos = torch.full((batch.shape[0], 1), 2)

    x = batch
    x = torch.cat([sos, x], dim=1)
    y = torch.cat([x[:, 1:], eos], dim=1)

    probabilities = model(x)
    loss = torch.nn.functional.cross_entropy(probabilities.view(-1, vocab_len), y.view(-1))
    wandb.log({"loss": loss})
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