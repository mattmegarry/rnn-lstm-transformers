#%%
import random
import torch
import math
import matplotlib.pyplot as plt

from char_dataset import CharDataset

torch.manual_seed(42)
max_seq_len = 26

ds = CharDataset()
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


class BesSimpleTransformer(torch.nn.Module):
  def __init__(self):
    super(BesSimpleTransformer, self).__init__()
    # Embedding part of the model
    self.embedding    = torch.nn.Embedding(29, 7)
    self.pos_emb      = self.get_pos_matrix()
    # Mask tensor trick
    self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
    # First decoder block
    self.layer_00_key = torch.nn.Linear(7, 11)
    self.layer_00_qry = torch.nn.Linear(7, 11)
    self.layer_00_val = torch.nn.Linear(7, 11)
    self.layer_00_ffw = torch.nn.Linear(11, 7)
    # Second decoder block
    self.layer_01_key = torch.nn.Linear(7, 11)
    self.layer_01_qry = torch.nn.Linear(7, 11)
    self.layer_01_val = torch.nn.Linear(7, 11)
    self.layer_01_ffw = torch.nn.Linear(11, 7)
    # Output of the model
    self.map_to_vocab = torch.nn.Linear(7, 29)

  def forward(self, x):
    
    # plot_attention_heatmap(self.embedding(torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28])))
    #print(x)
    #plot_attention_heatmap(self.embedding.weight)
    #print(self.embedding.__dict__)
    #print(self.embedding_initial.weight - self.embedding.weight)
    emb = self.embedding(x)
    print(emb)
    """ print(emb)
    print(emb.shape)
    plot_attention_heatmap(emb) """
    pos = self.pos_emb[0:x.shape[0], :]
    emb = emb + pos
    """ print(emb)
    print(emb.shape)
    plot_attention_heatmap(emb) """

    key = self.layer_00_key(emb)
    qry = self.layer_00_qry(emb)
    val = self.layer_00_val(emb)
    att = torch.mm(qry, key.t())
    msk = self.mask[0:x.shape[0], 0:x.shape[0]]
    att = att.masked_fill(msk == 0, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=1)
    res = torch.mm(att, val)
    res = self.layer_00_ffw(res)

    key = self.layer_01_key(res)
    qry = self.layer_01_qry(res)
    val = self.layer_01_val(res)
    att = torch.mm(qry, key.t())
    msk = self.mask[0:x.shape[0], 0:x.shape[0]]
    att = att.masked_fill(msk == 0, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=1)
    res = torch.mm(att, val)
    res = self.layer_01_ffw(res)
    """ plot_attention_heatmap(res) """

    out = self.map_to_vocab(res)
    """ plot_attention_heatmap(res) """
    return out, att

  def get_pos_matrix(self):
    store = torch.zeros(max_seq_len, 7)
    for pos in range(max_seq_len):
      for i in range(0, 7, 2):
        denominator = 10000 ** (2 * i / 7)
        store[pos, i] = math.sin(pos / denominator)
        if i + 1 < 7: store[pos, i + 1] = math.cos(pos / denominator)
    return store


m = BesSimpleTransformer()
opt = torch.optim.SGD(m.parameters(), lr=0.01)

#%%
for epoch in range(10):
  for idx, batch in enumerate(dl):

    sos = torch.tensor([2])
    eos = torch.tensor([1])

    x = batch[0]
    x = torch.cat([sos, x])
    print(x)
    y = torch.cat([x[1:], eos])
    print(y)

    p, attn = m(x)
    if idx % 1000 == 0:
      """ vocab = tokenizer.get_vocab()
      intermediate_output, _ = m(torch.tensor(tokenizer.encode(''.join(vocab[3:]))))
      plot_output_heatmap(intermediate_output, reference=vocab, candidate=vocab) """
    l = torch.nn.functional.cross_entropy(p, y)
    # if idx % 1000 == 0: print("Loss:", l.item())
    l.backward()
    opt.step()
    opt.zero_grad()

  x = tokenizer.decode([random.randint(3, 25)])
  x = torch.cat([sos, torch.tensor(tokenizer.encode([x]))])
  while True:
    p, _ = m(x)
    p = torch.nn.functional.softmax(p, dim=1)
    p = torch.argmax(p, dim=1)
    x = torch.cat([x, p[-1].unsqueeze(0)])
    if p[-1] == 1 or len(p.tolist()) == 17: break
  #print("Generate:", tokenizer.decode(x.tolist()))

# %%
x = tokenizer.decode([random.randint(3, 25)])
print(x)
x = torch.cat([sos, torch.tensor(tokenizer.encode([x]))])
print(x)
p, attn = m(x)
plot_attention_heatmap(p)
plot_attention_heatmap(attn)



# %%
print(''.join(vocab[3:]))
# %%
