import torch
import math

class DecoderModel(torch.nn.Module):
  def __init__(self, max_seq_len, vocab_len, embedding_dimensions):
    super(DecoderModel, self).__init__()
    
    self.max_seq_len = max_seq_len
    self.vocab_len   = vocab_len
    self.embedding_dimensions = embedding_dimensions

    self.embedding    = torch.nn.Embedding(self.vocab_len, self.embedding_dimensions)
    self.pos_emb      = self.get_pos_matrix()

    self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))

    self.layer_00_key = torch.nn.Linear(7, 11)
    self.layer_00_qry = torch.nn.Linear(7, 11)
    self.layer_00_val = torch.nn.Linear(7, 11)
    self.layer_00_ffw = torch.nn.Linear(11, 7)

    self.layer_01_key = torch.nn.Linear(7, 11)
    self.layer_01_qry = torch.nn.Linear(7, 11)
    self.layer_01_val = torch.nn.Linear(7, 11)
    self.layer_01_ffw = torch.nn.Linear(11, 7)

    self.map_to_vocab = torch.nn.Linear(7, self.vocab_len)

  def forward(self, x):
    emb = self.embedding(x)
    pos = self.pos_emb[0:x.shape[0], :]
    emb = emb + pos

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

    out = self.map_to_vocab(res)

    return out

  def get_pos_matrix(self):
    store = torch.zeros(self.max_seq_len, 7)
    for pos in range(self.max_seq_len):
      for i in range(0, 7, 2):
        denominator = 10000 ** (2 * i / 7)
        store[pos, i] = math.sin(pos / denominator)
        if i + 1 < 7: store[pos, i + 1] = math.cos(pos / denominator)
    return store