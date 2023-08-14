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

    self.attn_one = SelfAttention(self.max_seq_len, self.embedding_dimensions)
    self.attn_two = SelfAttention(self.max_seq_len, self.embedding_dimensions)

    self.map_to_vocab = torch.nn.Linear(self.embedding_dimensions, self.vocab_len)

  def forward(self, x):
    emb = self.embedding(x)
    pos = self.pos_emb[0:x.shape[0], :]
    emb = emb + pos

    res = self.attn_one(x, emb)
    res = self.attn_two(x, res)

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
  
class SelfAttention(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dimensions):
        super(SelfAttention, self).__init__()
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.key = torch.nn.Linear(embedding_dimensions, 11)
        self.qry = torch.nn.Linear(embedding_dimensions, 11)
        self.val = torch.nn.Linear(embedding_dimensions, 11)
        self.feed_forward = torch.nn.Linear(11, embedding_dimensions)

    def forward(self, x, x_embeddings):
        key = self.key(x_embeddings)
        qry = self.qry(x_embeddings)
        val = self.val(x_embeddings)

        att = torch.mm(qry, key.t())
        msk = self.mask[0:x.shape[0], 0:x.shape[0]]
        att = att.masked_fill(msk == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=1)
        res = torch.mm(att, val)

        res = self.feed_forward(res)
        return res