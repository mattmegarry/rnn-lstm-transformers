import torch
import torch.nn as nn

# 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576

class SelfAttentionEinsum(nn.Module):
    def __init__(self, embed_size, heads): # heads is the number of portions to split the embedding into
        super(SelfAttentionEinsum, self).__init__() # initialize the parent class
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # number of training examples processed at once (batch size)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys]) # einsum is Einstein summation
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out after einsum: (N, query_len, heads, head_dim), then reshape to (N, query_len, heads*head_dim)

        out = self.fc_out(out)
        return out
    
class SelfAttentionHead(nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = self.emb_dim // self.heads
        
    def forward(self, queries, keys, values, mask):
        
        batch_size, seq_len, _ = queries.shape
        
        outputs = torch.empty(batch_size, seq_len, self.emb_dim)
        
        for i in range(batch_size):
            for h in range(self.heads):
                
                start = h * self.head_dim
                end = (h+1) * self.head_dim
                
                # Slice on dimension 2
                q = queries[i, :, start:end]
                k = keys[i, :, start:end] 
                v = values[i, :, start:end]  
                
                # Attention calculation
                weights = q @ k.T / self.emb_dim**0.5
                weights = nn.functional.softmax(weights, dim=-1)
                
                weighted_v = weights @ v
                
                # Copy to output 
                outputs[i, :, start:end] = weighted_v
                
        return outputs
    
""" optimized_attn = SelfAttentionEinsum(8, 2)
raw_attn = SelfAttentionHead(8, 2)

values = torch.rand(1, 4, 8)
keys = torch.rand(1, 4, 8)
query = torch.rand(1, 4, 8)

print(optimized_attn(values, keys, query, None).shape)
print(optimized_attn(values, keys, query, None))

print(raw_attn(query, keys, values, None).shape)
print(raw_attn(query, keys, values, None)) """

# Input sequence containing repeats
input_seq = torch.tensor([[1, 2, 3, 4, 5],  
                          [3, 3, 3, 6, 6],
                          [1, 2, 5, 4, 5],])

input_seq = torch.tensor([[1, 28, 674, 672, 400, 400, 400, 1 ],  
                          [1, 28, 28, 2, 4, 5, 7, 800 ],
                          [1, 27, 673, 671, 402, 399, 401, 2],])

emb_dim = 4
heads = 2

# Embed inputs 
embed = nn.Embedding(10, emb_dim)
queries = embed(input_seq)
keys = queries
values = queries

# Create model
model = SelfAttentionHead(emb_dim, heads)
model_einsum = SelfAttentionEinsum(emb_dim, heads)

outputs = model(queries, keys, values, mask=None)
print(outputs.shape)
print(outputs)

outputs_einsum = model_einsum(queries, keys, values, mask=None)
print(outputs_einsum.shape)
print(outputs_einsum)

# nn.Embeddding lookup table
# https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
#
