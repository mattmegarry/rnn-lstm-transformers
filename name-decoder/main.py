#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

from vis_utils import plot_embedding_heatmap
#%%
words = open('names.txt', 'r').read().splitlines()
print(len(words))
print(words[:10])
words = words[:10] # lets start with a small dataset!
# actually, lets start with a small dataset that covers all the characters we need (26)
words = ['azlan', 'katya', 'lexie', 'frank', 'every', 'harry', 'mikey', 'bojom', 'wcpago', 'saqua', 'david']
#%%
chars = sorted(list(set(''.join(words))))
print(chars)
stoi = {s:i for i,s in enumerate(chars)}
#stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
print(stoi)
#%%

embed_size = 8
vocab_size = len(itos)
print(vocab_size)

# Does nn.Embedding just create  of random numbers of specified size?
embedding = nn.Embedding(vocab_size, embed_size)

print(embedding)
print(type(embedding))
# doing a plot of this may not be meaningful ???
# why is it dimension 17 x 4 ? I would imagine it should be 10 x 4 - oh! 17 is vocab size when we have 10 words haha
plot_embedding_heatmap(embedding.weight.detach().numpy())

# Encode as IDs
input_ids = list(itos.keys())
#input_ids = np.array(input_ids)

print(input_ids)
print(torch.tensor(input_ids))
# Get embeddings
# ValueError: expected sequence of length 5 at dim 1 (got 7)
embeddings = embedding(torch.tensor(input_ids))
print(embeddings.detach().numpy().shape)
plot_embedding_heatmap(embeddings.detach().numpy())

# %%
# Positional encoding
seq_len = 26 # For convenience - for now?
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

positional_encoding = getPositionEncoding(seq_len, d=embed_size, n=100)
print(positional_encoding)

# Positional encoding only
plot_embedding_heatmap(positional_encoding)

# Add positional encoding
pos_encoded_embeddings = embeddings.detach().numpy() + positional_encoding
plot_embedding_heatmap(pos_encoded_embeddings)
#%%

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SingleHeadSelfAttention, self).__init__()
        self.embed_size = embed_size

        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)
    
    def forward(self, values, keys, queries):
        queriesAtTimeStep = self.queries(queries)
        keysAtTimeStep = self.keys(keys)
        valuesAtTimeStep = self.values(values)
        
        # attention calculation
        weights = queriesAtTimeStep @ keysAtTimeStep.T / self.embed_size**0.5
        print(weights.shape)
        print(weights)
        
        attention_weights = nn.functional.softmax(weights, dim=-1)
        print("\n")
        print(attention_weights)
        
        weighted_values = attention_weights @ valuesAtTimeStep
        print("\n")
        print(weighted_values)
        # -- end attention calculation

        outputs = self.fc_out(weighted_values)
        return outputs

#%%

attention = SingleHeadSelfAttention(embed_size)

for word in words[:1]:
    char_ids = [stoi[c] for c in word]
    sequence_pos_embeddings = []
    for i in range(len(char_ids)):
        sequence_pos_embeddings.append(torch.tensor(pos_encoded_embeddings[char_ids[i]], dtype=torch.float32))
        print(itos[char_ids[i]])
        # TO DO: EOS token
    print("\n")
    sequence_pos_embeddings = tuple(sequence_pos_embeddings)
    sequence_input = torch.cat(sequence_pos_embeddings, dim=0).reshape(len(char_ids), embed_size)
    attention(sequence_input, sequence_input, sequence_input)
    
        
# %%
