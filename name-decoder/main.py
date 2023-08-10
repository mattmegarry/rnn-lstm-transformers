#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from vis_utils import plot_embedding_heatmap
#%%
words = open('names.txt', 'r').read().splitlines()
print(len(words))
print(words[:10])
words = words[:10] # lets start with a small dataset!
# actually, lets start with a small dataset that covers all the characters we need (26)
words = ['azlan', 'cpago', 'saqua', 'david', 'lexie', 'frank', 'every', 'harry', 'mikey', 'katya', 'bojom']
#%%
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
print(stoi)
#%%

embed_size = 5
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
input_ids = [[stoi[c] for c in chars] for chars in words]
input_ids = np.array(input_ids).flatten()
input_ids = np.unique(input_ids) # this will be so slow for large datasets
print(input_ids)
# Get embeddings
# ValueError: expected sequence of length 5 at dim 1 (got 7)
embeddings = embedding(torch.tensor(input_ids))
print(embeddings.detach().numpy().shape)
plot_embedding_heatmap(embeddings.detach().numpy())

# %%
