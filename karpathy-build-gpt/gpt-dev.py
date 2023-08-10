#%%

# Name generation
# One attention block at a time
# Printing the attention heads (heatmaps of attention) - what were the weights responsible for 

#%%
text = open('input.txt', 'r').read()
len(text)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
# %%
