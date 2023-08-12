#%%
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# I will implement attention scoring as well as calculating an attention context vector.

"""
Let’s start by looking at the inputs we’ll give to the scoring function. We will assume we’re in 
the first step in the decoding phase. The first input to the scoring function is the hidden state 
of the decoder (assuming a toy RNN with three hidden nodes — not usable in real life, 
but easier to illustrate)
"""

#%%
decoder_hidden_state = np.array([5, 1, 20])
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(decoder_hidden_state)), annot=True, cmap=sns.light_palette("purple", as_cmap=True), linewidths=1)




# %%
