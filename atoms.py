#%%
import numpy as np
from matplotlib import pyplot as plt
import torch

#%%
result = np.random.randn(100000)
print(result)
plt.hist(result, bins = 100)
plt.show()

#%%
print(np.array(1).shape)
print(np.array([]).shape)
print(np.array([1]).shape)
print(np.array([1,1]).shape)
print(np.array([1,1,1]).shape)

print(np.array([[]]).shape)
print(np.array([[[]]]).shape)
print(np.array([[[1]]]).shape)

print("")

print(torch.tensor(1).shape)
print(torch.tensor([]).shape)
print(torch.tensor([1]).shape)
print(torch.tensor([1,1]).shape)
print(torch.tensor([1,1,1]).shape)

print(torch.tensor([[]]).shape)
print(torch.tensor([[[]]]).shape)
print(torch.tensor([[[1]]]).shape)


# %%
np.array([1,2,3,4,5]).reshape(5,1)
