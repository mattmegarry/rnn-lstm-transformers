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


#%%
np.array([1,2,3,4,5]).reshape(5,1)

#%%
print(np.random.randn(10, 10))

# %%
print(np.array([[]]).shape[1])
print(np.array([[[]]]).shape[2])
print(np.array([2,3],[1,6]))
print(np.array([[[[[1],[2],[3]]]]]).shape[3])
print(np.array([[[[[1],[2],[3]]]]],[]).shape[4]) #
array = np.array([[[[[1],[2],[3]]]]])
print(array[4])
print(np.array([[[4,3,5,2,4]]]).shape[2])
# %%
np.dot(np.array([2,2,1]),np.array([2,2,1]))
np.dot(np.array([[1,2,3]]), np.array([1,2,3]))
# %%
