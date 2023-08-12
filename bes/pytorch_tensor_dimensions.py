import torch

oned = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
twod = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
threed = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])

print(oned.shape)
print(oned.size())
print(oned.unsqueeze(0).shape)

print(twod.shape)
print(threed.shape)

# Python Dictionary Comprehension
list = ["cheeses", "apples", "fishes", "bananas", "oranges"]
fresh_quadratic_shopping = {"Fresh " + c:i ** 2 for i, c in enumerate(list)}
print(fresh_quadratic_shopping)