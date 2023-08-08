from pprint import pprint
import torch
import torch.nn.functional as F


test_input = [
  [
    [1, 8, -1, 1],  
    [2, 9, 0, 2],
    [3, 10, 1, 3]
  ],
  
  [ 
    [2, 3, 2, 20],
    [3, 4, 3, 21], 
    [4, 5, 4, 22]
  ]
]

print("\n\n")
print("The first sentence would be: 1 2 3.")
pprint(test_input, depth=5, width=30)

# assume we have some tensor x with size (b, t, k)
x = torch.tensor(test_input, dtype=torch.float32)

print("\n\n")
print("The input as a tensor:")
print(x)

print("\n\n")
print("The input transposed:")
print(x.transpose(1, 2))


""" raw_weights = torch.bmm(x, x.transpose(1, 2))

print(raw_weights.size())
print(raw_weights[0])

weights = F.softmax(raw_weights, dim=2)

print()
y = torch.bmm(weights, x) """