#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(1111)

# Generate random data
X = torch.rand(1000, 10)
y = torch.rand(1000, 1)

# Model with residual connection
model = nn.Sequential(
  nn.Linear(10, 32),       # 0
  nn.ReLU(),               # 1
  nn.Linear(32, 32),       # 2
  nn.ReLU(),               # 3 - Save residual here!
  nn.Linear(32, 32),       # 4
  nn.ReLU(),               # 5
  nn.Linear(32, 32),       # 6 - Add residual here!
  nn.ReLU()                # 7
)

# Forward pass
def forward_residual(X):
    out = model[0](X)
    out = model[1](out)
    out = model[2](out)
    out = model[3](out)
    residual = model[3](out)
    out = model[4](out)
    out = model[5](out)
    out = out + residual 
    out = model[6](out)
    out = model[7](out)
    return out

# Forward pass
def forward(X):
    out = model[0](X)  
    out = model[1](out)
    out = model[2](out)
    out = model[3](out)
    out = model[4](out)
    out = model[5](out)
    out = model[6](out)
    out = model[7](out)
    return out

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) 

layer_6_node_0_input_0_gradient_no_res = []
layer_6_node_0_input_0_gradient_res = []

losses = []
for epoch in range(100):
    y_pred = forward(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    layer_6 = model[6]
    layer_6_node_0_input_0_gradient_no_res.append(layer_6.weight.grad[0][0].item())
    optimizer.step()
    losses.append(loss.item())

losses_res = []
for epoch in range(100):
    y_pred = forward_residual(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    layer_6 = model[6]
    layer_6_node_0_input_0_gradient_res.append(layer_6.weight.grad[0][0].item())
    optimizer.step()
    losses_res.append(loss.item())

plt.plot(losses, label='No residual')
plt.plot(losses_res, label='Residual')
plt.legend()



#%%
plt.plot(layer_6_node_0_input_0_gradient_no_res, label='No residual')
plt.plot(layer_6_node_0_input_0_gradient_res, label='Residual')
plt.legend()
plt.show()
# %%