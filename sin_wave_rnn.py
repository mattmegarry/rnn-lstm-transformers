"""
"In conventional feed-forward neural networks, all test cases are considered to be independent.
The NN model would not consider the previous stock price values – not a great idea!"

RNNs are all about sequences - past training examples are involved in the current examples training.

"What does our network model expect the data to be like? It would accept a single sequence of length 50 as input. 
So the shape of the input data will be:
(number_of_records x length_of_sequence x types_of_sequences)"

"""
#%%
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#%%
def dataset(size =  200, timesteps = 25):
    x, y = [], []
    sin_wave = np.sin(np.arange(size))
    for step in range(sin_wave.shape[0]-timesteps):
        x.append(sin_wave[step:step+timesteps])
        y.append(sin_wave[step+timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)

# %%
class RNN:
    def __init__(self, x, y, hidden_units, lr = 0.01):
        self.x = x # shape [samples, timesteps, features = 1 in our case]
        print(self.x.shape)
        print(self.x[0])
        print(self.x[1])
        print(self.x[2])
        print(self.x)
        self.y = y # shape [samples, outputs]
        print(self.y.shape)
        print(self.y[0])
        print(self.y[1])
        print(self.y[2])
        print(self.y[3])
        print(self.y)
        self.mode = 'train'
        self.hidden_units = hidden_units
        print(self.hidden_units)
        self.lr = lr
        self.Wx = np.random.randn(self.hidden_units, self.x.shape[2])
        self.Wh = np.random.randn(self.hidden_units, self.hidden_units)
        self.Wy = np.random.randn(self.y.shape[1],self.hidden_units)

    def cell(self, xt, ht_1):
        ht = np.tanh(np.dot(self.Wx,xt.reshape(1,1)) + np.dot(self.Wh,ht_1))
        yt = np.dot(self.Wy,ht)
        return ht, yt
    
    def forward(self, sample, test_x = None, test_y = None):
        if self.mode == 'train':
            sample_x, sample_y = self.x[sample], self.y[sample]
        elif self.mode == 'eval':
            sample_x, sample_y = test_x[sample], test_y[sample]
        
        ht = np.zeros((self.hidden_units,1)) # first hidden state is zeros vector
        self.hidden_states = [ht] # collection of hidden states for each sample
        self.inputs = [] # collection of inputs for each sample
        for step in range(len(sample_x)):
            ht, yt = self.cell(sample_x[step],ht)
            self.inputs.append(sample_x[step].reshape(1,1))
            self.hidden_states.append(ht)
        
        self.error = yt - sample_y
        self.loss = 0.5*self.error**2
        self.yt = yt
        return self.loss, self.yt

    def backward(self):
        n = len(self.inputs)
        dyt = self.error # dL/dyt
        dWy = np.dot(dyt,self.hidden_states[-1].T) # dyt/dWy
        dht = np.dot(dyt, self.Wy).T # dL/dht = dL/dyt * dyt/dht ,where ht = tanh(Wx*xt + Wh*ht))
        dWx = np.zeros(self.Wx.shape)
        dWh = np.zeros(self.Wh.shape)
        # BPTT
        for step in reversed(range(n)):
            temp = (1-self.hidden_states[step+1]**2) * dht # dL/dtanh = dL/dyt * dyt/dht * dht/dtanh, where dtanh = (1-ht**2) 
            dWx += np.dot(temp, self.inputs[step].T) # dL/dWx = dL/dyt * dyt/dht * dht/dtanh * dtanh/dWx
            dWh += np.dot(temp, self.hidden_states[step].T) # dL/dWh = dL/dyt * dyt/dht * dht/dtanh * dtanh/dWh

            dht = np.dot(self.Wh, temp) # dL/dht-1 = dL/dht * (1 - ht+1^2) * Whh
        dWy = np.clip(dWy, -1, 1)
        dWx = np.clip(dWx, -1, 1)
        dWh = np.clip(dWh, -1, 1)
        self.Wy -= self.lr * dWy
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh

    def train_mode(self):
        self.mode = 'train'

    def eval_mode(self):
        self.mode = 'eval'   

#%%
x,y = dataset()
x_test, y_test = dataset(300)
x_test = x_test[250:]
y_test = y_test[250:]

#%%
""" TRAIN """
model = RNN(x,y,hidden_units=1)
epochs = 100

Ovr_loss = []
for epoch in tqdm(range(epochs)):
    for sample in range(model.x.shape[0]):
        loss, _ = model.forward(sample)
        model.backward()
    Ovr_loss.append(np.squeeze(loss / model.x.shape[0]))
    loss = 0

plt.plot(Ovr_loss)

#%%
""" TEST """
model.eval_mode()
test_outputs = []
for sample in range(len(x_test)):
    _, yt = model.forward(sample, x_test, y_test)
    test_outputs.append(yt)
model.train_mode()


plt.tight_layout()
plt.figure(dpi=120)
plt.subplot(121)
plt.plot(Ovr_loss)
plt.subplot(122)
plt.plot([i for i in range(len(x_test))],y_test,np.array(test_outputs).reshape(y_test.shape))

#%%
