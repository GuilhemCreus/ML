### Day 28 -- Basics of PyTorch 5
"""
We will now use what we have learned with the same case : an 'or' logic gate

We will see how simple it is to implement such a NN with PyTorch compared to doing it from scratch with numpy
"""

import torch
import torch.nn as nn
import torch.optim as optim
### CREATING THE NN CLASS

# don t forget that our network inherits from the super class nn.module
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(2, 2)      # 2 inputs -> 2 outputs
        self.output = nn.Linear(2, 1)      # output layer : 2 inputs -> 1 output
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))  # we play a bit with the activation functions
        return x

### TESTING OUR NETWORK
# initialization of an instance of the class
net = SimpleNet()

# training data
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])

# loss function
criterion = nn.MSELoss()

# simple gradient descent as the optimizer
optimizer = optim.RMSprop(net.parameters(), lr=0.5, alpha = 0.8)

# training loop
epochs = 140
for epoch in range(epochs):
    # forward
    outputs = net(X)
    loss = criterion(outputs, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():
    for i in range(len(X)):
        x_input = X[i]
        prediction = net(x_input).item()
        print(f"Input: {x_input.tolist()}, Predicted: {prediction:.4f}, Expected: {y[i].item()}")

"""
We see how simple it is to modify the activation functions for each layer
Same conclusion with the optimizer, with one line modified we can change our approach to the optimization of the network

It enables us to be more flexible when testing our network
For example, if we choose "sigmoid" for the first layer rather than "ReLU", our network performs more badly

And if we choose :
-RMSprop(net.parameters(), lr=0.5, alpha = 0.8) as optimizer
-ReLU as activation function for the first layer
-sigmoid for the second layer
We get a perfect prediction for only 140 epochs
"""

# Author GCreus
# Done via pyzo
