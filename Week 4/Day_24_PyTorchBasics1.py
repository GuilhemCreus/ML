### Day 24 -- Basics of PyTorch 1
"""
In order to have more flexibility, we will now work with PyTorch which is a library that handles neural network easily
We will stick to this library for some days before creating our first deep Q learning algorithm
Doing so will also deepen my knowledge of PyTorch which is for the time, very limited
"""
import torch
import torch.nn as nn

### SIMPLE PYTORCH NETWORK
"""
Our network will inherit from the class nn.Module that provides all the base functionality for building neural networks such as parameter management, forward computation and integration with autograd
"""
class ORNet(nn.Module):
    def __init__(self):
        # super() calls the method from the parent class
        super(ORNet, self).__init__()

        # nn.Linear create a linear layer, i.e output = input @ W.T + b
        self.hidden = nn.Linear(2, 2)   # one hidden layer : 2 neurons and 2 inputs
        self.output = nn.Linear(2, 1)   # output : 1 neuron and 2 inputs

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # outputs of the hidden layer
        x = torch.sigmoid(self.output(x))  # output of the output layer
        return x

### BACKPROPAGATION
"""
A quick overview of hooks and autograd :
-hooks are special functions that you can attach to modules or tensors to automatically intervene at key points in the flow of data through a neural network; like printing a tensor during a forward or extract a tensor during a backward
-autograd is a dynamic graph that PyTorch builds on the fly as operations are executed; each tensor operation is tracked as a node in this graph and the graph is used during backpropagation to automatically compute gradients with respect to each parameter
"""
# creating an instance of our class
network = ORNet()

# creating the "dataset" but using tensor because PyTorch works with tensor
# tensors are like numpy array but organized in such a way that tensors are well fited for matrix calculus
training_data = [
    (torch.tensor([0., 0.]), torch.tensor([0.])),
    (torch.tensor([0., 1.]), torch.tensor([1.])),
    (torch.tensor([1., 0.]), torch.tensor([1.])),
    (torch.tensor([1., 1.]), torch.tensor([1.]))
]

# parameters
learning_rate = 0.5
epochs = 1000

# training loop, in the future we will use a module of PyTorch that does this automatically
for epoch in range(epochs):
    for x, y in training_data:
        # we initialize the gradients to zero
        network.zero_grad()

        # we do not call the forward method, instead we use __call__ method from nn.Module which automatically calls the forward method and handles autograd hooks, context tracking, and other internal PyTorch mechanisms
        output = network(x)

        # MSE calculation
        loss = ((output - y) ** 2).mean()

        # backward gradients calculation using autograd, now all parameters have .grad value that represents the gradient of MSE regarding that parameter
        loss.backward()

        # manually updating parameters and we deactivate temporarily autograd so the graph behind autograd doesn't think that the operation "param -= learning_rate * param.grad" needs to be differentiated which could cause trouble
        with torch.no_grad():
            for param in network.parameters():
                param -= learning_rate * param.grad


### TESTING OUR NETWORK
for x, y in training_data:
    pred = network(x)
    print(f"Input: {x.tolist()}, Predicted: {pred.item():.4f}, Expected: {y.item()}")

"""
We got approximately the same results and PyTorch is much flexible and well optimized compared to our previous code
"""

# Author GCreus
# Done via pyzo
