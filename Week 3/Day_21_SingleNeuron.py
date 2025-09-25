### Day 21 -- Single neuron
"""
Lastly, we will see how a single neuron works in a neural network
Why a neuron ? Because they form a neural network and neural networks are used in reinforcement learning as an extension of Q-tables when the states are too complex to be treated with Q-table
For example, it happens when all the states form a continuous space and not a discrete space

Before diving into creating such a solution, we will before explain how a neuron work
And then, how a neural network works
"""

import numpy as np

### SINGLE NEURON
"""
A single neuron stems from the biological neuron
It can takes multiple inputs and applies weights to them; then it sums them up and  adds a bias

The result is then passed through an activation function

So mathematically speaking it can be represented as :

    output = activation(w1 * x1 + w2 * x2 + ... + wn * xn + b)

With :
-x1, x2, ..., xn are the inputs (features)
-w1, w2, ..., wn are the weights
-b is the bias
-activation is a non-linear function like sigmoid, ReLU, or tanh (generally sigmoid)

We can also explain the above without mathematcis
Remind that :
-the inputs are multiplied by their respective weights and summed to equal a value I
-the bias represents a value to overcome in order to activate the neuron
-so if I + b is > 0, the neuron is activated and it sends a positive value to the neurons of the next layer; otherwise it sends 0
-in most case if I + b > 0, the neuron sends 1  but there is a really short continuous transition from 0 to 1

The purpose of the activation function is to introduce non-linearity so the network can learn complex patterns

This is the basic block of all neural networks

In our example we will use the most used activation function : sigmoid
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# we generate two random inputs
inputs = np.array([5.5, 0.8])

# with two random weights
weights = np.array([0.4, 0.7])

# we then add a random bias
bias = 0.1

# we calculate the input to be sent to the activation function
z = np.dot(inputs, weights) + bias

# we then calculate the output to be sent to the next layer
output = sigmoid(z)

print("The neuron will send :", output)
# So here the output is 0.94 so the neuron is activated

# Author GCreus
# Done via pyzo
