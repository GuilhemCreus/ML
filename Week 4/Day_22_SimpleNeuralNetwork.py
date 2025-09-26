### Day 22 -- Neural network without backpropagation
"""
We will now put multiple layers of neurons together in order to create a neural network
A neural network is made of multiple layers of single neurons connected to each other

We will show how to optimize it in our next code
"""

import numpy as np

### CLASS NEURON
"""
In order to create multiple instances of Neuron in different layers with ease, we will use OOP for this code

We will create a class Neuron and a class NeuralNetwork that contains layers of Neuron

The class Neuron will simply have a constructor and an activation function as we did yesterday
"""

class Neuron:
    def __init__(self, weights : list[float], bias : float):
        self.weights = weights
        self.bias = bias

    def activate(self, input : list[float]) -> float:
        return 1/ (1 + np.exp(-(np.dot(self.weights, input) + self.bias)))


### CLASS NEURAL NETWORK
"""
The class NeuralNetwork will be called through its constructor that needs the structure of the layers
The structure of the layer for the constructor is a list of int, the length of the list specifies the number of layers and the integer at list[i] specifies the number of Neuron for the layer i

With :
-the layer 0 as the input layer, i.e the data that we feed into the network
-the layer -1 as the output layer

For a given layer, all the neurons will be feeded the same inputs, it could be the input data or the output of the previous layer
So, for layer 1, the input data fed through all neurons will be the input of the network; so for layer 1, each neurons will have to have weights with the same dimension as the input data
And for layer i, the input data fed through all neurons will be the ouput of the previous layer; so for layer i, each neurons will have to have weights with the same dimension as the outputs of layer i-1, i.e the number of neurons before the layer because each neuron gives one output so if the dimension of the output is 5 this is equivalent as having 5 neurons before
"""

class NeuralNetwork():
    def __init__(self, layers : list[int]):
        self.layers = []

        for i in range(1, len(layers)):
            input_lenght = layers[i - 1]
            output_length = layers[i]

            layer = []
            for _ in range(output_length):
                bias = np.random.randn()
                weights = np.random.randn(input_lenght).tolist()
                neuron = Neuron(weights, bias)

                layer.append(neuron)

            self.layers.append(layer)

    def feedforward(self, input : list[float]) -> list[float]:
        for layer in self.layers:
            output = []
            for neuron in layer:
                output.append(neuron.activate(input))

            # the input for the following layer will be the output of the previous
            input = output

        return input

### TESTING OUR NETWORK
network = NeuralNetwork([2, 5, 5, 2])
input = [1, 2]

output = network.feedforward(input)
print("The network has the following as output : ", output)

"""
We will see tomorrow how can we train a neural network with gradient descent
"""

# Author GCreus
# Done via pyzo
