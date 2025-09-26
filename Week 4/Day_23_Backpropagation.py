### Day 23 -- Backpropagation
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

    def activate_prime(self, input : list[float]) -> float:
            sigmoid = self.activate(input)
            return sigmoid * (1 - sigmoid)

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

class NeuralNetwork:
    def __init__(self, layers: list[int]):
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

    def feedforward(self, input: list[float]) -> tuple(list[list[float]]):
        """
        Args :
            -input : the input of the network
    
        Return a 
        """
        activations = [input]  # list of the activation output of each neuron per layer
        zs = []  # list of z (sum of weight * input) per layer

        for layer in self.layers:
            z = []
            activation = []
            
            for neuron in layer:
                z_i = np.dot(neuron.weights, input) + neuron.bias
                a_i = neuron.activate(input)
                
                z.append(z_i)
                activation.append(a_i)
                
            zs.append(z)
            activations.append(activation)
            input = activation

        return activations, zs

    def backpropagate(self, x: list[float], y: list[float], learning_rate: float = 0.1):
        # forward pass
        activations, zs = self.feedforward(x)

        # backward pass
        delta = []  # deltas per layer
        grad_biases = [None] * len(self.layers)
        grad_weights = [None] * len(self.layers)

        # step 1: output layer error
        output = np.array(activations[-1])
        y = np.array(y)
        error = output - y
        
        z = np.array(zs[-1])
        
        # here is the derivate of the cost function for the output
        delta_output = error * np.array([neuron.activate_prime(z_i) for neuron, z_i in zip(self.layers[-1], z)])
        
        delta.insert(0, delta_output)

        grad_biases[-1] = delta_output
        grad_weights[-1] = np.outer(delta_output, activations[-2])

        # step 2: hidden layers
        # the range goes from 2 to nb_layers + 1 because we indent by -l from -2 to 0
        for l in range(2, len(self.layers)+1):
            z = np.array(zs[-l])
            # sp = array of derivate of the activation of each neuron for z_i
            sp = np.array([neuron.activate_prime(z_i) for neuron, z_i in zip(self.layers[-l], z)])
            # w_next = matrix of weights for weights of the next layer with row_i
            w_next = np.array([neuron.weights for neuron in self.layers[-l + 1]])
            
            delta_next = delta[0]
            delta_hidden = np.dot(w_next.T, delta_next) * sp

            delta.insert(0, delta_hidden)
            grad_biases[-l] = delta_hidden
            grad_weights[-l] = np.outer(delta_hidden, activations[-l - 1])

        # Step 3: Gradient descent update
        for l in range(len(self.layers)):
            for i, neuron in enumerate(self.layers[l]):
                # small trick to substract a value to all element in a list
                neuron.weights = (np.array(neuron.weights) - learning_rate * grad_weights[l][i]).tolist()
                neuron.bias -= learning_rate * grad_biases[l][i]

    def train(self, data: list[tuple[list[float], list[float]]], epochs: int = 1000, learning_rate: float = 0.1):
        for epoch in range(epochs):
            for x, y in data:
                self.backpropagate(x, y, learning_rate)

    def predict(self, x: list[float]) -> list[float]:
        output, _ = self.feedforward(x)
        return output[-1]

### TESTING OUR NETWORK
# Problème : fonction OR
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [1])
]

# Création du réseau : 2 entrées, 2 neurones cachés, 1 sortie
net = NeuralNetwork([2, 2, 1])
net.train(training_data, epochs=1000, learning_rate=0.5)

# Test du réseau
for x, y in training_data:
    pred = net.predict(x)
    print(f"Input: {x}, Predicted: {pred}, Expected: {y}")

###
"""
We will see tomorrow how can we train a neural network with gradient descent
"""

# Author GCreus
# Done via pyzo
