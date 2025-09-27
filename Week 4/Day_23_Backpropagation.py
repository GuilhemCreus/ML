### Day 23 -- Backpropagation
"""
We will now see the most spreaded algorithm to fit neural networks to problems
It is called backpropagation due to the way it works

In fact, this algorithm start from the last layer of the network through the first layer
"""
import numpy as np

### CLASS NEURON
class Neuron:
    def __init__(self, weights : list[float], bias : float):
        self.weights = weights
        self.bias = bias

    # smalle change here in order to be able to take precisely one z
    def activate(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))

    def activate_prime(self, z: float) -> float:
        sigmoid = self.activate(z)
        return sigmoid * (1 - sigmoid)

### CLASS NEURAL NETWORK
"""
Math explanation is well explained here : https://www.youtube.com/watch?v=tIeHLnjs5U8
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

    def feedforward(self, input: list[float]) -> tuple[list[list[float]], list[list[float]]]:
        """
        Args :
            -input : the input of the network

        Return a tuple containing two lists,
            -the first one is the list of all output of each neuron in each layer
            -the second is the list of the z fed for each neuron in each layer
        """
        activations = [input]  # list of the activation output of each neuron per layer
        zs = []  # list of z (sum of weight * input) per layer

        for layer in self.layers:
            z = []
            activation = []

            for neuron in layer:
                z_i = np.dot(neuron.weights, input) + neuron.bias
                a_i = neuron.activate(z_i)

                z.append(z_i)
                activation.append(a_i)

            zs.append(z)
            activations.append(activation)
            input = activation

        return activations, zs

    def backpropagate(self, x: list[float], y: list[float], learning_rate: float = 0.1) -> None:
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

        # we initialize the delta for the layers below the last one
        delta.insert(0, delta_output)

        grad_biases[-1] = delta_output
        # the outer product enables us to have grad_weights[i, j] = deltaj.ai(l-1)
        grad_weights[-1] = np.outer(delta_output, activations[-2])

        # step 2: hidden layers
        # the range goes from 2 to nb_layers + 1 because we indent by -l from -2 to 0
        for l in range(2, len(self.layers)+1):
            z = np.array(zs[-l])
            # sp = array of derivate of the activation of each neuron regarding z at that neuron
            sp = np.array([neuron.activate_prime(z_i) for neuron, z_i in zip(self.layers[-l], z)])
            # w_next = matrix of weights for weights of the next layer with row_i
            w_next = np.array([neuron.weights for neuron in self.layers[-l + 1]])

            delta_next = delta[0]
            delta_hidden = np.dot(w_next.T, delta_next) * sp

            # we move all delta to the next slot in the list in order to have the first slot free
            # then we insert the new delta inside
            # with this order, when we will check the previous layer, the delta in the next layer is delta[0]
            delta.insert(0, delta_hidden)

            grad_biases[-l] = delta_hidden
            # the outer product enables us to have grad_weights[i, j] = deltaj.ai(l-1)
            grad_weights[-l] = np.outer(delta_hidden, activations[-l - 1])

        # step 3: gradient descent update
        for l in range(len(self.layers)):
            # for each neuron in each layer, we do a gradient descent
            for i, neuron in enumerate(self.layers[l]):

                # small trick to substract a value to all element in a list
                neuron.weights = (np.array(neuron.weights) - learning_rate * grad_weights[l][i]).tolist()
                neuron.bias -= learning_rate * grad_biases[l][i]

    def train(self, data: list[tuple[list[float], list[float]]], epochs: int = 1000, learning_rate: float = 0.1) -> None:
        for epoch in range(epochs):
            for x, y in data:
                self.backpropagate(x, y, learning_rate)

    def predict(self, x: list[float]) -> list[float]:
        output, _ = self.feedforward(x)
        return output[-1]

### TESTING OUR NETWORK
# training data : Basic OR function with y as the second item for each row
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [1])
]

# network with two inputs, one hidden layer with 2 neurons and one output
net = NeuralNetwork([2, 2, 1])


net.train(training_data, epochs=1000, learning_rate=0.5)

# testing our network
for x, y in training_data:
    pred = net.predict(x)
    print(f"Input: {x}, Predicted: {pred}, Expected: {y}")


# Author GCreus
# Done via pyzo
