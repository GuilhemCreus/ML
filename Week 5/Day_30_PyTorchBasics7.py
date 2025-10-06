### Day 30 -- Basics of PyTorch 7
"""
We have compared deep vs wide networks with a simple example

Now let's consider another issue when dealing with selecting the hyperparameters and the configuration of a network

In this script, we are comparing different activation functions
by training a simple neural network to approximate the function y = -exp(x) + sin(5x)

This allows us to observe:
- saturation (Sigmoid, Tanh)
- dying neurons (ReLU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

X = torch.linspace(-5, 5, 200).unsqueeze(1)
y = np.exp(-X) * np.sin(5 * X)
### NETWORK CLASS WITH FLEXIBILITY REGARDING ACTIVATION FUNCTIONS
class SimpleNet(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            activation_fn,
            nn.Linear(64, 64),
            activation_fn,
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

# the activation functions we will look
activation_functions = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "ELU": nn.ELU()
}

### TRAINING FUNCTION
def train_model(model, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses, model(X).detach()

### TESTING
# manual seed so you can have the same results as mine
torch.manual_seed(0)
results = {}
epochs = 1000
lr = 0.01

for name, act_fn in activation_functions.items():
    model = SimpleNet(act_fn)
    losses, preds = train_model(model, epochs=epochs, lr=lr)
    results[name] = {
        "losses": losses,
        "preds": preds
    }


plt.close('all')
plt.figure(figsize=(16, 8))

# loss plots
plt.subplot(1, 2, 1)
for name, data in results.items():
    plt.plot(data["losses"], label=name)
plt.title("Loss (MSE) over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# prediction
plt.subplot(1, 2, 2)
plt.plot(X.numpy(), y.numpy(), label="x^3", color='black', linewidth=2)
for name, data in results.items():
    plt.plot(X.numpy(), data["preds"].numpy(), label=name, linestyle='--')
plt.title("Predictions with Different Activations")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()
"""
The above script should print the loss graph through epochs with the final prediction by activation functions

This experiment is also a great way to illustrate two important issues in neural networks: saturation and dead neurons

-Saturation occurs when the input to an activation function falls into a region where its derivative is near zero, this happens with functions like sigmoid and Tanh for large positive or negative values
In such cases, the gradient during backpropagation becomes very small (not zero unline dead neurons), leading to very slow learning, we clearly see this phenomena here because the network struggles to adjust the weights

-Dead neurons are neurons that always output zero and stop learning entirely, this typically happens with ReLU where negative inputs are clipped to zero
If a neuron's weights push it into the negative input range permanently, its output stays at zero, and since ReLU’s gradient is also zero there, no gradient flows back, effectively making the neuron "dead" because the weights will not be updated

These phenomena are directly visible in the training results:
-sigmoid shows high loss and flat predictions due to saturation
-ReLU gets stuck with poor predictions and high loss, likely due to dead neurons
-tanh performs better but still suffers mild saturation at input extremes

Conclusion: How activation functions impact learning

With the complex function y = exp(-x) * sin(5x), we observe the following:

| Activation | Pros | Cons |
|------------|------|------|
| ReLU | Fast on positives | Fails completely on negatives (dead neurons), high loss |

| LeakyReLU | Handles negatives, decent convergence | Spikes in loss curve, possibly unstable |

| Sigmoid | Smooth output | Strong saturation, can't learn high amplitude patterns |

| Tanh | Centered output, decent approximation | Still suffers from saturation in extremes |

| ELU | Handles negatives, flexible, smooth | Some instability in training, better generalization |

Key takeaways:
- Saturation is a major issue for Sigmoid and Tanh in wide input domains
- ReLU is fragile with negative inputs, especially when the task involves values < 0
- LeakyReLU and ELU offer better trade-offs, often yielding more stable and expressive models
- The choice of activation should be guided by the data range, symmetry, and the type of non-linearity needed
"""

# Author GCreus
# Done via pyzo
