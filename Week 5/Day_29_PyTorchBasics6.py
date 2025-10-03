### Day 29 -- Basics of PyTorch 6
"""
In this script, we are going to try to fit a NN model to a simple function :
y = sin(x)

Through this case, we will show that a deep network performs better than a wide network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# y = sin
X = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100).unsqueeze(1)
y = torch.sin(X)

### WIDE VS DEEP
## WIDE NETWORK
class WideNet(nn.Module):
    def __init__(self):
        super(WideNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40),  # wide layer
            nn.Sigmoid(),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        return self.net(x)

## DEEP NETWORK
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x)

## TRAINING FUNCTION
def train_model(model, name, epochs=500, lr=0.01):
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


## TESTING BOTH NETWORKS
wide_model = WideNet()
deep_model = DeepNet()
epochs = 500
lr = 0.01

wide_losses, wide_preds = train_model(wide_model, "LargeNet", epochs = epochs, lr = lr)
deep_losses, deep_preds = train_model(deep_model, "DeepNet", epochs = epochs, lr = lr)

plt.close('all')
plt.figure(figsize=(14, 8))

# loss plot
plt.subplot(1, 2, 1)
plt.plot(wide_losses, label="Wide Network")
plt.plot(deep_losses, label="Deep Network")
plt.title("Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# predictions
plt.subplot(1, 2, 2)
plt.plot(X.numpy(), y.numpy(), label="sin(x)", color='black')
plt.plot(X.numpy(), wide_preds.numpy(), label="Wide Network", linestyle='--')
plt.plot(X.numpy(), deep_preds.numpy(), label="Deep Network", linestyle='dotted')
plt.title("Predictions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

"""
With this configuration, generally the deep network ends up being more precise than the wide network

This could be due to the fact that a deeper network can learn multiple features of the model instead of having only 1 layer that attempts to capture all features at once

Also, deeper networks are less exposed to overfitting because each layer cannot learn overly complex patterns in order to produce useful representations for the next layer

Last but not least, having multiple layers enables us to have multiple activation functions in the network and this can be useful,  this allows the network to learn under specific constraints :
-learn fast with ReLU (or LeakyReLU if we have mainly negative inputs) in the first layer
-send a result between -1 and 1 with Tanh

Finally, deeper networks offer more flexibility and expressive power which (when combined with proper training techniques) often leads to better generalization
"""

# Author GCreus
# Done via pyzo
