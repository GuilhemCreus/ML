### Day 31 -- Basics of PyTorch 8
"""
Yesterday we talked about some issues when manipulating activation functions on data that was out of the scope of the activation function

Today, we will illustrate a method to deal with such issue

Now inputs are standardized using the formula:

    X_scaled = (X - mean(X)) / std(X)
    (It's like a StandardScaler)

Scaling the inputs ensures that all features are centered around 0 with a unit variance, it helps because it :
-keeps activations within a reasonable range
-prevents exploding or vanishing gradients
-reduces the risk of "dying ReLU" neurons (neurons stuck forever at 0 output)

Although our target values are not scaled here, scaling the inputs is often enough to stabilize training when using ReLU activations



We also explicitly switch the model between two modes:

  model.train() → sets the network to "training mode"
  model.eval()  → sets it to "evaluation (inference) mode"

These modes control the behavior of specific layers such as Dropout or Batch Normalization:
-in train mode: Dropout is active, and BatchNorm updates its statistics
-in eval mode: Dropout is disabled, and BatchNorm uses its learned stats

Dropout:
-during training, randomly "drops" (disables) some neurons
-this prevents the network from relying too much on specific paths

Batch normalization :
-normalizes activations of each layer during training so that they have mean like 0 and std like 1 for each batch
-this keeps the network stable
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# data
X_np = np.linspace(-5, 5, 200).reshape(-1, 1)

# scaling
X_mean = X_np.mean()
X_std = X_np.std()
X_np_scaled = (X_np - X_mean) / X_std
X = torch.tensor(X_np_scaled, dtype=torch.float32)

y_np = np.exp(-X_np) * np.sin(5 * X_np)
y = torch.tensor(y_np, dtype=torch.float32)

### NETWORK CLASS WITH HE INITIALIZATION FOR RELU
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

### TRAINING FUNCTION
def train_model(model, X, y, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    # enter training mod
    model.train()
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # enter eval mod
    # also preds is directly stored as numpy for plt
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
    return losses, preds

### TESTING
# manual seed so you can have the same results as mine
torch.manual_seed(0)
epochs = 1000
lr = 0.01

model = SimpleNet()
losses, preds = train_model(model, X, y, epochs=epochs, lr=lr)

plt.close('all')
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
# real function
plt.plot(X_np, y_np, label="True function", color='black', linewidth=2)

# preds plot
plt.plot(X_np, preds, label="Model predictions", linestyle='--')
plt.title("Predictions (scaled inputs, unscaled outputs)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

"""
The difference is astonishing with yesterday results
"""

# Author GCreus
# Done via pyzo
