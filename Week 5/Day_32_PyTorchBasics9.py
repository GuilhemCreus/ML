### Day 32 -- Basics of PyTorch 9
"""
Neural networks learn via backpropagation, which depends on the flow of gradients from the output layer back to the input layer
If the inputs and weights are not properly scaled this gradient flow can be disrupted, causing two main issues : Exploding and Vanishing gradients.
Since we have looked at solving the "input" side of the problem yesterday, we will now look at adjusting the weights

-Exploding gradients:
    -if weights values are very large, the activations can grow exponentially
    -this leads to very large gradients during backpropagation, which destabilizes training

-Vanishing gradients:
    -if weights values are too small or centered far from zero, activations might saturate (this is the case for sigmoid or tanh for example) or get stuck in regions where their derivatives are zero (especially with ReLU)
    -this leads to gradients becoming too small, so the model stops learning effectively

Solution (for ReLU as an example) :
Initialize biases to a small value > 0 (e.g., 0.1)
This:
-encourages neurons to be active from the start
-reduces the risk of "dying ReLU"
-can accelerate learning

For weights :
For ReLU activations, half of the inputs become zero (negative values are cut off), so this means the output variance is roughly half the input variance
To keep the variance stable across layers, we need to compensate for this reduction

Kaiming initialization sets the variance of the weights to  2/fan_in (where fan_in is the number of inputs to the layer) which doubles the variance of the weights
This compensates for the halving effect of ReLU and ensures the activations do not vanish and neurons don't die because if the input follows a normal distribution(0, sigma**2), Var(output) = Var(ReLU(value in neuron)) = 1/2 * Var(value in neuron)
And Var(value in neuron) = fan_in * Var(w) * Var(input)
So  Var(value in neuron) = fan_in * Var(w) * sigma**2
If we want Var(output) equal to 1, then Var(value in neuron) must equal 2

So by scaling data in such a way that sigma = 1, we have Var(w) = 2/fan_in

By selecting the weights in such a way that Var(w) = 2/fan_in for each layers, we could enhance stability and improve the learning of our network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# data
X_np = np.linspace(-5, 5, 200).reshape(-1, 1)
X_mean = X_np.mean()
X_std = X_np.std()
X_np_scaled = (X_np - X_mean) / X_std
X = torch.tensor(X_np_scaled, dtype=torch.float32)

y_np = np.exp(-X_np) * np.sin(5 * X_np)
y = torch.tensor(y_np, dtype=torch.float32)

### NETWORK CLASS WITH CUSTOM BIAS INITIALIZATION
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(1, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        # initialization
        for layer in [self.linear1, self.linear2, self.linear3]:
            # weights : He init
            # we tell torch to initialize the weights for the layer as explained above (with a mean of 0 also)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            # biais initialization > 0
            nn.init.constant_(layer.bias, 0.05)

        self.net = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3
        )

    def forward(self, x):
        return self.net(x)

### TRAINING FUNCTION
def train_model(model, X, y, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    model.train()
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

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
plt.plot(X_np, y_np, label="True function", color='black', linewidth=2)
plt.plot(X_np, preds, label="Model predictions", linestyle='--')
plt.title("Predictions (with bias init)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

"""
The MSE was around 60 yesterday, now it is only 6

So each improvement takes us closer to more precise predictions

This can be combined with what we have seen already :
-scaled inputs (already done)
-scaled outputs between layers (BatchNorm, already done yesterday)
"""

# Author GCreus
# Done via pyzo
