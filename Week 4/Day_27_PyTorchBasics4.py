### Day 27 -- Basics of PyTorch 4
"""
This code is the following of the yesterday work where we have seen two majors algorithms related to the optim library

Today, we are going to see the other two
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
### RMSPROP
"""
With ADAgrad we have :
w(i + 1, j) = w(i, j) - learning rate * 1/[epsilon + sqrt(SUM velocity(i + 1, j)] * grad(w(i, j))
With : velocity(i + 1, j) = velocity(i, j) + grad(w(i, j))**2

This method updates the learning rate by decreasing it over time for parameters that receive larger gradients, and remains relatively higher for those with smaller or infrequent updates

The major issue with this method is the learning rate can only decrease over time
So at some points, the neural network will not be able to learn more

One solution brought by RMSprop (Root Mean Square Propagation) is to forget older gradients in the denominator by using an exponential mobile mean instead of a simple one
With this method :
velocity(t + 1) = Beta * velocity(t) + (1 - Beta) * grad(w(t))**2
Beta (or alpha) : the decay parameter between 0 and 1, it tells us how much of the previous velocity terms is remembered
"""

# x1 (first feature) is constant
# x2 (second feature) increases linearly
x_train = torch.tensor([[1.0, 1.0],
                        [1.0, 2.0],
                        [1.0, 3.0],
                        [1.0, 4.0],
                        [1.0, 5.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])  # y = 2 * x2 + 1

# 1 linear neuron, no bias to focus on weights
model = nn.Linear(in_features=2, out_features=1, bias=False)

# RMSprop optimizer with a learning rate of 1
optimizer = optim.RMSprop(model.parameters(), lr=1.0, alpha = 0.9)

# loss function
loss_fn = nn.MSELoss()

# lists to store the effective learning rate for both weights
effective_lrs_w0 = []  # For weight x1 (constant feature)
effective_lrs_w1 = []  # For weight x2 (varying feature)

# accumulated squared gradients (as used by RMSprop internally)
accumulated_grad_squared = torch.zeros_like(model.weight.data)

# training loop
epochs = 50
for epoch in range(epochs):
    # forward pass
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    # zero gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # manually track RMSprop-like behavior
    with torch.no_grad():
        for param in model.parameters():
            # optimizer.param_groups is a dict containing the parameters and their value
            alpha = optimizer.param_groups[0]['alpha']
            accumulated_grad_squared = alpha * accumulated_grad_squared + (1 - alpha) * param.grad.data ** 2


            # compute per-parameter effective learning rates
            # we go through optimizer.defaults because eps, as a default parameter we did not change, is stored there
            effective_lr = 1.0 / (accumulated_grad_squared.sqrt() + optimizer.defaults['eps'])
            effective_lrs_w0.append(effective_lr[0, 0].item())  # For weight of x1
            effective_lrs_w1.append(effective_lr[0, 1].item())  # For weight of x2

    # optimizer step
    optimizer.step()

# plot effective learning rates
plt.close('all')
plt.figure(figsize=(14, 8))
plt.plot(effective_lrs_w0, label='Learning Rate for weight x1 (constant feature)')
plt.plot(effective_lrs_w1, label='Learning Rate for weight x2 (varying feature)')

plt.xlabel("Epoch")
plt.ylabel("Effective Learning Rate")

plt.title("RMSprop: Per-parameter Learning Rates Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### ADAM (ADAPTIVE MOMENT ESTIMATION)
"""
We saw that SGD use momentum without updating the learning rate dynamically
We also saw that RMSprop updates the learning rate dynamically with stability through time but without momentum

What Adam does is combining the two ideas above to have a method that uses momentum and dynamic learning rate

To do so, Adam keeps track of the velocity (same foruma of veolicty for RMSprop) :
velocity(t + 1) = Beta2 * velocity(t) + (1 - Beta2) * grad(w(t))**2

And it keeps track of momentum :
m(t + 1) = Beta1 * m_(t) + (1 - Beta1) * grad(w(t))

But initially, these two terms would biased toward 0 by the fact that the initial m and vecolity vector are all just zeros
To unbias these terms, the solution was to scale them regarding betas :
-m_hat(t) = m(t) / (1 - Beta1^t)
-v_hat(t) = veolicty(t) / (1 - Beta2^t)

Through time, m_hat and v_hat will get closer to m and velocity
We then update the weights :
w(t+1) = w(t) - lr * m_hat / (sqrt(v_hat) + epsilon)
"""
# x1 (first feature) is constant
# x2 (second feature) increases linearly
x_train = torch.tensor([[1.0, 1.0],
                        [1.0, 2.0],
                        [1.0, 3.0],
                        [1.0, 4.0],
                        [1.0, 5.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])  # y = 2 * x2 + 1

# 1 linear neuron, no bias to focus on weights
model = nn.Linear(in_features=2, out_features=1, bias=False)

# Adam optimizer with a learning rate of 1
optimizer = optim.Adam(model.parameters(), lr=1.0)

# loss function
loss_fn = nn.MSELoss()

# lists to store the effective learning rate for both weights
effective_lrs_w0 = []  # For weight x1 (constant feature)
effective_lrs_w1 = []  # For weight x2 (varying feature)

# Adam vectors initialization
m = torch.zeros_like(model.weight.data)
v = torch.zeros_like(model.weight.data)

# Adam parameters
beta1 = optimizer.param_groups[0]['betas'][0]
beta2 = optimizer.param_groups[0]['betas'][1]
epsilon = optimizer.defaults['eps']

# accumulated squared gradients (as used by Adam internally)
accumulated_grad_squared = torch.zeros_like(model.weight.data)

# training loop
epochs = 50
for epoch in range(epochs):
    # forward pass
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    # zero gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # manually track Adam-like behavior
    with torch.no_grad():
        for param in model.parameters():
            grad = param.grad.data

            # moments calculation
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2

            # bias correction (useful for the first epochs)
            # we add + 1 to the epochs to prevent division by 0
            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))

            #dynamic learning rate update
            effective_lr = 1.0 / (v_hat.sqrt() + epsilon)

            effective_lrs_w0.append(effective_lr[0, 0].item())  # For weight of x1
            effective_lrs_w1.append(effective_lr[0, 1].item())  # For weight of x2

    # optimizer step
    optimizer.step()

# plot effective learning rates
plt.close('all')
plt.figure(figsize=(14, 8))
plt.plot(effective_lrs_w0, label='Learning Rate for weight x1 (constant feature)')
plt.plot(effective_lrs_w1, label='Learning Rate for weight x2 (varying feature)')

plt.xlabel("Epoch")
plt.ylabel("Effective Learning Rate")

plt.title("Adam: Per-parameter Learning Rates Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Author GCreus
# Done via pyzo
