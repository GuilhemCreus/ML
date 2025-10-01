### Day 26 -- Basics of PyTorch 3
"""
Now that we have a better understanding of the infrastructure of PyTroch, we are going to see the four main algorithms to optimize the parameters of a neural network in order to reduce the loss function

This will enhance our understanding of the optim library that we will see in a few days
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
### STOCHASTIC GRADIENT DESCENT (SGD)
"""
We have already seen the gradient descent method :
w(i+1) = wi - learning rate * gradLossFunction(w(i))

This method is simple and pretty useful but it has some drawbacks, if the learning rate is too low, this method can take quite a long time to find an optimal set of parameters and it can fall into a local minimum
If the learning rate is too high, the method can miss the global/local minimum because the resolution of its movements is too large compared to the required narrow movements to be able to enter in the minimum

One possible solution to mitigate these issues is to introduce a notion of inertia or momentum in the updates
The idea is to accumulate a velocity vector based on the past gradients and use it to smooth out the updates
This way, the algorithm maintains some of the direction of previous updates (i.e., momentum) helping it to move more consistently in valleys and potentially escape shallow local minima

This leads to SGD with momentum :
w(i+1) = wi + velocity(i+1)
With :
-velocity(i+1) = momentum * velocity(i) - learning rate * gradLossFunction(w(i))
-momentum a parameter that defines how much we keep our previous velocity in the future movements

This is exactly what the method SGD of the optim library :
torch.optim.SGD(our_model.parameters(), lr=0.01, momentum=0.9)

We are going to see it in action in the following example
"""
x_train = torch.tensor([[1.0], [2.0]], requires_grad=False)
y_train = torch.tensor([[3.0], [5.0]], requires_grad=False)
momentum = 0.9
learning_rate = 0.1

# simple linear equation : y = x @ W.T + B
model = nn.Linear(in_features=1, out_features=1)

# simple parameters
with torch.no_grad():
    model.weight.fill_(0.0)  # w = 0
    model.bias.fill_(0.0)    # b = 0

# SGD optimizer with momentum
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# loss_fn is the MSELoss funtion
loss_fn = nn.MSELoss()

# -------- before first training
print("Before the training :")
print(f"Weight w: {model.weight.item():.4f}, Bias b: {model.bias.item():.4f}")

# -------- first training
# forward pass
y_pred = model(x_train)

loss = loss_fn(y_pred, y_train)

# backward pass with SGD
optimizer.zero_grad()
loss.backward()

print("\nGradients :")
print(f"dL/dw: {model.weight.grad.item():.4f}")
print(f"dL/db: {model.bias.grad.item():.4f}")

velocity = 0
velocity = velocity * momentum - learning_rate * model.weight.grad.item()

print(f"\nIn our case, we have :\nvelocity(0) = 0\nvelocity(1) = momentum * velocity(0) - learning rate * gradLossFunction(w(0))\nvelocity(1) = 0.9 * 0 - 0.1 * dL/dw: {model.weight.grad.item():.4f}")

print(f"So velocity(1) = {velocity:.4f}")

# parameters update
optimizer.step()

# -------- after first training
print("\nAfter the FIRST update ---------")
print(f"Poids w: {model.weight.item():.4f}, Biais b: {model.bias.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# -------- second iteration
y_pred = model(x_train)

loss = loss_fn(y_pred, y_train)

# backward pass with SGD
optimizer.zero_grad()
loss.backward()

print("\nGradients :")
print(f"dL/dw: {model.weight.grad.item():.4f}")
print(f"dL/db: {model.bias.grad.item():.4f}")

velocity = velocity * momentum - learning_rate * model.weight.grad.item()
print(f"\nLikewise velocity(2) = {velocity:.4f}")
print(f"\nSo w(1 + 1) = w(1) + velocity(1 + 1)\nw(2) = w(1) + velocity(2)\nw(2) = w : {model.weight.item():.4f} + velocity(2) : {velocity:.4f}")

# parameters update
optimizer.step()

# -------- after second training
print("\nAfter the SECOND update ---------")
print(f"Poids w: {model.weight.item():.4f}, Biais b: {model.bias.item():.4f}")
print(f"Loss: {loss.item():.4f}")

### SGD WITH NESTEROV
"""
SGD with nesterov is simply an improvement of the prevous method with the same idea behind
Nesterov simply takes into consideration the weight and the velocity in the gradient, with this modification, SGD with Nesterov adapts its velocity by anticipating where it will be if it keeps its current velocity at its current position (current position : w, velocity = v)

The equation is now :
w(i+1) = wi + velocity(i+1)
With :
-velocity(i+1) = momentum * velocity(i) - learning rate * gradLossFunction(w(i) + momentum * velocity(i))
-momentum a parameter that defines how much we keep our previous velocity in future movements

We can clearly see that when updating the velocity for each iteration, the method is looking ahead to where its velocity will lead it
By looking ahead, Nesterov momentum can adjust its trajectory before overshooting hence leading to more controlled and (often) faster convergence compared to classical momentum
"""

### ADAGRAD
"""
The essence of this method is that each individual weight and bias now has its own dynamic learning rate
We update the learning rate of each parameter in a way that the parameters that got high  amplitude changes will now be learning less than the parameters that had changes but with less magnitude

In other terms, this learning rate decreases over time for parameters that receive larger gradients, and remains relatively higher for those with smaller or infrequent updates
This is achieved by scaling the learning rate by the square root of the sum of the squares of all previous gradients for each parameter

w(i + 1, j) = w(i, j) - learning rate * 1/[epsilon + sqrt(SUM grad(w(i, j))**2)] * grad(w(i, j))
With :
-i : iteration
-j : the identifier of the weight
-epsilon : small constant added for numerical stability to avoid division by zero

By calling velocity(i + 1, j) = velocity(i, j) + grad(w(i, j))**2
We can write : w(i + 1, j) = w(i, j) - learning rate * 1/[epsilon + sqrt(SUM velocity(i + 1, j)] * grad(w(i, j))

This method enables the loss function to explore other parameters that would be disregarded otherwise
But we can easily see that the learning rate can only decrease over time
So at some points, the neural network will not be able to learn more
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

# adaGrad optimizer with a learning rate of 1
optimizer = optim.Adagrad(model.parameters(), lr=1.0)

# loss function
loss_fn = nn.MSELoss()

# lists to store the effective learning rate for both weights
effective_lrs_w0 = []  # For weight x1 (constant feature)
effective_lrs_w1 = []  # For weight x2 (varying feature)

# accumulated squared gradients (as used by AdaGrad internally)
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

    # manually track AdaGrad-like behavior
    with torch.no_grad():
        for param in model.parameters():
            accumulated_grad_squared += param.grad.data ** 2

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

plt.title("AdaGrad: Per-parameter Learning Rates Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
Because the gradient with respect to w0 isn't multiplied by x1, this gradient should be less than th one associated with w1 which inside contains a multiply factor of x2

So grad(x2) > grad(x1), this is why in our plot the learning rate of x2 is lower than the other one
"""

"""
We will see RMSprop & Adam tomorrow
"""

# Author GCreus
# Done via pyzo
