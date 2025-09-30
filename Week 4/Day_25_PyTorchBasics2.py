### Day 25 -- Basics of PyTorch 2
"""
Before going deeper into PyTorch, let's take a quick look at autograd, the hooks and some activation functions to know
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
### AUTOGRAD
"""
Autograd is PyTorch's automatic differentiation engine
So basically autograd is a graph that follows each tensor that needs to be observed

Every time a tensor that is being tracked is involved in an equation, autograd manages to calculate the associated gradient of the tensor regarding that equation when needed by calling the backward() method
This is why autograd is a graph because when we call .backward(), PyTorch automatically computes the gradients of the output with respect to all tracked inputs using this graph by going backward (i.e PyTorch traverses the dynamic computation graph in reverse order), hence the name backward

In order to let PyTorch know that a tensor should be tracked, we specify the argument : requires_grad to True when declaring a tensor
"""
# we initialize a tensor with "requires_grad=True" to let autograd know that this tensor should be tracked for gradient calculation
x = torch.tensor([2.0], requires_grad=True)

# simple function
y = x**2 + 3*x + 1

# we call autograd on this function
y.backward(retain_graph=True)
# when called, it will delete the graph associated to the tensor to free memory
# by setting the argument "retain_graph" to True PyTorch will keep it for our next example

print(f"x: {x.item()}")
print(f"y: {y.item()}")
print(f"dy/dx (gradient): {x.grad.item()}\n")  # dy/dx = 2x + 3 = 2*2 + 3 = 7

"""
Now let's see the graph in action by creating another function that depends on x but indirectly
"""
# z depends on y, hence on x
z = 2 * y + 5

# we reset the grad of x because if we call z.backward, x.grad += dz/dx = dy/dx + dz/dx
x.grad.zero_()

# we call autograd on this function
z.backward()
# without setting the argument "retain_graph" to True, PyTorch will delete the graph

print(f"x: {x.item()}")
print(f"y: {y.item()}")
print(f"z: {z.item()}")
print(f"dz/dx (gradient): {x.grad.item()}")  #dz/dx = d(2y+5)/dx = 2 * dy/dx = 2 * 7 = 14

### HOOKS
## GRADIENT HOOK
"""
Hooks are callback functions that are called when certain conditions are met
We will see hooks directly linked on tensor related to gradient (gradient hook on tensor, this hook is related to the autograd system)
Then hooks linked to forward pass or backward pass (these hooks are related to nn.Module)

Here is a quick example of a gradient hook
"""
# same tensor again
x = torch.tensor([2.0], requires_grad=True)

def grad_hook(grad):
    print(f"Hook triggered, original gradient : {grad}\n")
    return grad * 3  # we simply triple the gradient associated with the tensor whatever the function associated

# we properly handle the hook with a handler
hook_handle = x.register_hook(grad_hook)

# simple function
y = x**2

# we call autograd on this function
y.backward()

print(f"Final gradient of x, dy/dx : {x.grad.item()}")  # 2x * 3 = 8

# removing properly the hook
hook_handle.remove()

## NN.MODULE HOOKS
"""
Now let's see a case of hooks associated with forward/backward pass of layers
"""
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # y = wx + b

    def forward(self, x):
        return self.linear(x) # y = x can be considered an activation function

model = SimpleNetwork()

x = torch.tensor([[1.0]], requires_grad=True)

target = torch.tensor([[2.0]])

# hook on linear layer for debugging for example
def forward_hook(module, input, output):
    print(f"Forward Hook: input={input}, output={output}")

def backward_hook(module, grad_input, grad_output):
    print(f"Backward Hook: grad_input={grad_input}, grad_output={grad_output}")

# properly handling the hooks
fwd_handle = model.linear.register_forward_hook(forward_hook)
bwd_handle = model.linear.register_full_backward_hook(backward_hook)

# forward pass
print("\nWe will forward pass the input in the network")
output = model(x)

# loss function
loss = ((output - target) ** 2).mean()

# backward gradient calculation regarding the loss function
print("\nWe will backward pass in the network to calculate the gradients")
loss.backward()

# properly removing the hooks
fwd_handle.remove()
bwd_handle.remove()

"""
We see that these hooks are activated following certain conditions
We can use hooks to modify a certain aspect of the network for example, or simply for debugging
"""

### IMPORTANT ACTIVATION FUNCTION
"""
Activation functions introduce non-linearity in neural networks which allows neural networks to learn complex relationships beyond simple linear relations

Below are some of the most commonly used activation functions in PyTorch :
"""

# input values to visualize activations
x_vals = torch.linspace(-10, 10, 100)

# apply activation functions
activations = {
    "ReLU": F.relu(x_vals),
    "Leaky ReLU": F.leaky_relu(x_vals, negative_slope=0.1),
    "Sigmoid": torch.sigmoid(x_vals),
    "Tanh": torch.tanh(x_vals),
    "ELU": F.elu(x_vals),
    "GELU": F.gelu(x_vals)
}

# we plot all activations
plt.close('all')
plt.figure(figsize=(14, 8))

for i, (name, act) in enumerate(activations.items(), 1):
    # i : the index of the element
    # name : the name of the function
    # act : a tensor of values taken by the function for x_vals

    plt.subplot(2, 3, i)

    # we use .numpy() to convert the tensor to a numpy Array
    # likewise for the each function values
    plt.plot(x_vals.numpy(), act.numpy())
    plt.title(name)
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.show()

"""
Quick comment on ReLU = max(0, x)
It is very easy to compute but when x < 0, the neuron can "die" because no gradient stems from x constant 0 and so no more learning is possible

Leaky ReLU solves that !
Leaky ReLU = x if x > 0
             alpha * x otherwise with alpha very small
With this function, the neuron can still learn even if x < 0 but slowly

GELU and ELU are a bit more complex and probably out of scope for our reinforcement learning problem
"""

# Author GCreus
# Done via pyzo
