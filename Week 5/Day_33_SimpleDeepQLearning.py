### 33 -- Simple Deep Q Learning
"""
Let's now use our knowledge of PyTorch to construct a deep Q learning
In deep Q learning, a neural network tries to predict the Q value correctly instead of using MonteCarlo or a Q table

A deep Q learning is most useful when dealing with complex problems where the number of states is enormous
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


### DEFINING THE ENVIRONMENT
"""
We create the grid, the transitions with stochastic noise and the actions one more time
"""
states = [(i, j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']
action_index = {a: i for i, a in enumerate(actions)}
index_action = {i: a for i, a in enumerate(actions)}

transition_probs = {
    'up': {'up': 0.8, 'left': 0.1, 'right': 0.1},
    'down': {'down': 0.8, 'left': 0.1, 'right': 0.1},
    'left': {'left': 0.8, 'up': 0.1, 'down': 0.1},
    'right': {'right': 0.8, 'up': 0.1, 'down': 0.1},
}

action_effect = {
    'up': (-1, 0), 'down': (1, 0),
    'left': (0, -1), 'right': (0, 1)
}

### FUNCTIONS
"""
This neural network takes a single grid state as input and outputs a Q-value for each of the 4 possible actions

-input : a 2D normalized position (i, j), scaled between 0 and 1
-output : a tensor of shape (1, 4) corresponding to the predicted Q-values for ['up', 'down', 'left', 'right']

During training, we compute the MSE between:
-the predicted Q-value for the action that was actually taken (extracted using .gather), this Q-value represents the expected reward if the agent follows this action and land at "next_state"
-and the target Q-value computed from the reward and the max Q-value of the "next_state", following the Bellman equation:
    target = reward + gamma * max(Q(next_state))   (if not terminal)

This network learns to approximate the optimal Q-function Q(s, a) of the  environment
"""
# ----- Deep Q-Network Model -----
class DQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ----- Convert state to tensor -----
def state_to_tensor(state):
    norm_state = [state[0] / 3.0, state[1] / 3.0]  # as we have seen, we normalize the inputs
    return torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0)
    # unsqueeze(0) turns the tensor into a batch (batch_size, inputs) because this the type expected in a network


# ----- Epsilon-greedy action selection -----
"""
Here, in a nutshell, the network will take the current state as input and return the index of the best action to take (according to the network)
Or, the agent will explore depending on the value of epsilon
"""
def choose_action(model, state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            q_values = model(state_to_tensor(state)).numpy().flatten()
            # .flatten() turns an array into a 1D array
            max_q = np.max(q_values)
            best_actions = [index_action[i] for i, q in enumerate(q_values) if q == max_q]
            # if two actions lead to the same expected reward, we choose randomly between the two
            return random.choice(best_actions)

# ----- Environment step -----
def step(state, action):
    reward = 0
    moves = list(transition_probs[action].items())
    move_choices, probs = zip(*moves)
    chosen_move = random.choices(move_choices, weights=probs)[0]
    dx, dy = action_effect[chosen_move]
    new_state = (state[0] + dx, state[1] + dy)
    if new_state not in states:
        new_state = state
    if new_state == (3, 3):
        reward = 1
    return new_state, reward

# ----- Extract learned policy -----
def extract_policy_from_model(model):
    policy = {}
    for state in states:
        if state == (3, 3):
            policy[state] = 'E'
        else:
            with torch.no_grad():
                q_values = model(state_to_tensor(state))
                best_action_idx = q_values.argmax().item()
                policy[state] = index_action[best_action_idx]
    return policy

# ----- Print policy grid -----
def print_policy(policy):
    arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'E': 'E'}
    grid = [['' for _ in range(4)] for _ in range(4)]
    for (i, j) in states:
        action = policy[(i, j)]
        grid[i][j] = arrow[action]
    for row in grid:
        print(' '.join(row))

### HYPERPARAMETERS, INITIALIZATION AND TRAINING
gamma = 0.90
epsilon = 0.18
episodes = 600
max_steps = 300
lr = 0.1

policy_net = DQN()
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ----- Training loop -----
for episode in range(episodes):
    state = (0, 0)
    for step_count in range(max_steps):
        action = choose_action(policy_net, state)
        next_state, reward = step(state, action)
        done = reward == 1

        # preparing tensor for training on this single transition
        state_tensor = state_to_tensor(state)
        next_state_tensor = state_to_tensor(next_state)
        action_tensor = torch.tensor([[action_index[action]]])
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        done_tensor = torch.tensor([[done]], dtype=torch.float32)

        # we gather the prediction of q for the action that was taken
        q_values = policy_net(state_tensor).gather(1, action_tensor)

        # we compute the target q value using the bellman equation
        # target = reward + gamma * max(Q(next_state)) if not terminal
        with torch.no_grad():
            q_next = policy_net(next_state_tensor).max(1)[0].unsqueeze(1)
            q_target = reward_tensor + gamma * q_next * (1 - done_tensor)

        loss = loss_fn(q_values, q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # this is actually like in Q table, but here everything is predicted using a neural network, so it is possible that the errors spread through the whole network because even the target used to train the network is calculated via the network itself

        state = next_state
        if done:
            break

# ----- Display final policy -----
policy_dqn = extract_policy_from_model(policy_net)
print_policy(policy_dqn)

"""
I took extra time for this one because I had troubles with convergence
I simply forgot to scale the inputs
"""

# Author GCreus
# Done via pyzo
