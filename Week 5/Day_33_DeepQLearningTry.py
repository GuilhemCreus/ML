### 33 -- Simple Deep Q Learning
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----- GridWorld settings -----
states = [(i, j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']
action_index = {a: i for i, a in enumerate(actions)}
index_action = {i: a for i, a in enumerate(actions)}

# Transition probabilities (with stochastic noise)
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

# ----- Hyperparameters -----
gamma = 0.90
epsilon = 0.05
episodes = 400
max_steps = 200
lr = 0.01

# ----- Always use CPU -----
device = torch.device("cpu")

# ----- Deep Q-Network Model -----
class DQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ----- Convert state to tensor -----
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# ----- Epsilon-greedy action selection -----
def choose_action(model, state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            q_values = model(state_to_tensor(state)).numpy().flatten()
            max_q = np.max(q_values)
            best_actions = [index_action[i] for i, q in enumerate(q_values) if q == max_q]
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

# ----- Initialize model and optimizer -----
policy_net = DQN()
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ----- Training loop (no replay buffer) -----
for episode in range(episodes):
    state = (0, 0)
    for step_count in range(max_steps):
        action = choose_action(policy_net, state)
        next_state, reward = step(state, action)
        done = reward == 1

        # Train on this single transition
        state_tensor = state_to_tensor(state)
        next_state_tensor = state_to_tensor(next_state)
        action_tensor = torch.tensor([[action_index[action]]])
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        done_tensor = torch.tensor([[done]], dtype=torch.float32)

        q_values = policy_net(state_tensor).gather(1, action_tensor)

        with torch.no_grad():
            q_next = policy_net(next_state_tensor).max(1)[0].unsqueeze(1)
            q_target = reward_tensor + gamma * q_next * (1 - done_tensor)

        loss = loss_fn(q_values, q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

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

# ----- Display final policy -----
policy_dqn = extract_policy_from_model(policy_net)
print_policy(policy_dqn)
