import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Grid settings
states = [(i, j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']
action_index = {a: i for i, a in enumerate(actions)}
index_action = {i: a for i, a in enumerate(actions)}

# Transition probabilities with noise
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

# Hyperparameters
gamma = 0.99
epsilon = 0.15
episodes = 400
max_steps = 100
lr = 0.01
batch_size = 64
memory_size = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Convert state to tensor
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

# Choose action with epsilon-greedy
def choose_action(model, state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            q_values = model(state_to_tensor(state)).cpu().numpy().flatten()
            max_q = np.max(q_values)
            best_actions = [index_action[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)


# Step function (unchanged)
def step(state, action):
    moves = list(transition_probs[action].items())
    move_choices, probs = zip(*moves)
    chosen_move = random.choices(move_choices, weights=probs)[0]
    dx, dy = action_effect[chosen_move]
    new_state = (state[0] + dx, state[1] + dy)
    if new_state not in states:
        new_state = state
    reward = 1 if new_state == (3, 3) else 0
    return new_state, reward

# Replay buffer
memory = deque(maxlen=memory_size)

# Initialize networks and optimizer
policy_net = DQN().to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Training loop
for episode in range(episodes):
    state = (0, 0)
    for step_count in range(max_steps):
        action = choose_action_dqn(policy_net, state)
        next_state, reward = step(state, action)

        memory.append((state, action_index[action], reward, next_state, reward == 1))
        state = next_state

        if reward == 1:
            break

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

            states_tensor = torch.tensor(states_b, dtype=torch.float32, device=device)
            actions_tensor = torch.tensor(actions_b, dtype=torch.int64, device=device).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards_b, dtype=torch.float32, device=device).unsqueeze(1)
            next_states_tensor = torch.tensor(next_states_b, dtype=torch.float32, device=device)
            dones_tensor = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = policy_net(states_tensor).gather(1, actions_tensor)

            with torch.no_grad():
                q_next = policy_net(next_states_tensor).max(1)[0].unsqueeze(1)
                q_target = rewards_tensor + gamma * q_next * (1 - dones_tensor)

            loss = loss_fn(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Extract policy from trained DQN
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

def print_policy(policy):
    arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'E': 'E'}
    grid = [['' for _ in range(4)] for _ in range(4)]
    for (i, j) in states:
        action = policy[(i, j)]
        grid[i][j] = arrow[action]
    for row in grid:
        print(' '.join(row))

# Final policy
policy_dqn = extract_policy_from_model(policy_net)
print_policy(policy_dqn)
