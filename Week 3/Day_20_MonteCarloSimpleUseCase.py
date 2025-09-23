### Day 20 -- Monte Carlo Simulation in a simple RL problem
"""
We will see now another metod to guide our agent through the grid by using Monte Carlo and not using the Q-Table values
"""

import numpy as np
import random
import matplotlib.pyplot as plt
# The following library is useful to create dict with list as value for keys
from collections import defaultdict

### AN AGENT TRYING TO FIND ITS WAY OUT OF A GRID
# Creating the grid (4x4 2D grid)
states = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3)]

actions = ['up', 'down', 'left', 'right']

# Transitions probability with noise to simulate natural accident
transition_probs = {
    'up': {'up': 0.8, 'left': 0.1, 'right': 0.1},
    'down': {'down': 0.8, 'left': 0.1, 'right': 0.1},
    'left': {'left': 0.8, 'up': 0.1, 'down': 0.1},
    'right': {'right': 0.8, 'up': 0.1, 'down': 0.1},
}

# Movement results
action_effect = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# MC-table : each state will have a score for every action possible
MC = {state: {action: 0.0 for action in actions} for state in states}

returns = defaultdict(list)

# no learning rate, we only do a mean of the reward for an episode
gamma = 0.99     # discount factor which reflects how much the agent takes into account possible future rewards, with 0.99 here, our agent is really concerned about the future
epsilon = 0.15    # exploration probability, i.e this is uncertainty about action chosen by the agent, this isn't the same as uncertainty of the environment defined above within transition_probs which reflects possible natural accidents

"""
With Monte Carlo, the value MC(s, a) is an estimate of the expected return (cumulative discounted reward) obtained after taking action a in state s, based on the episodes experienced

The value is updated using the average of all observed returns for the (state, action) pair across all episodes
For example, for one episode :
Step 1 : No reward
Step 2 : No reward
Step 3 : Reward = 1 so :

G3 = 1
G2 = 0 (reward at step 2) + gamma * 1 (reward at step 3)
...

So :
MC(s, a) = average(G_1, G_2, ..., G_n) only if at one time the agent was at the state s with the action a planned for the episode
(where G_t = reward_t + gamma * reward_{t+1} + gamma^2 * reward_{t+2} + ...)

And, we doe the above for each episode

We store all value for each (state, action) pair in a list called "returns", and after each episode we add the new return and recompute the average to update the MC-table

So if the pair (s, a) has a score of 0.8 and that during an episode the agent followed the same couple (s, a) at a moment, we update the value (s, a) in the MC table by doing a mean of the value of (s, a) in the table and the value of (s, a) observed in the episode

Unlike Q learning, Monte Carlo principally differs because :
-there is no estimate of the next state’s value involved in the update, we observe the reward for each pair (s, a) that represented the agent at one time in the episode based on the actual reward obtained, not the expected reward
-updates happen only at the end of episodes

By doing enough episodes, in theory, each pair (s, a) should get closer to their expected reward for that pair, hence the name Monte Carlo because it uses the law of large numbers to explain this convergence
"""

def choose_action(state):
    """
    Arg :
        -state : the current state of the agent

    Return the action that the agent should take, its depend on the willingness of the agent to explore (epsilon) and on the best known action (MC-table)
    """
    if random.random() < epsilon:
        return random.choice(actions)  # the agent might choose to explore randomly
    else:
        max_value = max(MC[state].values())
        best_actions = [a for a, v in MC[state].items() if v == max_value]
        return random.choice(best_actions)
        # the agent might choose the best possible action for a given state, i.e the action with the greatest score in the MC-table for the current state
        # here we explicitly tell the agent to chose randomly between the actions with the same score in the MC table to force him to try different actions even in exploitation mode



def step(state, action):
    """
    Args :
        -state : the current state of the agent
        -action : what action is intended by the agent according to the policy

    Return a tuple (new_state, reward) with :
        -new_state : the new state of the agent after the step
        -reward : a reward for the agent if he reaches the bottom right corner of the grid
    """
    moves = list(transition_probs[action].items()) # A list containing the possible moves for an action and their probability
    move_choices, probs = zip(*moves) # Unpack the moves and their probability in two lists
    chosen_move = random.choices(move_choices, weights=probs)[0]

    dx, dy = action_effect[chosen_move]
    new_state = (state[0] + dx, state[1] + dy)

    # The agent cannot go out of the grid
    if new_state not in states:
        new_state = state  # The agent will bounce back to its current position

    # Reward could follow a normal distribution once the agent has reached the way out in order to simulate the uncertainty of the environment even when everything has been done correctly
    # Here we will simply choose a determinist reward of 1
    reward = 1 if new_state == (3, 3) else 0

    return new_state, reward

def generate_episode(start_state, max_steps=200):
    """
    Args :
        -start_state : the initial state of the agent
        -max_steps : maximum amount of step the agent could do in one simulation, 200 by default

    Return a full episode containing each move (state + action) of the agent through it with the final reward obtained (or not)
    """
    episode = []
    state = start_state
    reward = 0

    for _ in range(max_steps):
        action = choose_action(state)
        state, reward_temp = step(state, action)
        episode.append((state, action, reward_temp))

        if reward_temp == 1:

            break

    return episode

def monte_carlo_learning(episodes=6000):
    for _ in range(episodes):
        episode = generate_episode((0, 0))

        G = 0
        for t in reversed(range(len(episode))):
            # using reversed enables us to go through G(result) to G(1) from the last step to the first step
            # it makes easier the calculation of G(i) by going backward

            state, action, reward = episode[t]
            G = reward + gamma * G

            returns[(state, action)].append(G)
            MC[state][action] = np.mean(returns[(state, action)])
            # hence the usefulness of a dict with list value because of np.mean()

monte_carlo_learning()

def extract_policy(MC):
    policy = {}
    for state in states:
        if state == (3, 3):
            policy[state] = 'E'  # exit of the grid
        else:
            best_action = max(MC[state], key=MC[state].get)
            policy[state] = best_action
    return policy

def print_policy(policy):
    arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'E': 'E'}
    grid = [['' for _ in range(4)] for _ in range(4)]

    for (i, j) in states:
        action = policy[(i, j)]
        grid[i][j] = arrow[action]

    for row in grid:
        print(' '.join(row))

policy = extract_policy(MC)

print_policy(policy)

"""
MC is sometimes better because it is unbiased, we use observation and not estimate to update the value table
MC is also less hard to implement in complex environment because we only need observations and averages
"""

# Author GCreus
# Done via pyzo
