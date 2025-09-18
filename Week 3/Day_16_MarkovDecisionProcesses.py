### Day 15 -- Markov Decision Process
"""
So far, our Markov Chain model considered that the transition between states only depends on the current state.
However, in many decision-making problems the transition also depend on the action taken at a state

In such cases, we switch from Markov Chains to a Markov Decision Process, i.e a MDP

In an MDP, the transition is no longer defined by P(s' | s) but rather by P(s' | s, a) where:
-s is the current state
-a is the action taken
-s' is the next state

That means that the action affects the transition probabilities
"""

import numpy as np
import random

### AN AGENT TRYING TO FIND ITS WAY OUT OF A GRID
"""
For the moment, we will focus on the basic components of a Reinforcement Learning problem:

1. The agent :
    -is the entity that takes actions in the environment
    -in our case, the agent is trying to find a way out of the 4x4 grid

2. The reward :
    -is the signal given to the agent to indicate how well it is doing.
    -the agent receives a reward of 1 only when it reaches the goal state

3. The policy (most important part) :
    -is the strategy the agent uses to choose actions based on the current state
    -for now, we use a completely random policy, i.e the agent picks an action randomly, without learning
    -more advanced RL algorithms will improve this policy over time based on past experiences and rewards

And we will create the link to MDP at then end of our code
"""


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

    # Reward could follow a normal distribution ounce the agent has reached the way out in order to simulate the incertainty of the environment even when everything has been done correctly
    # Here we will simply chose a determinist reward of 1
    reward = 1 if new_state == (3, 3) else 0
    print("End reached\n") if reward == 1 else None

    return new_state, reward

def simulate_episode(start_state, max_steps=100):
    """
    Args :
        -start_state : the initial state of the agent
        -max_steps : maximum amount of step the agent could do in one simulation, 100 by default

    Return nothing but follows each action of the agent on the grid
    """
    state = start_state
    total_reward = 0

    for _ in range(max_steps):
        action = random.choice(actions)  # completely random policy
        next_state, reward = step(state, action)
        print(f"State: {state}, Action: {action}, Next: {next_state}, Reward: {reward:.2f}")
        total_reward += reward
        state = next_state

    print(f"Total reward: {total_reward:.2f}")

# Start the simulation
simulate_episode(start_state=(0, 0), max_steps=100)

"""
In the simulation above we have seen that an agent navigating in a 4x4 grid choosing random actions and receiving rewards
This setup corresponds to the structure of a MDP which consists of:

    1) A set of states S:
    in our case all the positions on the 4x4 grid like (0,0)

    2) A set of actions A:
    ['up', 'down', 'left', 'right']

    3) A transition function P(s' | s, a):
    this function defines the probability of moving to state s' when taking action a in state s

    4) A reward function R(s, a, s'):
    this function returns the reward obtained when transitioning from state s to s' via action a

    5) A policy π(a | s):
    the agent's behavior, i.e. how it selects actions

So, we can summarize our environment as an MDP:

    MDP = (S, A, P, R, π)

This is the foundation of all Reinforcement Learning algorithms.

Now, what makes it a Markov process?

Because the transition to the next state depends only on the current state and the action taken and not on any past action
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)

This is called the Markov property

In our case:
-at each time step, the agent chooses an action based only on the current state.
-the environment responds with a new state and a reward, according to P(s' | s, a) and R(s, a, s').

This clearly matches the definition of a Markov Decision Process
"""

# Author GCreus
# Done via pyzo
