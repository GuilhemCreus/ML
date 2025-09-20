### Day 18 -- Basic Decision Policy Implementation
"""
Today we simply try to get the decision policy that stems from the Q-Table
To do so, we simply take the same code as yesterday and we implement a policy function
"""

import numpy as np
import random

### AN AGENT TRYING TO FIND ITS WAY OUT OF A GRID
"""
The policy tells the agent what action it should take at each cell in the grid to maximize its chances of reaching the goal efficiently

So the policy directly stems from the Q-Table, the policy is simply a quick recap (in our case) of the optimal step for each state
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

# Q-table : each state will have a score for every action possible
Q = {state: {action: 0.0 for action in actions} for state in states}

alpha = 0.1      # learning rate
gamma = 0.99     # discount factor which reflects how much the agent takes into account possible future rewards, with 0.99 here, our agent is really concerned about the future
epsilon = 0.1    # exploration probability, i.e this is uncertainty about action chosen by the agent, this isn't the same as uncertainty of the environment defined above within transition_probs which reflects possible natural accidents

"""
With all these values in mind, let's explain how Q-table is calculated for each state and each action :
Q(s, a) <- Q(s, a) + alpha * [immediate_reward + gamma * max​Q(s', a') - Q(s, a)]

With max​Q(s', a') the maximum expected reward for state s' which depends on the action a' taken
So max​Q(s', a') is the expected reward for state s' if the agent choose the best action a' at the state s'

So, for each step, Q(s, a) will be updated according to its current value and the difference between (immediate reward + expected reward at next state if the agents choose the best action at the next state) and the current expected reward Q(s, a)

Let's reformulate that again, so the fact that an action a at a state s will move the agent to a state where he will be closer to reward will be able to improve the potential reward at the state s for that action a
Think of it like : the reward state will spread its potential reward to its neighbor states that can reach it with actions, which will themselves spread to theirs neighbor states...
"""

def choose_action(state): # last time the action chosen was completely random
    """
    Arg :
        -state : the current state of the agent

    Return the action that the agent should take, its depend on the willingness of the agent to explore (epsilon) and on the best known action (Q-table)
    """
    if random.random() < epsilon:
        return random.choice(actions)  # the agent might choose to explore randomly
    else:
        return max(Q[state], key=Q[state].get)  # the agent might choose the best possible action for a given state, i.e the action with the greatest score in the Q-table for the current state



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

def q_learning_episode(start_state, max_steps=100):
    """
    Args :
        -start_state : the initial state of the agent
        -max_steps : maximum amount of step the agent could do in one simulation, 100 by default

    Return the reward (so 0 or 1) for one episode and update the Q-table
    """
    state = start_state
    reward = 0

    for _ in range(max_steps):
        action = choose_action(state)
        next_state, reward_temp = step(state, action)

        # Q-learning update, the main idea here is to update Q(s, a) by adding alpha * (reality - prediction)
        # With reality : immediate reward + future reward
        # Prediction : current value of Q(s, a)
        best_next_action = max(Q[next_state], key=Q[next_state].get)
        # best_next_action simply select action with the highest score in the Q-table at the next state

        reality = reward_temp + gamma * Q[next_state][best_next_action]
        # reality computes the immediate reward carried by the agent and the expected reward he can achieve by going in the next state chosen and by chosing the best_next_action, discounted by gamma

        reality_less_prediction = reality - Q[state][action]
        # reality_less_prediction is simply the difference between what the agent expect to obtain regarding what reward he can carry if he follows best_NEXT_action in its NEXT state and the actual predicted reward he expect to carry if he stay in its CURRENT state with its CURRENT action

        Q[state][action] += alpha * reality_less_prediction

        state = next_state
        reward = reward_temp

        if reward_temp == 1:  # goal reached !! waouhh
            break

    return reward

episodes = 60000
total_reward = 0

for i in range(episodes):
    total_reward += q_learning_episode((0, 0))

print(f"Mean reward per episode is {total_reward/episodes:.2f}")

def extract_policy(Q):
    """
    Args:
        Q: The Q-Table for the Grid Problem

    Returns:
        policy: a dict that links each state with the optimal action at that state
    """
    policy = {}
    for state in Q:
        best_action = max(Q[state], key=Q[state].get)
        policy[state] = best_action
    return policy

"""
We now print the policy for our Grid Problem with a simple loop
"""
policy = extract_policy(Q)

print("Policy (best action for each state):\n")
for i in range(4):
    for j in range(4):
        state = (i, j)
        action = policy[state]
        print(f"{action:^6}", end=' ')
    print()

# Author GCreus
# Done via pyzo
