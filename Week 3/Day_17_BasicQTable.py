### Day 15 -- Markov Decision Process
"""
Yesterday, we have introduced the basics of MDP and today, we are going to work on the same example as yesterday but, this time we are going to explore how the agent will learn within the environment
"""

import numpy as np
import random
import matplotlib.pyplot as plt

### AN AGENT TRYING TO FIND ITS WAY OUT OF A GRID
"""
In this version, we will go a step further and implement a learning mechanism based on Q-learning, one of the most popular Reinforcement Learning algorithms
We will mainly on focus on the two following :

1) The learning process (Q-learning):
    - the agent maintains a Q-table which estimates the expected reward of each action at each state
    - the agent balances exploration and exploitation based on epsilon
    - after each step, the Q-value is updated using the reward and the estimated future rewards

2) The policy:
    - is now learned from data, instead of being random
    - over time, the agent improves its policy by updating its Q-values, leading to better decision-making
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

episodes = np.linspace(1000, 20000, 20, dtype=int) # episode, not epoch
mean_reward = []

"""
Now, let's see how  our program is able to learn regarding the maximum amount of simulation (i.e the number of episode) he can do to improve its Q-table
We will measure the ability of the agent to learn by looking at the mean amount of reward per episode
We will plot this measure for different maximum number of episode allowed
"""
for i in range(len(episodes)):
    Q = {state: {action: 0.0 for action in actions} for state in states}
    all_rewards = 0
    for ep in range(episodes[i]):
        reward = q_learning_episode(start_state=(0, 0))
        all_rewards += reward
    mean_reward.append(all_rewards/episodes[i])

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(episodes, mean_reward, marker='o')
ax.set_xlabel('Amount of episodes')
ax.set_ylabel('Mean reward per episode')

plt.show()

"""
We clearly see that our program is able to learn now compared to yesterday where the choice of action was completely random

Tomorrow we will see the rest of the basics of RL which is policy
Pretty simple because it derives from Q-table and we have just seen Q-table so easy job
"""

# Author GCreus
# Done via pyzo
