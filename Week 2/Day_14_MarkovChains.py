### Day 14 -- Introduction to Markov Chains
"""
Today we are going to take a quick view at Markov Chains with an exemple
A Markov chain is a stochastic process where the probability of moving to the next state depends only on the current state, not on the sequence of events that preceded it

Markov Chains are useful tools to deal with conditional events
"""

import numpy as np
import random

### BASIC CASE
"""
For example, let's take the following case as our study case

Imagine the weather has only 2 states : Sunny or Rainy
If it's sunny today, there is 70% chance that it will be sunny tomorrow
If it's sunny today, there is 30% chance that it will be rainy tomorrow
If it's rainy today, there is 50% chance that it will be sunny tomorrow
If it's rainy today, there is 50% chance that it will be rainy tomorrow

We can define a transition matrix that summarize the information above
transition matrix =[
    [0.7, 0.3],
    [0.5, 0.5]]

So if we are at state Sunny, we can look at the first row (index 0 of matrix), generate a random float between 0 and 1, like r = 0.73 for example
Then, we will go through each probability in the row and see if the SUM of the probability we have seen yet is greater than r

First iteration : we have SUM = 0 + 0.7 which is less than r
Second iteration : we have SUM = 0.7 + 0.3 = 1 which is higher than r so we select the second state as the next state

And we do the same thing until we have enough data
"""

### PYTHON IMPLEMENTATION
states = ["Sunny", "Rainy"]
transition_matrix = [
    [0.7, 0.3],  # From Sunny
    [0.5, 0.5]]  # From Rainy

# A dict to associate a state to its index
state_to_index = {state: i for i, state in enumerate(states)}

# Function to simulate weather over n days
def markov_weather_simulation(start_state, days):
    current_state = start_state
    weather_sequence = [current_state]

    for _ in range(days):
        current_index = state_to_index[current_state]

        # We select a random state with respect to the probability in the row of the current state
        next_state = random.choices(
            states,
            weights=transition_matrix[current_index]
        )[0]
        weather_sequence.append(next_state)
        current_state = next_state

    return weather_sequence

sequence = markov_weather_simulation(start_state="Sunny", days=10)
print("Weather forecast:", sequence)

### SIMPLE IMPROVEMENT
"""
Instead of looping inside the transition matrix n times, we can simply calculate the power n of transition matrix to get the transition probability after n steps with (i, j) giving the probability of going to state j from state i in n steps
"""

P = np.array([
    [0.7, 0.3],
    [0.5, 0.5]
])

n = 5
Pn = np.linalg.matrix_power(P, n)
print(Pn)
print(f"So the probability to go from state Sunny (index 0) to state Rainy (index 1) in {n} steps is {Pn[0][1]}")

### BASIC OPERATION WITH A MARKOV CHAIN
"""
Now let's assume that today, there is 80% chance that it will be Sunny and 20% chance that it will be Rainy
We represent this with the following vector :
v =[
    [0.8],
    [0.2]]

Knowing these probabilities, how can we know the probability of the same events tomorrow ?
We simply do P * v
"""
v = np.array([
    [0.8],
    [0.2]
])

Pv = P @ v

print(f"Knowing that today there is {v[0]} chance of sun and {v[1]} of rain,\nThen tomorrow, there is {Pv[0]} chance of sun and {Pv[1]} of rain")

### STATIONARY IN MARKOV CHAINS
"""
With the above notion in mind, we can think of a distribution of probability that will never change when multiplied to the transition matrix
In other term, this is the distribution of probability that is 'stable' with respect to the transition matrix, no matter how much time we multiply the transition matrix to the distribution, it will remain the same
And, no matter what initial vector of distribution of probability, after a huge amount of step, the vector will converge to the stationary vector

We can find this distribution by solving v = P * v

We will work on that tomorrow


In reinforcement learning, this concept is really useful because it can help us find the behaviour of an agent after a long time (huge amount of step) under a given policy
"""

# Author GCreus
# Done via pyzo
