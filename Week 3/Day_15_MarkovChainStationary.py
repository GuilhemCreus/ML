### Day 15 -- Introduction to Markov Chains pt2
"""
Yesterday we have seen the basics of Markov Chains
Today we are going to see the last part that we didn't see yesterday which is solving the equation v = P * v in order to find the stationary vector
"""

import numpy as np
import random

### EQUATION SYSTEM

import numpy as np

P = np.array([
    [0.7, 0.3],
    [0.5, 0.5]
])

# Transpose to have the following equation  P.T v = v
A = P.T - np.eye(2)
# We add the constraint that the sum of values of v must add up to 1
A = np.vstack([A, np.ones(2)])
b = np.array([0, 0, 1])

# We resolve using linalg which is a solver for systems of equations
stationary = np.linalg.lstsq(A, b, rcond=None)[0]
print("Stationary distribution:", stationary)
