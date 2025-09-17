### Day 15 -- Introduction to Markov Chains pt2
"""
Yesterday we have seen the basics of Markov Chains
Today we are going to see the last part that we didn't see yesterday which is solving the equation v = P * v in order to find the stationary vector
"""

import numpy as np
import random

### EQUATION SYSTEM
"""
Let's supose that the two values in the vector v are : [s, r]
Finding s and r such that v = P * v is the same as resolving the following system :
for P = np.array([
    [0.7, 0.3],
    [0.5, 0.5]])
    
We have P * v =([
    [0.7 * s + 0.3 * r],
    [0.5 * s + 0.5 * r]])

So v = P * v is equivalent as resolving :
s = 0.7 * s + 0.3 * r
r = 0.5 * s + 0.5 * r

Yet we only have two equations for two unknowns, but if we remembre the constraint that s + r = 1, we have now three equations so we can solve the system
"""
import numpy as np

P = np.array([
    [0.7, 0.3],
    [0.5, 0.5]
])

# Transpose to have the following equation  P.T v = v
A = P.T - np.eye(2)
# We add the constraint that the sum of values of v must add up to 1
A = np.vstack([A, np.ones(2)])

# With b, the equation will be the following : A * X = b (with X the vector v we are looking for so [s, r] in our case)
b = np.array([0, 0, 1])

# We resolve using linalg which is a solver for systems of equations
stationary = np.linalg.lstsq(A, b, rcond=None)[0]
print("Stationary distribution:", stationary)
