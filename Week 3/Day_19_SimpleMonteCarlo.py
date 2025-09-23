### Day 19 -- Simple Monte Carlo Simulation
"""
In the previous example, we updated the Q-table values under the temporal difference method which updates the Q-table values at each step using the expected future gains without waiting for the end of the episode

But, there are other methods to calculate Q-table values
Today, we are going to introduce one of them which is Monte Carlo
Using Monte Carlo to calculate Q-table values wait the end of the epsidoe before updating the Q-table and is less biased because it uses real gains and not expected future gains to calculate Q-table values
"""

import random
import matplotlib.pyplot as plt

### SIMPLE MONTE CARLO
"""
The idea is to simulate random points (x, y) uniformly distributed in a square of size 2x2 centered at the origin, so x and y are both in [-1, 1]

We then check whether each point falls inside the unit circle with radius = 1 and area = pi
To do so, we calculate the distance of the point from the origin using distance : distance = sqrt(x^2 + y^2)

The probability that a random point falls inside the circle is equal to the ratio of the areas: (area of circle) / (area of square) = π / 4
And, we estimate the area of circle by counting the points within the circle divided by the overall number of points (which represents the area of the square)

Therefore, if we generate a large number of random points and count how many fall inside the circle,
we can estimate pi using the formula:
    pi ≈ 4 × (number of points inside circle / total number of points)

The more points we simulate, the more accurate the estimation becomes
This is the essence of Monte Carlo : solving problems using repeated random sampling
"""

def monte_carlo_pi(n_points: int) -> float:
    inside_circle = 0
    x_in = []
    y_in = []

    x_out = []
    y_out = []

    for _ in range(n_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        distance = (x**2 + y**2)**(0.5)

        if distance <= 1:
            inside_circle += 1
            x_in.append(x)
            y_in.append(y)
        else:
            x_out.append(x)
            y_out.append(y)

    pi_estimate = 4 * inside_circle / n_points

    plt.scatter(x_in, y_in, color='blue', s=0.5, label='Inside circle')
    plt.scatter(x_out, y_out, color='red', s=0.5, label='Outside circle')
    plt.title(f"Estimate of pi with {n_points} points : {pi_estimate}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    return pi_estimate


n = 10000  # Nombre de points
estimation = monte_carlo_pi(n)
print(f"Estimate of pi with {n} points : {estimation}")

"""
So we have a pretty good estimation
If we increase the number of points, the result tends to be nearer to the real value of pi
"""

# Author GCreus
# Done via pyzo
