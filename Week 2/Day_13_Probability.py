### Day 13 -- Basics of probability
"""
Now that we have seen the basics of statistical tests, we move on the basics of probability starting with the basic concepts and thereafter we will see conditional probability (discrete and continuous)
"""

import numpy as np
import matplotlib.pyplot as plt

### MAIN CONCEPTS IN PROBABILITY
"""
A probability space consists of three elements :
Ω : the set of all possible outcomes of a random process
F : a σ-algebra/event space
With (Ω, F) a measurable space
P : a probability function

Let X be a set, a σ-algebra F is a set of parts A of X where :
-A is not empty
-the complement of A is also in F
-if we have a countable collection of sets in F, then their union is in F too

P is a probability function on Ω if :
-P -> [0, 1]
-P(Ω) = 1
-P(∅) = 0

We are going to use the above definitions as a ground for the following :

Let F be a σ-algebra, let A, B in F with P(B) > 0
We call conditional probability of A knowing B the amount : P(A|B) = P(A∩B) / P(B)
With P(A∩B) = P(A|B) * P(B) = P(B|A) * P(A)

With the same reasoning, let (Bn) for n in N a partition of the set Ω with each Bn in F and for each Bn : P(Bn) > 0
We have then for every A in F and for every n :
P(Bn|A) = P(A|Bn) * P(Bn) / SUM(P(Bk) * P(A|Bk))

Let's apply that in a quick example :
Let M be the event that a person is sick with P(M) = 0.5%
A test is positive at a rate of 99% if the person is sick, P(Positive|M) = 99%
A test is positive at a rate of 2% if the person is not sick, P(Positive|not M) = 2%

If a person randomly selected gets a positive result at the test above, what is the probability that he/she is really sick ?
"""

P_M = 0.005
P_positive_test_knowing_M = 0.99
P_positive_test_knowing_not_M = 0.02

"""
We are looking for P(M|Positive) = P(Positive|M) * P(M) / P(Positive)

With P(Positive) = P(Positive|M) * P(M) + P(Positive|not M) * P(not M)
And P(not M) = the complement of M = 1 - P(M) = 99.5%
"""

P_positive = P_positive_test_knowing_M * P_M + P_positive_test_knowing_not_M * (1 - P_M)

P_M_knowing_positive = P_positive_test_knowing_M * P_M / P_positive

### RANDOM VARIABLE
"""
A random variable is a function which output value depends on hasard
By essence, a random variable is a measurable function X : Ω → R, where Ω is the sample space
It permits us to translate real world events that are random

We distinguish discrete random variable and continuous random variable

A discrete random variable can takes only countable values
A discrete random variable has a probability mass function
The probability mass function of a random variable provides the possible values and their associated probabilities for all possible outcome
It is defined as follow : pX(x) = P(X = x)

A continuous random variable is the extension of the above notion
It can take any value in a continuum interval or space like R
A continuous random variable is defined by its probability density function (PDF) f(x), such that:
P(a <= X <= b) = ∫(a -> b) of f(x)dx
With :
-f(x) >= 0 for all x
-​∫(all space) of f(x)dx = 1
-P(X = x) = 0 for all points, this is due to the fact that for continuous random variables probabilities are calculated over intervals, not points
"""

### CONTINUOUS CONDITIONAL PROBABILITY
"""
Just as in the discrete case, a probability space is defined by three components :
Ω : the set of all possible outcomes of a random process
F : a σ-algebra/event space
With (Ω, F) a measurable space
P : a probability function

Let X and Y be continuous random variables defined on the same probability space
The joint cumulative distribution function (CDF) is defined as F:X,Y with :
F:X,Y (x, y) = P(X <= x, Y <= y)

X and Y are also associated with a joint probability density function f:X,Y (x, y) that stems from the CDF : f:X,Y (x, y) = ∂∂F:X,Y (x, y) / (∂x∂y)

Simply remind that this function gives the density of probability that X and Y take values near (x, y)

To get the marginal density of X, integrate the joint PDF over all possible values of Y : f:X(x) = ​∫(all possible values of Y) f:X,Y(x, y)dy likewise for f:Y

We finally arrive at the conditional probability density function :
Let f:X,Y(x, y) be the joint PDF of X, Y two random continuous variables and suppose f:Y(y) > 0
Then, the conditional density of X given Y = y is defined as :
f:X|Y(x|y) = f:X,Y(x, y)/f:Y(y)

Now let's work this with a practical example



Let (X,Y) be a random point uniformly distributed in the triangle defined by:
0 <= x <= 1 and 0 <= y <= x

This region is a right triangle under the line y = x, and the joint PDF is uniform over that triangle
Let’s compute the conditional density of Y given X = x
Since the distribution is uniform over the triangle, the joint PDF f:X,Y(x, y) is constant inside the triangle and 0 elsewhere, we first try to find the constant C
​∫​∫(all possible values) f:X,Y(x, y)dxdy = ​∫​∫(triangle area) f:X,Y(x, y)dxdy = C * triangle area = 1

And, triangle area is L * l / 2 = 1/2, so C = 2
So : f:X,Y(x, y) = 2 if 0 <= y <= x <= 1 and 0 otherwise

Now we only need f:X(x) to be able to fully calculate f:Y|X(y, x) for any x
f:X(x) = ​∫(all possible values of Y) f:X,Y(x, y)dxdy
f:X(x) = ​∫(y = 0 -> x) f:X,Y(x, y)dy = ​∫(y = 0 -> x) 2dy = 2x

So f:X(x) = 2x if 0 <= x <= 1 and 0 otherwise

Finally,
f:Y|X(y, x) = f:X,Y(x, y)/f:X(x) = 2 / 2x = 1 / x for 0 <= y <= x and 0 otherwise
"""
### PYTHON PLOT
# Generate uniform points inside the triangle
N = 2000000
x_vals = np.random.uniform(0, 1, N)
y_vals = np.random.uniform(0, 1, N)

# We will keep only the points below the line y <= x
mask = y_vals <= x_vals
x_tri = x_vals[mask]
y_tri = y_vals[mask]

# We fix here x close to a value <= 1 with some tolerance around and plot the conditional distribution of y
x_fixed = 0.5
tolerance = 0.02

# We select only the points around the fixed x
y_cond = y_tri[np.abs(x_tri - x_fixed) <= tolerance]

plt.close('all')
# bins : number of bars in the interval, alpha : transparency of the bars
plt.hist(y_cond, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title(f"Conditional distribution of Y given X around {x_fixed}")
plt.xlabel("Y")
plt.ylabel("Density")
plt.show()

"""
So we get kind of a flat surface with a density of 1/x approximately but with a higher number of data, we should get a flat plot

We worked on continuous conditional probability because they are useful when computing the expected value which is itsel useful when computing reward in reinforcement learning
"""

# Author GCreus
# Done via pyzo
