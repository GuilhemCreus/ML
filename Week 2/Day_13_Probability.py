### Day 13 -- Basics of probability
"""
Now that we have seen the basics of statistical tests, we move on the basics of probability starting with the basic concepts and thereafter we will see conditional probability (discrete and continuous)
"""


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

