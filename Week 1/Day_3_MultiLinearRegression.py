### Day 3 -- Multiple Linear Regression with Gradient Descent
"""
We will try to model a multi-linear relationship from a github dataset using a gradient descent to find the right parameters
"""

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is cleaned
"""
df = pd.read_csv("/Users/guilhemcreus/Desktop/Github/WEEK 1/SalaryUSA.csv")

df.head()


"""
Let's ensure that a relationship exists using a 3D plot
"""
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X1 = df['age'].values.tolist()
X2 = df['experience'].values.tolist()
Y = df['income'].values.tolist()

ax.scatter(X1, X2, Y, marker='o')
ax.set_xlabel('age')
ax.set_ylabel('experience')
ax.set_zlabel('income')

#plt.show()
"""
A relationship kinda exist
"""

### SETTING UP THE ERROR METRIC
"""
We want to replicate the relationship as follows : Y_pred = m1 * X1 + m2 * X2 + b
We can rewrite this relationship with vectors : Y_pred = VectorM * VectorX + b to make the problem standard to N dimensions
With (m1, m2,..., b) a set of parameters
"""

"""
To measure the performance of a set, we will compute the mean squared error of the predicted Y values against the real Y values
Our final goal is to find the set of pramaeters that minimize the mean of squared errors
Then, we first create a function that return the MSE for a given set of parameter
In the same way as a simple linear relationship
"""

"""
Before jumping into coding, let's detail the data structure that our functions will take as input
We will inject the set of parameters as input following this convention :
set = (m1, m2,... , b)

And, the independent variables as follow :
independent_variables_list = [X1, X2,... ] where X1 = [x1(0), x1(1) ...]
"""

def mse_value(parameters_set: list[float], independent_variables_list : list[list[float]], Y: list[float]) -> float:
    numberOfVariables = len(independent_variables_list)
    n = len(Y)
    res = 0

    for i in range(n):
        y_pred = parameters_set[-1]
        for k in range(numberOfVariables):
            y_pred += parameters_set[k] * independent_variables_list[k][i]
        res += (y_pred - Y[i])**2

    res /= n

    return res

###
"""
Let's try to calculate the MSE for dummy parameters
"""

independent_variables_list = [[1, 2, 3], [1, 2, 3]]
Y_test = [2, 4, 6]
parameters_set = [1, 1, 1]

print("For m1 = 1, m2 = 1 and b = 1, we have a MSE of : ", mse_value(parameters_set, independent_variables_list, Y_test))

parameters_set = [1, 1, 0]
print("\nFor m1 = 1, m2 = 1 and b = 0, we have a MSE of : ", mse_value(parameters_set, independent_variables_list, Y_test))

### USING THE GRADIENT DESCENT
"""
How could we find the optimal set of parameter from there ?
Since the MSE value depends on (m1, m2,..., b) this is a ND problem where the optimal set of parameter is a global minimum
In the same way as a simple linear relationship, we will try to seek for the global minimum with gradient descent on a N dimension space
But here, since we are not working on a plan, the intuition is much harder to see, hence the utility of working with simple linear regression first

Our goal is to minimize the following function :
1/n * SUM( (Y_pred(i) + b - Yi)**2 )
1/n * SUM( (VectorM * VectorX(i) + b - Yi)**2 )
The partial derivatives for this function are as follow:
dMSE/dmi = 2/n * SUM( (VectorM * VectorX(i) + b - Yi) * Xi )
dMSE/db = 2/n * SUM( (VectorM * VectorX(i) + b - Yi) )
"""

def gradient(parameters_set: list[float], independent_variables_list : list[list[float]], Y: list[float]) -> list[float]:
    numberOfParameters = len(parameters_set)
    n = len(Y)
    res = [0.0 for _ in range(numberOfParameters)]

    for i in range(n):
        y_pred = parameters_set[-1]  # b
        for p in range(numberOfParameters - 1):
            y_pred += parameters_set[p] * independent_variables_list[p][i]

        error = y_pred - Y[i]

        for k in range(numberOfParameters - 1):
            res[k] += error * independent_variables_list[k][i]
        res[-1] += error

    for k in range(numberOfParameters):
        res[k] = (2/n) * res[k]

    return res

# Now that the gradients are explicit, let's setup the gradient descent
def gradient_descent(initial_set: list[float], independent_variables_list : list[list[float]], Y: list[float], iterations: int, learning_rate: float) -> list[float]:
    parameters = []
    for i in range(len(initial_set)):
        parameters.append(initial_set[i])

    for i in range(iterations):
        temp_gradient = gradient(parameters, independent_variables_list, Y)
        for p in range(len(parameters)):
            parameters[p] = parameters[p] - learning_rate * temp_gradient[p]
        print(mse_value(parameters, independent_variables_list, Y))

    return parameters


### COMPARISON WITH REAL VALUES
"""
Now we are going to check that the values found by the gradient descent method properly represents the linear relationship
"""

"""
We first calculate the parameters found by gradient descent
"""
independent_variables_list = [X1, X2]
initial_parameters = [-850, 8000, 40000]
parameters = gradient_descent(initial_parameters, independent_variables_list, Y, 50000, 0.0004)

# Best set is [-86.80320182437751, 2153.115150820222, 30803.7559409089] for the moment

"""
Here we have to be very cautious since our data aren't normalized, we can diverge very fast and this is why the learning rate is very small in order to prevent extreme deviation
"""

"""
And we compute the predicted values
"""
Y_pred = []
for i in range(len(X1)):
    Y_pred.append(parameters[0] * X1[i] + parameters[1] * X2[i] + parameters[-1])

"""
Let's plot the curve using a 3D plot
"""
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X1 = df['age'].values.tolist()
X2 = df['experience'].values.tolist()
Y = df['income'].values.tolist()

ax.scatter(X1, X2, Y, marker='o')
ax.set_xlabel('age')
ax.set_ylabel('experience')
ax.set_zlabel('income')
ax.scatter(X1, X2, Y_pred, marker='p')

plt.show()


# Author GCreus
# Done via pyzo
