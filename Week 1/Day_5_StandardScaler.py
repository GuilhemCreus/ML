### Day 5 -- Implementing Standard Scaler for Multi Linear Regression
"""
For the last two days, I have been implementing multi linear regression and polynomial regression (which is a case of multi linear regression) without scaling the data which caused some issues if the learning rate wasn't small enough

This is due to the fact that without proper scaling, the gradient descent might go too far following one variable, but too slow following the others

In this code, we will try to implement it to multi linear regression
"""

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("...your_path/SalaryUSA.csv")

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

plt.show()
"""
A relationship kinda exist, and we see that there are outliers, but for this week we don't treat them for the moment
We will handle data cleaning in the future !
"""

### SETTING UP THE STANDARD SCALER
"""
In order to have features with mean 0 and standard deviation of 1, we will scale our features
To scale our features, we will use the following formula :
x_scaled(i) = (x(i) - x_mean)/standard_deviation(x)
"""

def standard_scaler(X: list[float]) -> list[float]:
    """
    Ags :
        X = the values of the real independent variable

    Return a new list of scaled values of X
    """

    mean_X = sum(X) / len(X)
    std_X = (sum((x - mean_X)**2 for x in X) / len(X))**0.5

    X_scaled = [(x - mean_X) / std_X for x in X]
    return X_scaled

"""
We then scale our features
"""
X1_scaled = standard_scaler(X1)
X2_scaled = standard_scaler(X2)

"""
From there, the rest of the programe is the same as my previous program on multi linear regression, except that I will use scaled features
"""

### SETTING UP THE ERROR METRIC
"""
We want to replicate the relationship as follows : Y_pred = m1 * X + m2 * X**2 + ... + mn * X**n + b
We can rewrite this relationship with vectors : Y_pred = VectorM * VectorX + b to make the problem standard to N dimensions
With (m1, m2,..., b) a set of parameters
"""

"""
To measure the performance of a set, we will compute the mean squared error of the predicted Y values against the real Y values
Our final goal is to find the set of pramaeters that minimize the mean of squared errors
We first create a function that lets us choose the degree of the polynomial function we will use to fit the data
Then, we create a function that return the MSE for a given set of parameter given a polynomial function
In the same way as a multi linear relationship but this time, we are kinda free to choose the independent variables by changing the degree of the polynomial function
"""

"""
Before jumping into coding, let's detail the data structure that our functions will take as input
We will inject the set of parameters as input following this convention :
set = (m1, m2,... , b)

And, the independent variables as follow :
independent_variables_list = [X, X**2,... ] where X**i = [x**i(0), x**i(1) ...]
"""

def mse_value(parameters_set: list[float], independent_variables_list : list[list[float]], Y: list[float]) -> float:
    """
    Ags :
        parameters_set = the set of parameters used to predict Y_pred
        independent_variables_list = a list of independent variables, each independent variable is itself alist of the values of the real independent variable
        Y = the real values of the dependent variable

    Return a new list of scaled values of X
    """
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
We first calculate the parameters found by gradient descent using scaled features
"""
independent_variables_list = [X1_scaled, X2_scaled]
initial_parameters = [0, 0, 0]
parameters = gradient_descent(initial_parameters, independent_variables_list, Y, 10000, 0.01)

# Best set is [-86.80320182437751, 2153.115150820222, 30803.7559409089] for the moment

"""
Here we have to be very cautious since our data aren't normalized, we can diverge very fast and this is why the learning rate is very small in order to prevent extreme deviation
"""

"""
And we compute the predicted values
"""
Y_pred = []
for i in range(len(X1)):
    Y_pred.append(parameters[0] * X1_scaled[i] + parameters[1] * X2_scaled[i] + parameters[-1])

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

"""
So now, comparing to our previous code, we no longer have problem of extreme deviations using the gradient descent
We could also have scaled the Y values in order to have proportional movement at each gradient descent step

Also, note here that we have found the coefficient for a relationship with the scaled features, in order to find the real relationship, we should expend the formula and adjust the coeffecients which is not a hard step but this is not the goal of this program, this program only aims at coding a StandardScaler
"""

# Author GCreus
# Done via pyzo
