### Day 8 -- Lasso Regression
"""
We will try to model a multi-linear relationship from a github dataset using a lasso regression to find the right parameters

Lasso regression looks like multi linear regression but we add a constraint linked to the sum of absolute value of the weights (bias excluded)
"""

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("/Users/guilhemcreus/Desktop/Github/WEEK 2/auto-mpg.csv")

df.head()

df_test = df.tail(2)
df_train = df.iloc[:-2]


### SETTING UP THE STANDARD SCALER
"""
Since we already know how the standard scaler work, we will now import it
"""

from sklearn.preprocessing import StandardScaler

X_train = df_train.drop(columns=['mpg'])
Y_train = df_train['mpg']

X_test = df_test.drop(columns=['mpg'])
Y_test = df_test['mpg']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

"""
We then put back our data in list format, with each column of each dataframe representing a list
"""
independent_variables_list = [X_train_scaled[:, i].tolist() for i in range(X_train_scaled.shape[1])]
Y = Y_train.tolist()


### SETTING UP THE ERROR METRIC
"""
We want to replicate the relationship as follows : Y_pred = m1 * X1 + m2 * X2 + ... + b
We can rewrite this relationship with vectors : Y_pred = VectorM * VectorX + b to make the problem standard to N dimensions
With (m1, m2,..., b) a set of parameters
"""

"""
To measure the performance of a set, we will compute the mean squared error of the predicted Y values against the real Y values under the constraint : SUMabs(weight) <= t (bias excluded)
Our final goal is to find the set of pramaeters that minimize the mean of squared errors with respect to the constraint

It sounds like Lagrande ?
Well, in fact we can use the Lagrange Format of this problem as follow :
We seek to minimize : SUM((y_pred - y)**2) + lamba * SUM(abs(weight)
(again, when computing lamba * SUM(abs(weight), we exclude bias)

We first create a function that return the goal function for a given set of parameter and a given lambda
"""

"""
Before jumping into coding, let's detail the data structure that our functions will take as input
We will inject the set of parameters as input following this convention :
set = (m1, m2,... , b)

And, the independent variables as follow :
independent_variables_list = [X1, X2,... ] where X1 = [x1(0), x1(1) ...]
"""

def goal_function(parameters_set: list[float], lambd : float, independent_variables_list : list[list[float]], Y: list[float]) -> float:
    """
    Ags :
        parameters_set = the parameters of the multi linear regression, i.e (m1, m2,... , b)
        lambd = lambda, the Lagrange multiplier (if we put lambda python will think it's a function)
        independent_variables_list = [X1, X2,... ] where Xi = [xi(0), xi(1) ...]
        Y = real values we try to predict

    Return the value of the goal function for a given lambda and a given set of parameters
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

    for k in range(numberOfVariables):
        res += lambd * abs(parameters_set[k])

    return res


### USING THE GRADIENT DESCENT
"""
Here again, the goal function depends on (m1, m2,..., b, lambda) this is a ND problem where the optimal set of parameter is a global minimum
In the same way as a simple linear relationship, we will try to seek for the global minimum with gradient descent on a N dimension space

Our goal is to minimize the following function :
1/n * [SUM((Y_pred(i) + b - Yi)**2)] + lambda * SUM(abs(weight)
1/n * [SUM((VectorM * VectorX(i) + b - Yi)**2)] + lambda * norm1(VecotrM)]

The partial derivatives for this function are as follow:
dMSE/dmi = 2/n * [SUM((VectorM * VectorX(i) + b - Yi) * Xi] + lambda * sign(mi)
dMSE/db = 2/n * SUM((VectorM * VectorX(i) + b - Yi))
"""

def gradient(parameters_set: list[float], lambd : float, independent_variables_list : list[list[float]], Y: list[float]) -> list[float]:
    """
    Ags :
        parameters_set = the parameters of the multi linear regression, i.e (m1, m2,... , b)
        lambd = lambda, the Lagrange multiplier (if we put lambda python will think it's a function)
        independent_variables_list = [X1, X2,... ] where Xi = [xi(0), xi(1) ...]
        Y = real values we try to predict

    Return the gradient of the goal function as a vector (list)
    """
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

    for k in range(numberOfParameters - 1):
        res[k] = (2/n) * res[k]
        if parameters_set[k] > 0:
            res[k] += lambd
        elif parameters_set[k] < 0:
            res[k] -= lambd
        else:
            res[k] =+ 0

    res[-1] = (2/n) * res[-1]

    return res

# Now that the gradients are explicit, let's setup the gradient descent
def gradient_descent(initial_set: list[float], lambd : float, independent_variables_list : list[list[float]], Y: list[float], iterations: int, learning_rate: float) -> list[float]:
    """
    Ags :
        initial_set = the initial parameters of the multi linear regression, i.e (m1, m2,... , b)
        lambd = lambda, the Lagrange multiplier (if we put lambda python will think it's a function)
        independent_variables_list = [X1, X2,... ] where Xi = [xi(0), xi(1) ...]
        Y = real values we try to predict
        iterations = how many times we will calibrate parameters with the gradient descent
        learning_rate = how large the parameters movement will before and after the gradient descent

    Return the gradient of the goal function as a vector (list)
    """
    parameters = []
    for i in range(len(initial_set)):
        parameters.append(initial_set[i])

    for i in range(iterations):
        temp_gradient = gradient(parameters, lambd, independent_variables_list, Y)
        for p in range(len(parameters)):
            parameters[p] = parameters[p] - learning_rate * temp_gradient[p]
        print(goal_function(parameters, lambd, independent_variables_list, Y))

    return parameters


### COMPARISON WITH REAL VALUES
"""
Now we are going to check that the values found by the gradient descent method properly represents the linear relationship
"""

"""
We first calculate the parameters found by gradient descent
"""
initial_parameters = [1.0] * (len(independent_variables_list) + 1) # on mets le biais
parameters = gradient_descent(initial_parameters, 0.01, independent_variables_list, Y, 3000, 0.01)


"""
And we compute the predicted values
"""
Y_pred = []
X_test_list = [X_test_scaled[:, i].tolist() for i in range(X_test.shape[1])]
Y_test_list = Y_test.tolist()

res = []
for i in range(X_test.shape[0]):
    temp_float = parameters[-1]
    for k in range(X_test.shape[1]):
        temp_float += parameters[k] * X_test_list[k][i]
    res.append(temp_float)

print(res)
print(Y_test_list)

"""
By printing the parameters, we can see that lasso regression has put an important weight on the characteristic 'weight' which is logical since miles per gallon depends heavily on the mass of the car; also most of other parameters are near zero because they do not heavily impact y

This is an important feature of lasso regression, it finds by itself which parameter has the most influence on y, this is due to lambda (L1 penalty) which balances the tradeoff between bias and variance in the resulting coefficients.

In some special case, we could have used Karush-Kuhn-Tucke conditions to find an optimal solution explicitly but this would have required us to study deeper the constrait function and wether or not the problem is convex
"""

# Author GCreus
# Done via pyzo
