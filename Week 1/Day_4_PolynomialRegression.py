### Day 4 -- Polynomial Regression with Gradient Descent
"""
We will try to model a polynomial relationship from a github dataset using a gradient descent to find the right parameters
This problem is approximately the same as a multi linear regression with X**i as the independent variables
"""

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("...your_path/IceCream.csv")

df.head()


"""
Let's ensure that a polynomial relationship exists using a 2D plot
"""
plt.close('all')
X = df['Temperature'].values.tolist()
Y = df['Ice Cream Sales'].values.tolist()
plt.plot(X, Y, "ob", label = "Raw data")

plt.show()


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

def create_polynomial_function(degree: int, X : list[float]) -> list[list[float]]:
    """
    Ags :
        degree = the degree of the polynomial function we want to use
        X = the values of the real independent variable

    Return a list of power of X, with each power of X a list itself containing the values of the independent variable to the right power
    """
    res = []
    n = len(X) #the number of values each X**i will have to take

    for i in range(degree):
        res.append([])
        for k in range(n):
            res[i].append(X[k]**(i+1)) #we add +1 because the index starts at 0

    return res

def mse_value(parameters_set: list[float], X: list[float], Y: list[float]) -> float:
    """
    Ags :
        parameters_set = the parameters of the polynomial function, i.e (m1, m2,... , b)
        X = the values of the real independent variable
        Y = the real values we try to replicate

    Note that we don't need to input the degree of our polynomial function, we will deduct it from the number of parameters

    Return the MSE for the set of parameters
    """
    degree = len(parameters_set) - 1
    list_of_powers_of_X = create_polynomial_function(degree, X)
    number_of_variables = len(list_of_powers_of_X)

    n = len(Y)
    res = 0

    for i in range(n):
        y_pred = parameters_set[-1]
        for k in range(number_of_variables):
            y_pred += parameters_set[k] * list_of_powers_of_X[k][i]
        res += (y_pred - Y[i])**2

    res /= n

    return res


"""
Let's try to calculate the MSE for dummy parameters
"""
X_test = [1, 2, 3]
Y_test = [1, 2, 3]
"""
Simple relationship here, Y = 1 * X
"""

parameters_set = [1, 0, 0, 0] #test with degree of 3 but with only X**1
print("For the relationship Y = X;\nWith m1 = 1, m2 = 0, m3 = 0 and b = 0, we have a MSE of : ", mse_value(parameters_set, X_test, Y_test), '\n')



X_test = [1, 2, 3]
Y_test = [1, 8, 27]
"""
Simple relationship here, Y = X**3 + 1
"""

parameters_set = [0, 0, 1, 1]
print("For the relationship Y = X**3;\nWith m1 = 0, m2 = 0, m3 = 1 and b = 1, we have a MSE of : ", mse_value(parameters_set, X_test, Y_test), '\n')

### USING THE GRADIENT DESCENT
"""
How could we find the optimal set of parameter from there ?
Since the MSE value depends on (m1, m2,..., b) this is a ND problem where the optimal set of parameter is a global minimum
In the same way as a multi linear relationship, we will try to seek for the global minimum with gradient descent on a N dimension space

With high dimensions, we might not find the global minimum but a local minimum, it depends on the initial parameters set

Our goal is to minimize the following function :
1/n * SUM( (Y_pred(i) + b - Yi)**2 )
1/n * SUM( (VectorM * VectorX(i) + b - Yi)**2 )
The partial derivatives for this function are as follow:
dMSE/dmi = 2/n * SUM( (VectorM * VectorX(i) + b - Yi) * X**i )
dMSE/db = 2/n * SUM( (VectorM * VectorX(i) + b - Yi) )
"""

def gradient(parameters_set: list[float], X: list[float], Y: list[float]) -> list[float]:
    """
    Ags :
        parameters_set = the parameters of the polynomial function, i.e (m1, m2,... , b)
        X = the values of the real independent variable
        Y = the real values we try to replicate

    Note that we don't need to input the degree of our polynomial function, we will deduct it from the number of parameters

    Return a list of gradient for each parameter in the set as follow :
    (dMSE/dm1, dMSE/dm2, ..., dMSE/dmn, dMSE/db)
    """
    degree = len(parameters_set) - 1
    list_of_powers_of_X = create_polynomial_function(degree, X)
    number_of_variables = len(list_of_powers_of_X)

    n = len(Y)
    res = [0.0 for _ in range(degree + 1)] #we add back the Y intercept b

    for i in range(n):
        y_pred = parameters_set[-1]  # b
        for p in range(number_of_variables):
            y_pred += parameters_set[p] * list_of_powers_of_X[p][i]

        error = y_pred - Y[i]

        for k in range(number_of_variables):
            res[k] += error * list_of_powers_of_X[k][i]
        res[-1] += error

    for k in range(number_of_variables + 1):
        res[k] = (2/n) * res[k]

    return res

"""
Now that the gradients are explicit, let's setup the gradient descent
"""
def gradient_descent(initial_set: list[float], X: list[float], Y: list[float], iterations: int, learning_rate: float) -> list[float]:
    """
    Ags :
        initial_set = the initial set of parameters of the polynomial function, i.e (m1, m2,... , b)
        X = the values of the real independent variable
        Y = the real values we try to replicate
        iterations = how many times we will calibrate parameters with the gradient descent
        learning_rate = how large the parameters movement will before and after the gradient descent

    Note that we don't need to input the degree of our polynomial function, we will deduct it from the number of parameters

    Return a list of (probably) optimal parameters as follow :
    (m1, m2,... , b)
    """
    parameters = []
    for i in range(len(initial_set)):
        parameters.append(initial_set[i])

    for i in range(iterations):
        temp_gradient = gradient(parameters, X, Y)
        for p in range(len(parameters)):
            parameters[p] = parameters[p] - learning_rate * temp_gradient[p]
        print(mse_value(parameters, X, Y))

    return parameters


### COMPARISON WITH REAL VALUES
"""
Now we are going to check that the values found by the gradient descent method properly represents the linear relationship
"""

"""
We first calculate the parameters found by gradient descent, also we can add degree if we want
"""

initial_parameters = [0, 0, 0]
parameters = gradient_descent(initial_parameters, X, Y, 1000, 0.0008)

"""
Here we have to be very cautious since our data aren't normalized, we can diverge very fast and this is why the learning rate is very small in order to prevent extreme deviation
"""


"""
And we compute the predicted values
"""
Y_pred = []
for i in range(len(X)):
    Y_pred.append(parameters[-1])
    for k in range(len(initial_parameters) - 1):
        Y_pred[i] += parameters[k] * X[i]**(k+1)

"""
Let's plot the curve
"""
plt.close('all')
plt.plot(X, Y, "ob", label = "Raw data")
plt.plot(X, Y_pred, label = "Prediction")
plt.legend(title = "Legend Title")
plt.title("Prediction vs Reality")
plt.show()


# Author GCreus
# Done via pyzo
