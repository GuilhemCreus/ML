### Day 2 -- Simple Linear Regression with Gradient Descent
# We will try to model a simple relationship from a github dataset, but this time, we will use a gradient descent to find the right parameters

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
# First, let's import the dataframe and check that the data is cleaned
df = pd.read_csv("/Users/guilhemcreus/Desktop/HousingPrices.csv")

df.head()

# Let's ensure that a linear relationship exists
plt.close('all')
X = df['YearsExperience'].values.tolist()
Y = df['Salary'].values.tolist()
plt.plot(X, Y, "ob", label = "Raw data")
#plt.show()

### SETTING UP THE ERROR METRIC
# We want to replicate the relationship as follows : Y_pred = m * X + b
# With m as the slope and with b as the Y_intercept


# To measure the performance of a set, we will compute the mean squared error of the predicted Y values against the real Y values
# Our final goal is to find the set of pramaeters that minimize the mean of squared errors
# Then, we first create a function that return the MSE for a given set of parameter
def mse_value(m_slope: float, b_Yintercept: float, X: list, Y: list):
    n = len(X)
    res = 0

    for i in range(n):
        res += (m_slope * X[i] + b_Yintercept - Y[i])**2

    res /= n

    return res

# Let's try to calculate the MSE for dummy parameters
print("For m = 1, and b = 0, we have a MSE of : ", mse_value(1.5, 1, X, Y))

### USING THE GRADIENT DESCENT
# How could we find the optimal set of parameter from there ?
# Since the MSE value depends only on m and b, this is a 2D problem where the optimal set of parameter is a global minimum
# This is where the intuition of Gradient Descent stems from

# Our goal is to minimize the following function :
# 1/n * SUM( (m * Xi + b - Yi)**2 )
# The partial derivatives for this function are as follow:
# dMSE/dm = 2/n * SUM( (m * Xi + b - Yi) * Xi )
# dMSE/db = 2/n * SUM( (m * Xi + b - Yi) )
def m_gradient(m_slope: float, b_Yintercept: float, X: list, Y: list):
    n = len(X)
    res = 0

    for i in range(n):
        res += (m_slope * X[i] + b_Yintercept - Y[i]) * X[i]

    res /= n
    res *= 2

    return res

def b_gradient(m_slope: float, b_Yintercept: float, X: list, Y: list):
    n = len(X)
    res = 0

    for i in range(n):
        res += (m_slope * X[i] + b_Yintercept - Y[i])

    res /= n
    res *= 2

    return res

# Now that the gradients are explicit, let's setup the gradient descent
def gradient_descent(m_slopeInitial: float, b_YinterceptInitial: float, X: list, Y: list, iterations: int, learning_rate: float):
    m = m_slopeInitial
    b = b_YinterceptInitial

    for i in range(iterations):
        m = m - learning_rate * m_gradient(m, b, X, Y)
        b = b - learning_rate * b_gradient(m, b, X, Y)
        #print(mse_value(m, b, X, Y))

    return m, b

### COMPARISON WITH REAL VALUES
# Now we are going to check that the values found by the gradient descent method properly represents the linear relationship

# We first calculate the parameters found by gradient descent
m,b = gradient_descent(1, 0, X, Y, 1000, 0.01)

# And we compute the predicted values
Y_pred = [m * x + b for x in X]

# We can then check that everything works as intended
plt.close('all')
plt.plot(X, Y, "ob", label = "Raw data")
plt.plot(X, Y_pred, label = "Salary prediction")
plt.legend(title = "Legend Title")
plt.title("Prediction vs Reality")
plt.show()


# Author GCreus
# Done via pyzo
