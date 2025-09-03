### Day 1 -- Simple Linear Regression
# We will try to model a simple relationship from a github dataset

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
# First, let's import the dataframe and check that the data is cleaned
df = pd.read_csv("...your_path/HousingPrices.csv")

df.head()

# Let's ensure that a linear relationship exists
plt.close('all')
X = df['YearsExperience'].values.tolist()
Y = df['Salary'].values.tolist()
plt.plot(X, Y, "ob", label = "Raw data")
#plt.show()

### MODELLING THE RELATIONSHIP
# Now, we will code the tools to replicate the linear relationship
# We want to replicate the relationship as follows : Y_pred = m * X + b
# With m as the slope, i.e : Cov(X, Y)/Var(X)
# And with b as the Y_intercept : b = mean(Y) - m * mean(X)

def list_mean(list: list): #compute the mean of a list
    res = 0
    n = len(list)

    for i in range(n):
        res += list[i]

    res /= n

    return res

def list_covariance(list1: list, list2: list): #compute the covariance of two lists
    res = 0
    n = len(list1)

    mean_list1 = list_mean(list1)
    mean_list2 = list_mean(list2)

    for i in range(n):
        res += (list1[i] - mean_list1) * (list2[i] - mean_list2)

    res /= n

    return res

# Once this is done, we compute the coefficients
m = list_covariance(X, Y)/list_covariance(X, X)
b = list_mean(Y) - m * list_mean(X)

# And we compute the predicted values
Y_pred = [m * x + b for x in X]

# We can then check that everything works as intended
plt.plot(X, Y_pred, label = "Salary prediction")
plt.legend(title = "Legend Title")
plt.title("Prediction vs Reality")
plt.show()

### COMPUTE THE ERRORS
# We will now compute the error metrics to evaluate our model
def SSE(list_Y_pred: list, list_Y: list): #compute the SSE (sum of the residuals errors) of our prediction
    n = len(list_Y_pred)
    res = 0

    for i in range(n):
        res += (list_Y_pred[i] - list_Y[i])**2

    return res

def SST(list_Y: list): #compute the SST (total error) of the real relationship
    n = len(list_Y)
    res = 0

    Y_mean = list_mean(list_Y)

    for i in range(n):
        res += (list_Y[i] - Y_mean)**2

    return res

### PRINT THE ERRORS METRICS
# We can now calculate the R squared value to see if the linear relationship found fits well the data
print("SSE = ", SSE(Y_pred, Y))
print("SST = ", SST(Y))

R2 = 1 - SSE(Y_pred, Y)/SST(Y_pred)
print("R2 = ", R2)

# Author GCreus
# Done via pyzo
