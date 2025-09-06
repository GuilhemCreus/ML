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
First, let's import the dataframe and check that the data is cleaned
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

#plt.show()
"""
A relationship kinda exist
"""


### SETTING UP THE ERROR METRIC
