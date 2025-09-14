### Day 12 -- MANOVA
"""
We have seen how the ANOVA works, we now try to see how MANOVA works

ANOVA asks: Do groups differ on one variable ?
Using variances in ANOVA, we can tell if the means (of one variable) of groups are the same or not, i.e it tells us if groups differ regarding one variable

MANOVA asks: Do groups differ on a set of variables taken together ?
ANOVA uses variances of groups for one variable and tells us if means are the same for one variable. But MANOVA assess whether the group differences on a combination of dependent variables are statistically significant

We might asks then, why don't we simply use ANOVA mutliple times on the different variables ? Using ANOVA (or any statistical test) multiple times increases the risk of type 1 error because each test has its own probability of producing a type 1 error
When we run multiple tests, these individual risks accumulate increasing the overall chance that at least one test will falsely reject the null hypothesis

The idea is still the same, we test whether the observed variation between groups is significantly greater than the variation within groups; but now we do this across multiple variables

It's all start with null hypothesis, the null hypothesis H₀ is a default assumption or baseline statement

To test wether the null hypothesis is true or not, we have to perform a test statistic that we choose from the four above depending on the data

Here's how the process works:

1. State the hypotheses:
   -Null hypothesis (H₀): all group means are equal across all dependent variables. So any observed differences across groups are due to random chance
   -Alternative hypothesis (H₁): At least one group differs significantly from the others on some variables

2. Calculate the MANOVA statistic
Unlike ANOVA, MANOVA uses different test statistic which are not simple F test :
Wilks’ lambda (most used one)
-Tests the proportion of variance in the DVs not explained by group differences
-Can be converted into an approximate F-statistic for significance testing

Pillai’s trace
-More robust to violations of MANOVA assumptions (like non-normality or unequal covariance)
-Measures the explained variance by the model
-Often preferred when assumptions are not well met

Hotelling’s trace

Roy’s largest toot
-Most powerful when group differences are large along a single dimension, but less robust overall

3. Compute the p-value, which tells us what is the probability of getting such a large test statistic if all group means were actually equal for all variables (i.e if the null hypothesis is true)

4. Compare the p-value with the significance level alpha (typically 0.05):
   - If p < alpha -> Reject the null hypothesis (the difference is statistically significant and at least one group is different in term of characteristic)
   - If p ≥ alpha -> Fail to reject the null hypothesis because the difference could be due to chance
   i.e the p-value is the lowest alpha for which we reject the null hypothesis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("...your_path/MANOVA.csv")

df.head()

sns.boxplot(x='Blood_Pressure_Abnormality', y='Age', data=df)

plt.show()

### HYPOTHESIS
"""
MANOVA assumes the following hypothesis

Multivariate Normality
-The dependent variables should follow a multivariate normal distribution within each group
-That means not only should each DV be normally distributed, but their joint distribution should also be normal

This assumption is most important when sample sizes are pretty small, when dealing with large samples, MANOVA is fairly robust

Homogeneity of Covariance Matrices
The variance/covariance matrices of the dependent variables should be equal across all groups
(This is simply an extension of the homogeneity of variance assumption in ANOVA)
(The Box’s M test is typically used to test this assumption)

Independence of Observations
Each observation must be independent of the others

Linearity
There should be linear relationships among all pairs of dependent variables within each group

No Multicollinearity / Singularity
The dependent variables should not be too highly correlated (multicollinearity), and no DV should be a perfect linear combination of others (singularity)


Based on violation or not of these assumptions, we should consider the right test statistic
If we do care about robustness to violations (like unequal covariances, non-normality) : we use Pillai’s trace

If we believe most of the difference is concentrated in 1 principal component : we should use Roy’s largest root

If assumptions are met (normality, homogeneity of covariance...): Wilks’ lambda is fine
"""

plt.close('all')
y_name = 'salt_content_in_the_diet'#Change y to other continuous DV so we can check that DV are normally distributed
#salt_content_in_the_diet Physical_activity BMI Age
sns.boxplot(x='Blood_Pressure_Abnormality', y=y_name, data=df)
plt.ylabel(y_name)
plt.xlabel('Blood_Pressure_Abnormality')

plt.tight_layout()
plt.show()

"""
By simple looking at continuous DV distribution, they seem to have a normal distribution, so we are simply going to use Wilks’ lambda
"""

### PREPARE THE METRICS
"""
All statistics are based on the eigenvalues lambda_i of the matrix :
inverse(E) * H

With H the hypothesis SSCP matrix and E the error SSCP matrix

E measures variability of observations within each group, relative to the group mean : SUM( SUM([Vector x(ij) - Vector mean(i)] * transpose[Vector x(ij) - Vector mean(i)])) for i representing group i; and j for each data in group i
The first SUM is indexed on i and the SUM within the first SUM is indexed on j
This formula is sorta of a sum of squares but for multiple variables

H measures variability between the group means, relative to the grand mean
So this is kinda the same as ANOVA but on multiple dimensions
H = SUM(ni * [Vector mean(i) - Vector grand mean] * transpose([Vector mean(i) - Vector grand mean])) with ni the number of observations in group i and vecotr mean i the vector representing means of DV for data in group i
Again, this is sorta a sum of squares to the grand mean but for multiple dimensions

For Wilks’ lambda, statistic value is computed by :
det(E)/det(E + H)
"""


group_col = 'Blood_Pressure_Abnormality'
dv_cols = ['Age', 'BMI', 'Physical_activity', 'salt_content_in_the_diet']

#We select necessary data
groups = df[group_col].unique()
grand_mean = df[dv_cols].mean().values.reshape(-1, 1)

#We initialize E and H
E = np.zeros((len(dv_cols), len(dv_cols)))
H = np.zeros((len(dv_cols), len(dv_cols)))

for group in groups:
    group_data = df[df[group_col] == group][dv_cols]
    n_i = group_data.shape[0]
    group_mean = group_data.mean().values.reshape(-1, 1)

    #Error matrix E
    centered = group_data.values - group_mean.T  # (n_i, p)
    E += centered.T @ centered

    #Hypothesis matrix H
    mean_diff = group_mean - grand_mean
    H += n_i * (mean_diff @ mean_diff.T)

print("E matrix is :", E)
print("H matrix is :", H)

from numpy.linalg import det
#We will not bother creating a det function, it will take a while for nothing

wilks_lambda = det(E) / det(E + H)
print("Wilks’ lambda value is :", wilks_lambda)

###CONVERTING TO A F TEST
"""
Wilks’ lambda is not by itsel a test statistic with p value
One way to get a p value from it is to transform Wilks' lambda into an approximate F statistic and then we compute the p value from the F distribution that stems from Wilks’ lambda

So to derive Wilks’ lambda to a F statistic, we need the following :
g: number of groups
p: number of dependent variables
N: total number of observations
df1 = p ( g − 1 )
df2 = N − 0.5 *​ (p + g + 1)

The two df represent the df for the numerator and the denominator of the following formula :
F = [1 - wilks_lambda**(1/s)] / wilks_lambda**(1/s) * df2 / df1

Where s = sqrt(p**2 * (g - 1)**2 - 4 / (p**2 + (g - 1)**2 - 5)) is the scaling factor used to normalize the test statistic based on the number of dependent variables and groups
It adjusts the shape of the distribution so we can better approximate the F-distribution from Wilks’ lambda but I really don't understand where it comes from : (
"""

from scipy.stats import f


wilks_lambda = det(E) / det(E + H)
N = df.shape[0]
g = df[group_col].nunique()
p = len(dv_cols)
df1 = p * (g - 1)
df2 = N - 0.5 * (p + g + 1)


numerator = (p ** 2) * (g - 1) ** 2 - 4
denominator = (p ** 2) + (g - 1) ** 2 - 5
s = np.sqrt(numerator / denominator)


F = ((1 - wilks_lambda ** (1/s)) / (wilks_lambda ** (1/s))) * (df2 / df1)


p_value = 1 - f.cdf(F, df1, df2)
print(f"If the null hypothesis were true, then the chance of observing a test statistic as extreme as {F} is {p_value} so very low\n")
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. The difference is statistically significant. So at least one group differ from the other regarding one or more variable.")
else:
    print("Fail to reject the null hypothesis. The difference could be due to chance.")


"""
This one took me time to understand and to write in order to properly explain the concept
"""

# Author GCreus
# Done via pyzo


