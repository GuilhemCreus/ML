### Day 11 -- Statistics Basics
"""
After understanding the t-test, we take a look at another fundamental statistical tool which is the chi squared test. While the t test compares means, the chi squared test is all about comparing frequencies between observations and expectations, i.e how often things happen compared to how often we expect them to

The purpose of the chi squared test is to measure whether there is a significant difference between the observed frequencies and the expected frequencies

Just like in the t-test, we start with a null hypothesis, here the assumption is that there is no difference between observed and expected data
So it answers : is any difference between the observed and expected frequencies is just due to chance ?

There are two main types of chi squared test :
1) Goodness of fit test
Tests if a single categorical variable follows a certain expected distribution

2) Test of independence (or Association)
Tests if two categorical variables are really independent

For today, we will focus on the test of independence
The assumptions here are :
-The expected frequencies are large enough (see further)
-The data must come from a well defined population
-Each observation must appear only once in one cell of the table without overlap (independence of observations)
-Observations are counts of observations (no percentage...)
-Homogeneity of variance

Here's how the process works :

1. State the hypotheses:
    -Null hypothesis : The variables are independent
    -Alternative hypothesis : The variables are dependent

2. Calculate the chi squared statistic

3. Determine the degrees of freedom of the problem (differs for goodness of fit and test of independence)

4. Compute the p-value, which tells us what is the probability of getting such a large chi squared statistic if the null hypothesis is true

5. Compare the p-value with the significance level alpha (typically 0.05):
   - If p < alpha -> Reject the null hypothesis (the difference is statistically significant and at least one group is different in term of characteristic)
   - If p ≥ alpha -> Fail to reject the null hypothesis because the difference could be due to chance
   i.e the p-value is the lowest alpha for which we reject the null hypothesis
"""

import pandas as pd
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("...your_path/chi-squared.csv")

df.head()

### PREPARE THE METRICS
"""
We now want to setup the functions that will help us find chi squared value

Let's recall that for a contingency table :
chi-squared = SUM( (Oij - Eij)**2 / Eij )
With Oij the actual data in row i column j, and Eij the expected value in row i column j
And Eij = (total for row i * total column j)/total for both

Recall that a chi squared distribution arises naturally when we sum the squares of independent standard normal variables
Under the null hypothesis, the observed values are the expected ones (or the expected ones + some noise), and with sufficient data amount we can apply the central limit theorem which tells us that the Oij - Eij behaves like a normal distribution because E(Oij) = Eij under H0 and Var(Oij) = Eij under central limit theorem; because Oij is a sum of counts, i.e a sum of 1 or 0 that follow a bernoullis with probability pij = Eij/N so Oij ​∼ Binomial(n,pij​) ≈ N(Eij​,Var(Oij​))
And Var(Oij) = Eij * (1 - Eij / N) ​∼ Eij for N large enough
So (Oij - Eij) / sqrt(Eij) ​∼ N(0, 1)

Then we will compute the degrees of freedom = (number of row - 1) * (number of column - 1)

Here we want to test whether political affiliation is independent of ethnicity
"""

def expected_value_i_j(i: int, j: int) -> float:
    """
    Ags :
        i = the index of the row we are looking at
        j = the index of the column we are looking at

    Return the value of the expected value at row i and column j
    """
    row_total = df.loc[i, "row_totals"]
    column_totals = df[["democrat", "independent", "republican"]].sum()
    total = column_totals.sum()

    column_name = ["democrat", "independent", "republican"][j]

    col_total = column_totals[column_name]
    expected = (row_total * col_total) / total

    return expected

def observed_value_i_j(i: int, j: int) -> float:
    """
    Ags :
        i = the index of the row we are looking at
        j = the index of the column we are looking at

    Return the value of the observed value at row i and column j
    """
    column_name = ["democrat", "independent", "republican"][j]
    return df.iloc[i][column_name]

def chi_squared() -> float:
    chi_squared = 0
    for i in range(df.shape[0] - 1): # we remove total for each row
        for j in range(df.shape[1] - 2): # we remove ethnicity and column total for each column
            chi_squared += (expected_value_i_j(i, j) - observed_value_i_j(i, j))**2 / expected_value_i_j(i, j)

    return chi_squared

degrees_of_freedom = (5 - 1) * (3 - 1) #5 ethnicity and 3 rows

### CALCULATING P-VALUE AND ALPHA
"""
As usual, we use an alpha of 0.05, and we compute the p value using the CDF of the chi squared distribution : p value = 1 - CDF(chi squared, df)
"""

from scipy.stats import chi2
p_value = chi2.sf(chi_squared(), degrees_of_freedom)

print(f"If the null hypothesis were true, then the chance of observing a test statistic as extreme as {chi_squared()} is {p_value} so very high\n")

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis because there is a statistically significant association between ethnicity and political vote")
else:
    print("Fail to reject the null hypothesis, no evidence of dependence")


# Author GCreus
# Done via pyzo
