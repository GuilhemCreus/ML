### Day 10 -- Statistics Basics
"""
Now that we have seen the purpose of the t-test and p value, let's move on F test and AMOVA based on the same dataset

The F-test is used to compare variances between two or more groups or to assess whether a statistical model fits the data better than another. In general, the F-test is about testing if the variability between groups is significant compared to the variability within groups

It's all start with null hypothesis, the null hypothesis H₀ is a default assumption or baseline statement that there is no significant variation within the data

To test wether the null hypothesis is true or not, we have to perform a F-test

The F-test tells us whether or not the differences in variability between groups (or models) are too large to be due to chance only

Here's how the process works:

1. State the hypotheses:
   -Null hypothesis (H₀): all group means are equal, i.e any observed differences are due to random variation (noise)
   -Alternative hypothesis (H₁): there is variation between groups (at least one group mean is different)

2. Calculate the F-statistic, which measures variability between groups based on the variability within groups

3. Compute the p-value, which tells us what is the probability of getting such a large F-statistic if all group means were actually equal (i.e if the null hypothesis is true)

4. Compare the p-value with the significance level alpha (typically 0.05):
   - If p < alpha -> Reject the null hypothesis (the difference is statistically significant and at least one group is different in term of characteristic)
   - If p ≥ alpha -> Fail to reject the null hypothesis because the difference could be due to chance
   i.e the p-value is the lowest alpha for which we reject the null hypothesis
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("...your_path.csv")

df.head()

group_A = df[df['Group'] == 'Drug A']['Blood Pressure Reduction']
group_B = df[df['Group'] == 'Drug B']['Blood Pressure Reduction']
group_C = df[df['Group'] == 'Drug C']['Blood Pressure Reduction']

### PREPARE THE METRICS
"""
We now want to setup the functions that will help us find F value

Let's recall that for two or more data groups :
F = Mean Square between groups (variance due to group differences) / Mean Square within groups (variance within groups, i.e. random/error variance)

With :
-Mean Square between groups = Sum of Squares between groups / degree of freedom between groups
-Sum of Squares between groups = SUM(ni * (mean(xi) - mean(x)**2)
(ni number of observations in group i; xi the values in group i and x the values in overall observations)
-degree of freedom between groups = k - 1 (k the amount of groups); we substract 1 because there is one equation linking all groups means which is overall mean

-Mean Square within groups = Sum of Squares within groups / degree of freedom within groups
-Sum of Squares within groups = SUM(squared deviation i)
(squared deviation i the squared deviation within group i, so explicitly SS = SUM(SUM(xij - mean(xi)**2))
-degree of freedom within groups = N - k (N the amount of observations overall and k the amount of groups); we substract k because there are k equations data points with means

Here, the null hypothesis assumes no significant differences between variability of groups (Drug A, Drug B & Drug C
"""
meanA = group_A.mean()
meanB = group_B.mean()
meanC = group_C.mean()

nA = len(group_A)
nB = len(group_B)
nC = len(group_C)

overall_mean = (nA * meanA + nB * meanB + nC * meanC) / (nA + nB + nC)

dfbetween = 2

MSbetween = (nA * (meanA - overall_mean)**2)
MSbetween += (nB * (meanB - overall_mean)**2)
MSbetween += (nC * (meanC - overall_mean)**2)
MSbetween /= dfbetween

squared_deviationA = ((group_A - meanA)**2).sum()
squared_deviationB = ((group_B - meanB)**2).sum()
squared_deviationC = ((group_C - meanC)**2).sum()

dfwithin = nA + nB + nC - 3

MS_within = squared_deviationA
MS_within += squared_deviationB
MS_within += squared_deviationC
MS_within /= dfwithin

F = MSbetween / MS_within

"""
We use sns here because seaborn is more specific to plotting statistical problems
"""
sns.boxplot(x='Group', y='Blood Pressure Reduction', data=df)
sns.swarmplot(x='Group', y='Blood Pressure Reduction', data=df, color='black', alpha=0.7) #alpha changes the transparency level of the dots

plt.title('Blood Pressure Reduction by Drug Group')
plt.ylabel('Reduction in Blood Pressure')
plt.xlabel('Drug')

plt.tight_layout()
plt.show()

### CALCULATING P-VALUE AND ALPHA
"""
The hard part will be to calculate p-value since we ususally chose and alpha = 0.05 which means that we assume a 5% risk of rejecting a true null hypothesis

But for the p value, recall that the p-value is the probability of observing a F-statistic as extreme or more extreme than the one we got (assuming the null hypothesis is true)
And, unlike the t-distribution, the F-distribution is NOT symmetric; it is right-skewed, and only positive values make sense
So p-value equals : P(F >= abs(F_calculated under H₀))

What does the formula above means ?
This is the probability that an F-distributed random variable (under H₀)
takes a value greater than the observed F-statistic.
If this probability is low, it suggests that the observed group differences
are unlikely to be due to chance -> we reject the null hypothesis.

So p-value tells us what is the probability that we can randomly find a F value (based on the Fisher–Snedecor distribution) more extreme than the one we got, still under the null hypothesis assumption, so if the p-value is less than the chosen significance level (alpha), it suggests that such an extreme result would be very unlikely if the null hypothesis were true so we reject the null hypothesis

to calculate P(F >= abs(f) we will use the CDF of F from scipy.stats
"""
from scipy.stats import f

p_value = f.sf(F, dfbetween, dfwithin)
print(f"If the null hypothesis were true, then the chance of observing a test statistic as extreme as {F} is {p_value} so very low\n")
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. The difference is statistically significant. So at least one group differ from the other regarding its proper characteristic.")
else:
    print("Fail to reject the null hypothesis. The difference could be due to chance.")

"""
What we have done here is an ANOVA : ANalysis Of VAriance which is an F test but it's a standardized way to apply F-tests to multiple group means
So when we do an ANOVA for only two groups, we are doing a simple F-test which is itself is the squared of a t-statistic, i.e F = t**2 for two groups
"""


# Author GCreus
# Done via pyzo
