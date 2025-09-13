### Day 9 -- Statistics Basics
"""
Before going further in our journey of machine learning, we need to study the basics of statistics and their meaning like p-value, R**2, t-test...

Let's start with t-test, the most famous one

The purpose of t test is to measure if the difference between the mean of two data groups (or one data group with a theorical mean) is due to noise or if the difference is significant enough so that noise definitely cannot explain it

It's all start with null hypothesis, the null hypothesis H₀ is a default assumption or baseline statement that there is no effect, no difference or no relationship between variables

To test wether the null hypothesis is true or not, we have to perform a t-test

The t-test compares the means of two groups (or, again, the mean of a group and the theorical expected mean) and asks:
'Is the observed difference big enough that it's unlikely to have happened by chance ?'

There are a few types of t-tests, but the most common one is:
Independent t-test: Compares two independent groups
-To do so, the data in each group should approximately follow a normal distribution
-Observations must be independent
-The two populations studied are assumed to have the same variance

Here's how the process works:

1. State the hypotheses:
   -Null hypothesis (H₀): μ1 = μ2
   -Alternative hypothesis (H₁): μ1 ≠ μ2

2. Calculate the t-statistic, which measures how many standard errors apart the means are; more precisely, it tells us how strong is the difference between the means is based on the variability of the data

3. Compute the p-value, which tells us the probability of observing such a difference or more extreme, assuming H₀ is true.

4. Compare the p-value with the significance level alpha (typically 0.05):
   - If p < alpha -> Reject the null hypothesis (the difference is statistically significant)
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
df = pd.read_csv("...your_path/drug_t_test.csv.csv")

df.head()

group_A = df[df['Group'] == 'Drug A']['Blood Pressure Reduction']
group_B = df[df['Group'] == 'Drug B']['Blood Pressure Reduction']

### PREPARE THE METRICS
"""
We now want to setup the functions that will help us find t value and pooled standard deviation of two data groups

Let's recall that for two data groups :
t = (mean1 - mean2) / (sp * (1/n1 + 1/n2)**(1/2))

T follows a Student's distribution because
-If both samples are approximately normally distributed, then mean1 - mean2 is normally distributed
-And sp**2 follows a chi squared distribution distribution because it stems from the sum of two independent sum of squared of normals
Finally, we have a result that stems from : normal/sqrt(chi squared / df) (or normal/sqrt(normalized chi squared) which follows a Student's distribution


with sp the pooled standard deviation of two data groups :
sp = sqrt( [(n1 - 1) * s1**2 + (n2 - 1) * s2**2] / degree of freedom )

and degree of freedom = n1 + n2 - 2

Here, the null hypothesis assumes no difference between Drug A and Drug B
"""
nA = len(group_A)
nB = len(group_B)

degree_of_freedom = nA + nB - 2

varA = group_A.var(ddof=1)  # divide by n - 1 to prevent Bessel bias (sample)
varB = group_B.var(ddof=1)

meanA = group_A.mean()
meanB = group_B.mean()

sp = ( ((nA-1)*varA + (nB-1)*varB) / (nA + nB - 2) )**(0.5)
t_value = (meanA - meanB) / (sp * ((1/nA + 1/nB)**(0.5)))


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

But for the p value, recall that the p-value is the probability of observing a t-statistic as extreme or more extreme than the one we got (assuming the null hypothesis is true)
It equals : P(T <= -abs(t)) + P(T >= abs(t)

What does the formula above means ?
It means that under the assumption that the null hypothesis is true, p equals the probability that we find a t value RANDOMLY more extreme than the one we got; hence T, the student random variable under the assumption that the null hypothesis is true (depends on the degree of freedom of our problem)

So p value tells us what is the probability that we can randomly find a t (following a Student's distribution, i.e T) value more extreme than the one we got, still under the null hypothesis assumption, so if the p-value is less than the chosen significance level (alpha), it suggests that such an extreme result would be very unlikely if the null hypothesis were true so we reject the null hypothesis

Here our distributions are symmetric so :
p = 2 * P(T >= abs(t)

to calculate P(T >= abs(t) we will use the CDF of T from scipy.stats
"""
from scipy.stats import t

p_value = 2 * (1 - t.cdf(abs(t_value), df=degree_of_freedom))
print(f"If the null hypothesis were true, then the chance of observing a test statistic as extreme as {t_value} is {p_value} so very low\n")
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis, the difference is statistically significant.")
else:
    print("Fail to reject the null hypothesis, the difference could be due to chance.")

"""
Other main usage of t test are :
-paired t test where we gather different measures of data of similar units (i.e the object we observe) or group of units across changes like prior and after an intervention
-t test for a single sample where we compare the significance of the mean of the sample with a reference
"""

# Author GCreus
# Done via pyzo
