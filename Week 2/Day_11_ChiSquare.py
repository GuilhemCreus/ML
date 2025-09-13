### Day 11 -- Statistics Basics
"""
After understanding the t-test, another fundamental statistical tool is the Chi-Squared Test (χ²). While the t-test compares means, the chi-squared test is all about frequencies—how often things happen compared to how often we expect them to.

The purpose of the chi-squared test is to measure whether there’s a significant difference between observed frequencies and expected frequencies in one or more categories.

This test is typically used with categorical data (i.e. data that can be put into categories like "Yes/No", "Red/Green/Blue", "Male/Female", etc.)

Just like in the t-test, we start with a null hypothesis (H₀); here the assumption is that there is no difference between observed and expected data. In other words:
Are any differences we see between the observed and expected frequencies are just due to chance ?

Goodness-of-Fit Test

Tests if a single categorical variable follows a certain expected distribution

Example: Is a die fair? Do all sides come up equally often?

Test of Independence (or Association)

Tests if two categorical variables are independent (not related)
"""

import pandas as pd
import matplotlib.pyplot as plt
