# importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


df = pd.read_csv("mushrooms.csv")
df.head(4)
colname = list(df)
"""for item in colname:
    val = df[item].unique()
    pos = np.arange(len(val))
    plt.hist(df[item])
    plt.title(item)
    ax = plt.axes()
    ax.set_xticks(pos + (1.0 / 2))
    ax.set_xticklabels(val)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()"""
cont = pd.crosstab(df[colname[3]], df[colname[5]])
print(chi2_contingency(cont))
