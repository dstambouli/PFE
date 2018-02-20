# importing libraries
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn import tree

df = pd.read_csv("mushrooms.csv")
df.head(4)
colname = list(df)


#histogram of frequency
"""def histPlots(colname):
    for item in colname:
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
def hito(tab):
    for item in tab:
        g = sns.countplot(df[item])
        plt.show()

# Contengency table

def contTab(tableau):
    pval = np.zeros(len(tableau)-1)
    tab = []
    for i in range(len(tableau)-1):
        temp = pd.crosstab(df[tableau[i+1]], df[tableau[0]])
        tab.append(temp)
        np.append(pval,chi2_contingency(temp)[1])
    return pval

#pValue = contTab(colname)
#print(len(pValue[pValue <= 0.05]))
#his = hito(colname)

y = df['class']
x = df.drop(labels = ["class"],axis = 1)
x = pd.get_dummies(x)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)


dot_data = tree.export_graphviz(clf, out_file=None)
#dot_data = tree.export_graphviz(clf, out_file=None, feature_names=colname[1:],class_names=colname[0], filled=True, rounded=True)
