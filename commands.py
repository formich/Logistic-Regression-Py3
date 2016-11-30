import numpy as np
import matplotlib.pyplot as plt
import liblogfit as lf
import pandas as pd

fram = pd.read_csv("framingham.csv")
fram.dropna(inplace=True) # remove observations containing NAs
X = np.ones((100,3)) # the first column will be for the intercept term
X[:,1:] = lf.normalize(fram[['age', 'glucose']].values[:100]) # features
Y = fram['TenYearCHD'].values[:100] # labels
th = lf.logfit(X, Y, alpha=0.1, itr=100) # logistic regression
print(th)
P = lf.h(th, X) # predicted probabilities
pd.crosstab(P > 0.2, Y) # confusion matrix

tpr, fpr = lf.tprfpr(P, Y) # True Positive Rate, False Positive Rate
plt.plot(fpr, tpr) # ROC curve
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show(block=False)

print("AUC = ", lf.auc(fpr, tpr)) # Area Under the Curve
