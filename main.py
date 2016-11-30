import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import liblogfit as lf

# Load csv into dataframe
df = pd.read_csv("resources/framingham.csv")

# remove NaN
df = df.dropna()

# normlize
df_bool = df[["TenYearCHD"]]
df_no_bool = lf.normalize(df.drop("TenYearCHD", axis=1))

# split dataframe
df_training = pd.concat([df_no_bool.head(df_no_bool.shape[0] - 20), df_bool.head(df_bool.shape[0] - 20)], axis=1)
df_testing = df_no_bool.tail(20)

# save dataframe as csv
df_training.to_csv("resources/fram-train.csv")
df_testing.to_csv("resources/fram-test.csv")

print(df_training)
print(df_testing)

# Generate matrix of features
X = np.ones((df_training.shape[0], df_training.shape[1]))
X[:, 1:] = df_training.drop("TenYearCHD", axis=1)

Y = df_training.values[:, -1]
th = lf.logfit(X, Y, alpha=0.001, itr=100)
print(th)
P = lf.h(th, X)

# draw plot
tpr, fpr = lf.tprfpr(P, Y)
plt.plot(fpr, tpr)
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig('1764939-roc.png')

print(lf.auc(fpr, tpr))

X_test = np.ones((df_testing.shape[0], df_testing.shape[1]+1))
X_test[:, 1:] = df_testing.values
P = lf.h(th, X_test)
print(P)
