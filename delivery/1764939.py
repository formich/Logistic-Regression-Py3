import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np


def z(x):
    """Evaluate the sigmoid function at x."""
    return 1.0/(1.0 + np.exp(-x))


def h(Theta, X):
    """Evaluate the sigmoid function at each element of <Theta,X>."""
    return np.array([z(np.dot(Theta, x)) for x in X])


def gradient(Theta, X, Y):
    """Compute the gradient of the log-likelihood of the sigmoid."""
    pX = h(Theta, X) # i.e. [h(x) for each row x of X]
    return np.dot((Y - pX), X)


def logfit(X, Y, alpha=1, itr=10):
    """Perform a logistic regression via gradient ascent."""
    Theta = np.zeros(X.shape[1])
    for i in range(itr):
        Theta += alpha * gradient(Theta, X, Y)
    return Theta


def normalize(X):
    """Normalize an array, or a dataframe, to have mean 0 and stddev 1."""
    return (X - np.mean(X, axis=0))/(np.std(X, axis=0))


def tprfpr(P, Y):
    """Return the False Positive Rate and True Positive Rate vectors of the given classifier."""
    Ysort = Y[np.argsort(P)[::-1]]
    ys = np.sum(Y)
    tpr = np.cumsum(Ysort)/ys # [0, 0, 1, 2, 2, 3,..]/18
    fpr = np.cumsum(1-Ysort)/(len(Y)-ys)
    return (tpr, fpr)


def auc(fpr, tpr):
    """Compute the Area Under the Curve (AUC) given vectors of
    false positive rate and true positive rate"""
    return(np.diff(tpr) * (1 - fpr[:-1])).sum()

# Load File
file_train = sys.argv[1]
file_test = sys.argv[2]

# load csv as dataframes
df_training = pd.read_csv(file_train)
df_testing = pd.read_csv(file_test)

# Generate matrix of features
X = np.ones((df_training.shape[0], df_training.shape[1]))
X[:, 1:] = df_training.drop("TenYearCHD", axis=1)

Y = df_training.values[:, -1]
th = logfit(X, Y, alpha=0.001, itr=100)
print(th)
P = h(th, X)

# draw plot and save PNG file
tpr, fpr = tprfpr(P, Y)
plt.plot(fpr, tpr)
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig('1764939-roc.png')

# Print AUC
print(auc(fpr, tpr))

X_test = np.ones((df_testing.shape[0], df_testing.shape[1]+1))
X_test[:, 1:] = df_testing.values
P = h(th, X_test)
print(P)
