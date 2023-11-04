# My Utility : 

import numpy as np   

# (Step 1) Information gain
def inform_gain(X, Y):
    return inform_estimate(Y) - entropy_xy(X, Y)

# (Step 2) Estimation of information
def inform_estimate(Y):
    # I(Y) = I(d1, d2, ..., dn) = -sum(p * log2(p)
    # p = di/N = probability of each class

    unique_y, counts_y = np.unique(Y, return_counts=True) # I(d1, d2, ..., dn)
    p = counts_y/np.sum(counts_y) # probability of each class
    return entropy(p)

# (Step 3) Entropy of the variables  
def entropy_xy(X, Y):
    # E(x) = Sum j=1 -> B: (d1j + d2j + ... + dnj)/N * I(d1j, d2j, ..., dnj)
    # B = number of bins (intervals) of the variable x of the dataset X = sqrt(N)
    # d1j, d2j, ..., dnj = number of data points in bin j of variable x

    # Get the entropy of X
    unique_x, counts_x = np.unique(X, return_counts=True) # I(d1j, d2j, ..., dnj)
    p_x = counts_x/np.sum(counts_x) # probability of each class
    entropy_x = entropy(p_x)

    # Get the entropy of Y
    unique_y, counts_y = np.unique(Y, return_counts=True) # I(d1j, d2j, ..., dnj)
    p_y = counts_y/np.sum(counts_y) # probability of each class
    entropy_y = entropy(p_y)

    return entropy_x + entropy_y # E(X) = E(x) + E(y)

# Entropy formula
def entropy(p):
    return -np.sum(p * np.log2(p))