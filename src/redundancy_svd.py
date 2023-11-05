# My Utility : 

import numpy as np   

# normalization of the data
def norm_data(X):
    # Y = X/sqrt(N-1)
    X = X/np.sqrt(np.shape(X)[0] - 1)

    return X

# (Step 4) SVD of the data
def svd_data(X, params):
    # (1) Center the data
    # x = X(:, i), i = 1, ..., K
    # mean = mean(x)
    # X(:, i) = X(:, i) - means
    mean = np.mean(X, axis=0)
    X = X - mean

    # (2) Normalize the data
    Y = np.asarray(norm_data(X))

    # (3) Descompose Y data
    # [U V S] = svd(Y)
    # U = right ortogonal matrix of eigenvectors
    # V = left ortogonal matrix of eigenvectors
    # S = diagonal matrix of eigenvalues

    U, S, V = np.linalg.svd(Y)
    
    V = V[:,:int(params[2])] # Number of singular vectors to keep

    return V