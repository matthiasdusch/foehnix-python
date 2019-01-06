import numpy as np

def logitprob(x, alpha):
    x = alpha[0] + x * alpha[1]
    return np.exp(x) / (1 + np.exp(x))
