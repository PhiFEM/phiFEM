import numpy as np

def levelset(x):
    return 0.25**2-(x[0,:]**2+x[1,:]**2)

def source_term(x):
    return np.zeros_like(x[:2,:])

def neumann(x):
    vals = np.ones_like(x[:2,:])
    vals[1,:] = -1.
    return vals