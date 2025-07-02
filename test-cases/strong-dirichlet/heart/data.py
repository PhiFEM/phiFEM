import numpy as np

def levelset(x):
    xs = x[0,:]
    ys = x[1,:]

    vals = xs**2 + (ys - (2.*(xs**2+np.abs(xs)-np.full_like(xs, 6.)))/(3.*(xs**2+np.abs(xs)+np.full_like(xs, 2.))))**2 - 36.
    return vals

def source_term(x):
    return np.ones_like(x[0,:])