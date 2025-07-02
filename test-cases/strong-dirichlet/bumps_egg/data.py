import numpy as np

angle = np.pi
coef = np.pi/angle/2.

def exp_fct(x):
    val = np.exp(-1000. * (x[0,:]**2 + x[1,:]**2))
    return val

def levelset(x):
    values = (x[0,:]**2 + x[1,:]**2 - np.ones_like(x[0,:]))
    values += 0.5 * exp_fct(x - np.tile(np.array([[0.5], [0.1], [0.]]), x.shape[1]))
    values += 0.5 * exp_fct(x - np.tile(np.array([[-0.7], [-0.7], [0.]]), x.shape[1]))
    values += 0.5 * exp_fct(x)
    return values

def source_term(x):
    return np.ones_like(x[0,:])