import numpy as np

angle = np.pi
coef = np.pi/angle/2.

def exp_fct(x):
    val = np.exp(-1000. * (x[0,:]**2 + x[1,:]**2))
    return val

def levelset(x):
    theta = np.arctan2(x[1, :], x[0, :])
    values = (x[0,:]**2 + x[1,:]**2 - np.ones_like(x[0,:]))
    values += 0.7 * np.cos(coef * theta)
    return values

def source_term(x):
    return np.ones_like(x[0,:])