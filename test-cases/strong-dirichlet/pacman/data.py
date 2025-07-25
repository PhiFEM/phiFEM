import numpy as np

shift = np.array([0., np.pi/32.])
angle = 99. * np.pi/100.
coef = np.pi/angle/2.

# Levelset taken from 1st test case in: https://onlinelibrary.wiley.com/doi/abs/10.1002/num.22878
def levelset(x):
    x_shift = x - np.tile(np.array([[shift[0]], [shift[1]], [0.]]), x.shape[1])
    theta = np.arctan2(x_shift[1, :], x_shift[0, :])
    values = 100. * (theta + angle) * (theta - angle) * (x_shift[0,:]**2 + x_shift[1,:]**2 - np.ones_like(x[0,:]))
    return values

def detection_levelset(x):
    x_shift = x - np.tile(np.array([[shift[0]], [shift[1]], [0.]]), x.shape[1])
    theta = np.arctan2(x_shift[1, :], x_shift[0, :])
    values = 100. * np.maximum((theta + angle) * (theta - angle), (x_shift[0,:]**2 + x_shift[1,:]**2 - np.ones_like(x[0,:])))
    return values

def source_term(x):
    return np.ones_like(x[0,:])