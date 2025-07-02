import numpy as np

# Levelset taken from 1st test case in: https://onlinelibrary.wiley.com/doi/abs/10.1002/num.22878
def levelset(x):
    R = 0.47
    theta0 = 0.
    r = np.sqrt(x[0,:]**2 + x[1,:]**2)
    theta = np.arctan2(x[1,:],x[0,:])
    val = r**4 * (np.full_like(r, 5.) + 3. * np.sin(7.*(theta-theta0) + 7. * np.pi/36.))/2. - R**4.
    return val

def source_term(x):
    return np.ones_like(x[0,:])