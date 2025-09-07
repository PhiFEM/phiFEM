import numpy as np

# Levelset
tilt_angle = np.pi/6.
def _rotation(angle, x):
    if x.shape[0] == 3:
        R = np.array([[np.cos(angle),   np.sin(angle), 0],
                      [-np.sin(angle),  np.cos(angle), 0],
                      [            0,               0, 1]])
    elif x.shape[0] == 2:
        R = np.array([[np.cos(angle),   np.sin(angle)],
                      [-np.sin(angle),  np.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(np.asarray(x))

def detection_levelset(x):
    y = np.sum(np.abs(_rotation(tilt_angle - np.pi/4., x)), axis=0)
    return y - np.sqrt(2.)/2.

def levelset(x):
    vect = np.full_like(x, 0.5)
    val = -np.sin(np.pi * (_rotation(tilt_angle, x - _rotation(-tilt_angle, vect)))[0, :]) * \
           np.sin(np.pi * (_rotation(tilt_angle, x - _rotation(-tilt_angle, vect)))[1, :])
    return val

# Analytical solution
def exact_solution(x):
    return np.cos(2. * np.pi * _rotation(tilt_angle, x)[0, :]) * \
           np.cos(2. * np.pi * _rotation(tilt_angle, x)[1, :])

# Source term
def source_term(x):
    return 8. * np.pi**2 * exact_solution(x) + exact_solution(x)

robin_coef = 1.
def robin_data(x):
    rx = _rotation(tilt_angle, x)

    def _dx(rx):
        return - 2. * np.pi * np.sin(2. * np.pi * rx[0,:]) * \
                              np.cos(2. * np.pi * rx[1,:])
    def _dy(rx):
        return - 2. * np.pi * np.cos(2. * np.pi * rx[0,:]) * \
                              np.sin(2. * np.pi * rx[1,:])

    vals = -_dy(rx)
    mask = np.where(np.abs(rx[1,:]) < rx[0,: ])[0]
    vals[mask] = _dx(rx[:,mask])
    mask = np.where(np.abs(rx[0,:]) < rx[1,:])[0]
    vals[mask] = _dy(rx[:, mask])
    mask = np.where(np.abs(rx[1,:]) < -rx[0,: ])[0]
    vals[mask] = -_dx(rx[:, mask])

    vals += robin_coef * exact_solution(x)
    return vals