import numpy as np

# Levelset
tilt_angle = np.pi/6.
def _rotation(angle, x):
    if x.shape[0] == 3:
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle),  np.cos(angle), 0],
                       [             0,               0, 1]])
    elif x.shape[0] == 2:
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(np.asarray(x))

def detection_levelset(x):
    y = np.sum(np.abs(_rotation(-tilt_angle + np.pi/4., x)), axis=0)
    return y - np.sqrt(2.)/2.

def levelset(x):
    vect = np.full_like(x, 0.5)
    val = -np.sin(np.pi * (_rotation(-tilt_angle, x - _rotation(tilt_angle, vect))[0, :])) * \
           np.sin(np.pi * (_rotation(-tilt_angle, x - _rotation(tilt_angle, vect))[1, :]))
    return val

# Analytical solution
def exact_solution(x):
    return np.sin(2. * np.pi * _rotation(-tilt_angle, x)[0, :]) * \
           np.sin(2. * np.pi * _rotation(-tilt_angle, x)[1, :])

# Dirichlet data
def dirichlet(x):
    return exact_solution(x)

# Source term
def source_term(x):
    return 8. * np.pi**2 * exact_solution(x)