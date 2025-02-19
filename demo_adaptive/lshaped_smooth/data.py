import jax.numpy as jnp
import numpy as np

tilt_angle = np.pi/6.
shift = np.array([np.pi/32., np.pi/32.])

def rotation(angle, x):
    if x.shape[0] == 3:
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [            0,              0, 1]])
    elif x.shape[0] == 2:
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(np.asarray(x))

def line(x, y, a, b, c):
    rotated = rotation(tilt_angle + np.pi / 4., np.vstack([x, y]))
    return a*rotated[0,:] + b*rotated[1,:] + np.full_like(x, c)

def expression_detection_levelset(x):
    x_shift = x[0, :] - np.full_like(x[0, :], shift[0])
    y_shift = x[1, :] - np.full_like(x[1, :], shift[1])

    line_1 = line(x_shift, y_shift, -1.,  0.,   0.)
    line_2 = line(x_shift, y_shift,  0., -1.,   0.)
    line_3 = line(x_shift, y_shift,  1.,  0., -0.5)
    line_4 = line(x_shift, y_shift,  0.,  1., -0.5)
    line_5 = line(x_shift, y_shift,  0., -1., -0.5)
    line_6 = line(x_shift, y_shift, -1.,  0., -0.5)

    reentrant_corner = np.minimum(line_1, line_2)
    top_right_corner = np.maximum(line_3, line_4)
    corner           = np.maximum(reentrant_corner, top_right_corner)
    horizontal_leg   = np.maximum(corner, line_5)
    vertical_leg     = np.maximum(horizontal_leg, line_6)
    return vertical_leg

def expression_levelset(x):
    x_shift = x - np.tile(np.array([[shift[0]], [shift[1]], [0.]]), x.shape[1])
    x_rot = rotation(tilt_angle, x_shift)
    theta = np.arctan2(x_rot[1, :], x_rot[0, :])
    values = np.cos((2./3.) * theta)
    x_rot_sq = rotation(tilt_angle + np.pi/4., x_shift)
    values *= x_rot_sq[1, :] + np.full_like(x_rot_sq[1, :], 0.5)
    values *= x_rot_sq[1, :] - np.full_like(x_rot_sq[1, :], 0.5)
    values *= x_rot_sq[0, :] + np.full_like(x_rot_sq[0, :], 0.5)
    values *= x_rot_sq[0, :] - np.full_like(x_rot_sq[0, :], 0.5)
    return values

# def expression_levelset(x):
#     x_shift = x - np.tile(np.array([[shift[0]],[shift[1]], [0.]]), np.shape(x)[1])
#     x_rot   = rotation(tilt_angle, x_shift)
#     r = np.sqrt(np.square(x_rot[0, :]) + np.square(x_rot[1, :]))
#     theta = np.arctan2(x_rot[1, :], x_rot[0, :]) + np.full_like(x_rot[0, :], np.pi / 2.)
#     values = -r**(2. / 3.) * np.sin((2. / 3.) * theta)
#     values *= x_rot[1, :] + np.full_like(x_rot[1, :], 0.5)
#     values *= x_rot[1, :] - np.full_like(x_rot[1, :], 0.5)
#     values *= x_rot[0, :] + np.full_like(x_rot[0, :], 0.5)
#     values *= x_rot[0, :] - np.full_like(x_rot[0, :], 0.5)
#     return values

def expression_rhs(x):
    return np.ones_like(x[0, :])

# FEM data
point_1 = rotation(tilt_angle - np.pi / 4. - np.pi/3., np.array([  0.,   0.]).T) + shift
point_2 = rotation(tilt_angle - np.pi / 4. - np.pi/3., np.array([  0., -0.5]).T) + shift
point_3 = rotation(tilt_angle - np.pi / 4. - np.pi/3., np.array([ 0.5, -0.5]).T) + shift
point_4 = rotation(tilt_angle - np.pi / 4. - np.pi/3., np.array([ 0.5,  0.5]).T) + shift
point_5 = rotation(tilt_angle - np.pi / 4. - np.pi/3., np.array([-0.5,  0.5]).T) + shift
point_6 = rotation(tilt_angle - np.pi / 4. - np.pi/3., np.array([-0.5,   0.]).T) + shift

geom_vertices = np.vstack([point_1, point_2, point_3, point_4, point_5, point_6]).T