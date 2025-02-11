import itertools
import numpy as np
import pytest
from phiFEM.phifem.poisson_dirichlet import FEMRefinementLoop
from phiFEM.phifem.continuous_functions import Levelset

"""
Data_n° = ("Fct formula", function of var x, geom_vertices (np.array shape (N,2)), bbox (np.array shape (2,2)))
"""
# Tilted square
def expression_1(x):
    return np.abs(x[0, :]) + np.abs(x[1, :]) - np.ones_like(x[0, :])
geom_vertices_1 = np.array([[1., 0., -1., 0.],
                            [0., 1., 0., -1.]])
bbox_1 = np.array([[-1.1, 1.1],
                   [-1.1, 1.1]])

# Unit circle
def expression_2(x):
    return x[0, :]**2 + x[1, :]**2 - np.ones_like(x[0, :])
bbox_2 = np.array([[-1.1, 1.1],
                   [-1.1, 1.1]])

tilt_angle = np.pi/6. + np.pi/2.

def rotation(angle, x):
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

def line(x, a, b, c):
    rotated = rotation(-tilt_angle, x)
    return a*rotated[0] + b*rotated[1] + np.full_like(x[0, :], c)

def expression_3(x):
    val = x - np.full_like(x, np.pi/32.)
    line_1 = line(val, -1.,  0.,   0.)
    line_2 = line(val,  0., -1.,   0.)
    line_3 = line(val,  1.,  0., -0.5)
    line_4 = line(val,  0.,  1., -0.5)
    line_5 = line(val,  0., -1., -0.5)
    line_6 = line(val, -1.,  0., -0.5)

    reentrant_corner = np.minimum(line_1, line_2)
    top_right_corner = np.maximum(line_3, line_4)
    corner           = np.maximum(reentrant_corner, top_right_corner)
    horizontal_leg   = np.maximum(corner, line_5)
    vertical_leg     = np.maximum(horizontal_leg, line_6)
    return vertical_leg

geom_vertices_3_noshift = np.array([[0., 0.5, 0.5, -0.5, -0.5,   0.],
                                    [0.,  0., 0.5,  0.5, -0.5, -0.5]])
geom_vertices_3_rot = rotation(np.pi/6., geom_vertices_3_noshift)
geom_vertices_3 = geom_vertices_3_rot + np.pi/32.

bbox_3 = np.array([[-0.9, 0.3],
                   [-0.7, 0.8]])

data_1  = ("|x|+|y|-1",
           expression_1,
           geom_vertices_1,
           bbox_1)
data_2  = ("x^2+y^2-1",
           expression_2,
           None,
           bbox_2)
data_3  = ("l-shaped",
           expression_3,
           geom_vertices_3,
           bbox_3)

testdata = [data_1, data_2, data_3]

@pytest.mark.parametrize("lc,data", itertools.product([0.1, 0.01, 0.005], testdata))
def test_mesh2d_from_levelset(lc, data):
    data_name     = data[0]
    levelset      = Levelset(data[1])
    geom_vertices = data[2]
    bbox          = data[3]

    x_min, x_max = -1., 1.
    y_min, y_max = -1., 1.
    nx = 1000
    ny = 1000
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)

    xx, yy = np.meshgrid(xs, ys)
    xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
    yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
    xxx = np.vstack([xx_rs, yy_rs])
    zz_rs = data[1](xxx)
    zz = zz_rs.reshape(xx.shape)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.contour(xx, yy, zz, [0.], linewidths=0.2)
    if geom_vertices is not None:
        ax.scatter(geom_vertices[0, :], geom_vertices[1, :])
    fig.savefig(f"{data_name}_{str(lc)}.png")

    dummy_loop = FEMRefinementLoop(lc,1,"uniform",levelset,"",geometry_vertices=geom_vertices, save_output=False)
    dummy_loop.set_bbox(bbox)
    
    boundary_vertices = dummy_loop.mesh2d_from_levelset()

    err_max_to_boundary = np.max(np.abs(levelset(boundary_vertices)))

    # We check if the distance to the boundary is at most half of the characteristic length of the points cloud.
    assert np.less(err_max_to_boundary/lc, 0.5), f"Error data {data_name}. Maximum relative error to the boundary: {err_max_to_boundary/lc}"

if __name__=="__main__":
    test_mesh2d_from_levelset(0.1, data_3)
