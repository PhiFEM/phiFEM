import os

import dolfinx as dfx
import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx.fem import assemble_scalar
from dolfinx.io import XDMFFile
from mpi4py import MPI

from phifem.mesh_scripts import compute_tags_measures

"""
Data_n° = ("Data name", "mesh name", levelset object, levelset discretization degree, "benchmark values", integrand)
"""


def integrand_1(n):
    nx = ufl.dot(ufl.as_vector((1, 0)), n)
    ny = ufl.dot(ufl.as_vector((0, 1)), n)
    return nx + ny


data_1 = (
    "line_in_square_quad_-1",
    "square_quad",
    lambda x: x[0] + 0.35,
    -1,
    [3.0, -3.0],
    integrand_1,
)
data_2 = (
    "line_in_square_quad_1",
    "square_quad",
    lambda x: x[0] + 0.35,
    -1,
    [3.0, -3.0],
    integrand_1,
)
data_3 = (
    "line_in_square_quad_2",
    "square_quad",
    lambda x: x[0] + 0.35,
    -1,
    [3.0, -3.0],
    integrand_1,
)
data_4 = (
    "line_in_square_quad_3",
    "square_quad",
    lambda x: x[0] + 0.35,
    -1,
    [3.0, -3.0],
    integrand_1,
)


def integrand_2(n):
    nx = ufl.dot(ufl.as_vector((1, 0)), n)
    ny = ufl.dot(ufl.as_vector((0, 1)), n)
    return ufl.algebra.Abs(nx) + ufl.algebra.Abs(ny)


data_5 = (
    "square_in_square_quad_-1",
    "square_quad",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.35,
    -1,
    [3.2, 2.4],
    integrand_2,
)
data_6 = (
    "square_in_square_quad_1",
    "square_quad",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.35,
    1,
    [3.2, 2.4],
    integrand_2,
)
data_7 = (
    "square_in_square_quad_2",
    "square_quad",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.35,
    2,
    [3.2, 2.4],
    integrand_2,
)
data_8 = (
    "square_in_square_quad_3",
    "square_quad",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.35,
    3,
    [3.2, 2.4],
    integrand_2,
)

data_9 = (
    "square_in_square_tri_-1",
    "square_tri",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.325,
    -1,
    [3.2, 2.4],
    integrand_2,
)
data_10 = (
    "square_in_square_tri_-1",
    "square_tri",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.325,
    1,
    [3.2, 2.4],
    integrand_2,
)
data_11 = (
    "square_in_square_tri_-1",
    "square_tri",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.325,
    2,
    [3.2, 2.4],
    integrand_2,
)
data_12 = (
    "square_in_square_tri_-1",
    "square_tri",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 0.325,
    3,
    [3.2, 2.4],
    integrand_2,
)

testdata = [
    data_1,
    data_2,
    data_3,
    data_4,
    data_5,
    data_6,
    data_7,
    data_8,
    data_9,
    data_10,
    data_11,
    data_12,
]

parent_dir = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "data_name, mesh_name, levelset, discrete_levelset_degree, benchmark_values, integrand",
    testdata,
)
def test_one_sided_integral(
    data_name,
    mesh_name,
    levelset,
    discrete_levelset_degree,
    benchmark_values,
    integrand,
    plot=False,
):
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as fi:
        mesh = fi.read_mesh()

    if discrete_levelset_degree > 0:
        cg_element = element(
            "Lagrange", mesh.topology.cell_name(), discrete_levelset_degree
        )
        cg_space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test = dfx.fem.Function(cg_space)
        levelset_test.interpolate(levelset)
        detection_degree = discrete_levelset_degree
        levelset_test_cg = dfx.fem.Function(cg_space)
        levelset_test_cg.x.array[:] = levelset_test.x.array[:]
    else:
        detection_degree = 1
        cg_element = element("Lagrange", mesh.topology.cell_name(), detection_degree)
        cg_space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test_cg = dfx.fem.Function(cg_space)
        levelset_test_cg.interpolate(levelset)
        levelset_test = levelset

    cells_tags, facets_tags, _, d_from_inside, d_from_outside, _ = (
        compute_tags_measures(mesh, levelset_test, detection_degree, box_mode=True)
    )

    n = ufl.FacetNormal(mesh)
    test_int_mesh_in = integrand(n) * d_from_inside
    val_test_mesh_in = assemble_scalar(dfx.fem.form(test_int_mesh_in))

    test_int_mesh_out = integrand(n) * d_from_outside
    val_test_mesh_out = assemble_scalar(dfx.fem.form(test_int_mesh_out))

    if plot:
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(
            mesh, cells_tags, ax, expression_levelset=levelset, linewidth=0.1
        )
        plt.savefig(data_name + "_cells_tags.png", dpi=500, bbox_inches="tight")
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(
            mesh, facets_tags, ax, expression_levelset=levelset, linewidth=1.5
        )
        plt.savefig(data_name + "_facets_tags.png", dpi=500, bbox_inches="tight")

        print(val_test_mesh_in)
        print(val_test_mesh_out)

    assert np.isclose(val_test_mesh_in, benchmark_values[0], atol=1.0e-20)
    assert np.isclose(val_test_mesh_out, benchmark_values[1], atol=1.0e-20)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tags_plot.plot import plot_mesh_tags

    test_data = data_9
    test_one_sided_integral(
        test_data[0],
        test_data[1],
        test_data[2],
        test_data[3],
        test_data[4],
        test_data[5],
        plot=True,
    )
