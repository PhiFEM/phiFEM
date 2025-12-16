import os

import dolfinx as dfx
import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx.fem import assemble_scalar
from dolfinx.io import XDMFFile
from mpi4py import MPI

from src.phifem.mesh_scripts import compute_tags_measures


def generate_levelset_1(mode):
    def levelset_1(x):
        return x[0] + 0.35

    return levelset_1


def integrand_1(n):
    nx = ufl.dot(ufl.as_vector((1, 0)), n)
    ny = ufl.dot(ufl.as_vector((0, 1)), n)
    return nx + ny


data_1 = (
    "line_in_square_quad",
    "square_quad",
    generate_levelset_1,
    [3.0, -3.0],
    integrand_1,
)


def generate_levelset_2(mode):
    if mode.__name__ == "ufl":

        def mode_max(x, y):
            return ufl.conditional(x > y, x, y)
    elif mode.__name__ == "numpy":

        def mode_max(x, y):
            return np.maximum(x, y)

    def levelset_2(x):
        return mode_max(abs(x[0]), abs(x[1])) - 0.35

    return levelset_2


def integrand_2(n):
    nx = ufl.dot(ufl.as_vector((1, 0)), n)
    ny = ufl.dot(ufl.as_vector((0, 1)), n)
    return abs(nx) + abs(ny)


data_2 = (
    "square_in_square_quad",
    "square_quad",
    generate_levelset_2,
    [3.2, 2.4],
    integrand_2,
)


def generate_levelset_3(mode):
    if mode.__name__ == "ufl":

        def mode_max(x, y):
            return ufl.conditional(x > y, x, y)
    elif mode.__name__ == "numpy":

        def mode_max(x, y):
            return np.maximum(x, y)

    def levelset_3(x):
        return mode_max(abs(x[0]), abs(x[1])) - 0.325

    return levelset_3


data_3 = (
    "square_in_square_tri",
    "square_tri",
    generate_levelset_3,
    [3.2, 2.4],
    integrand_2,
)

testdata = [
    data_1,
    data_2,
    data_3,
]

testdegrees = [1, 2, 3]
testdiscretize = [True, False]

parent_dir = os.path.dirname(__file__)


@pytest.mark.parametrize("discretize", testdiscretize)
@pytest.mark.parametrize("detection_degree", testdegrees)
@pytest.mark.parametrize(
    "data_name, mesh_name, generate_levelset, benchmark_values, integrand",
    testdata,
)
def test_one_sided_integral(
    data_name,
    mesh_name,
    generate_levelset,
    benchmark_values,
    integrand,
    detection_degree,
    discretize,
    plot=False,
):
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as fi:
        mesh = fi.read_mesh()

    if discretize:
        levelset = generate_levelset(np)
        cg_element = element("CG", mesh.topology.cell_name(), detection_degree)
        cg_space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test = dfx.fem.Function(cg_space)
        levelset_test.interpolate(levelset)
    else:
        x_ufl = ufl.SpatialCoordinate(mesh)
        levelset_test = generate_levelset(ufl)(x_ufl)

    cells_tags, facets_tags, _, d_from_inside, d_from_outside, _ = (
        compute_tags_measures(mesh, levelset_test, detection_degree, box_mode=True)
    )

    n = ufl.FacetNormal(mesh)
    test_int_mesh_in = integrand(n) * d_from_inside
    val_test_mesh_in = assemble_scalar(dfx.fem.form(test_int_mesh_in))

    test_int_mesh_out = integrand(n) * d_from_outside
    val_test_mesh_out = assemble_scalar(dfx.fem.form(test_int_mesh_out))

    if plot:
        levelset = generate_levelset(np)
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
    from meshtagsplot import plot_mesh_tags

    test_data = data_1
    test_degree = 1
    test_discretize = True
    test_one_sided_integral(
        *test_data,
        test_degree,
        test_discretize,
        plot=True,
    )
