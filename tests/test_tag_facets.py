import os

import dolfinx as dfx
import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx.io import XDMFFile
from mpi4py import MPI

from phifem.mesh_scripts import _tag_facets

"""
Data_n° = ("Data name", "mesh name", levelset object, "facets benchmark name", "facets benchmark name")
"""


def gen_levelset(x0, a, x1, b, c):
    def generate_levelset(mode):
        def levelset(x):
            return (a * x[0] - x0) ** 2 + (b * x[1] - x1) ** 2 + c

        return levelset

    return generate_levelset


data_1 = ("circle_in_circle", "disk", gen_levelset(0.0, 1.0, 0.0, 1.0, -0.125))


data_2 = (
    "boundary_crossing_circle",
    "disk",
    gen_levelset(0.0, 1.0, -0.5, 1.0, -0.125),
)


data_3 = (
    "circle_in_square",
    "square_quad",
    gen_levelset(0.0, 1.0, 0.0, 1.0, -0.125),
)


def generate_levelset_4(mode):
    if mode.__name__ == "ufl":

        def mode_max(x, y):
            return ufl.conditional(x > y, x, y)
    elif mode.__name__ == "numpy":

        def mode_max(x, y):
            return np.maximum(x, y)

    def levelset(x):
        return mode_max(abs(x[0]), abs(x[1])) - 1.0

    return levelset


data_4 = (
    "square_in_square",
    "square_tri",
    generate_levelset_4,
)


data_5 = (
    "ellipse_in_square",
    "square_quad",
    gen_levelset(0.0, 1.0, 0.1, 0.3, -0.65),
)

data_6 = (
    "circle_near_boundary",
    "coarse_square",
    gen_levelset(0.5, 1.0, 0.5, 1.0, -0.2),
)


def generate_levelset_7(mode):
    def atan2(y, x):
        if mode.__name__ == "numpy":
            return mode.arctan2(y, x)
        elif mode.__name__ == "ufl":
            return mode.atan2(y, x)

    def levelset(x):
        val = (
            mode.sqrt(x[0] ** 2 + x[1] ** 2)
            * (abs(atan2(x[1], x[0])) * mode.sin(1.0 / abs(atan2(x[1], x[0]))))
            - 0.25
        )
        return val

    return levelset


data_7 = (
    "nasty_levelset",
    "square_tri",
    generate_levelset_7,
)
testdata = [data_1, data_2, data_3, data_4, data_5, data_6, data_7]

testdegrees = np.arange(1, 5).astype(np.int32)


def tst_space(mesh, degree):
    elment = element("Lagrange", mesh.topology.cell_name(), degree)
    return dfx.fem.functionspace(mesh, elment)


testdiscretize = [True, False]

testsingle_layer_cut = [True, False]

parent_dir = os.path.dirname(__file__)


@pytest.mark.parametrize("discretize", testdiscretize)
@pytest.mark.parametrize("detection_degree", testdegrees)
@pytest.mark.parametrize("single_layer_cut", testsingle_layer_cut)
@pytest.mark.parametrize("data_name, mesh_name, generate_levelset", testdata)
def test_compute_meshtags(
    data_name,
    mesh_name,
    generate_levelset,
    detection_degree,
    discretize,
    single_layer_cut,
    save_as_benchmark=False,
    plot=False,
):
    data_name = data_name + "_" + str(detection_degree)
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as fi:
        mesh = fi.read_mesh()

    space = tst_space(mesh, detection_degree)

    middle = "_"
    if discretize:
        middle += "discretize_"

    if single_layer_cut:
        middle += "single_layer_"

    benchmark_facets_name = data_name + middle

    if discretize:
        levelset = generate_levelset(np)
        cg_element = element("Lagrange", mesh.topology.cell_name(), detection_degree)
        space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test = dfx.fem.Function(space)
        levelset_test.interpolate(levelset)
    else:
        x_ufl = ufl.SpatialCoordinate(mesh)
        levelset_test = generate_levelset(ufl)(x_ufl)

    # Must pass points of dimension == dimension of the element on which we wish to evaluate the expression.
    reference_facet_points = np.array([[0.0], [1.0]], dtype=np.float64)
    levelset_expression = dfx.fem.Expression(levelset_test, reference_facet_points)

    facets_tags = _tag_facets(
        mesh, levelset_expression, single_layer_cut=single_layer_cut
    )
    all_facets = dfx.mesh.locate_entities(
        mesh, 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    bdy_facets = dfx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    print(len(all_facets))
    print(len(bdy_facets))
    print(len(all_facets) - len(bdy_facets))
    # To save benchmark
    if save_as_benchmark:
        facets_benchmark = np.vstack([facets_tags.indices, facets_tags.values])
        np.savetxt(
            os.path.join(
                parent_dir,
                "tests_data",
                "test_tag_facets",
                benchmark_facets_name + ".csv",
            ),
            facets_benchmark,
            delimiter=" ",
            newline="\n",
        )

    else:
        try:
            facets_benchmark = np.loadtxt(
                os.path.join(
                    parent_dir,
                    "tests_data",
                    "test_tag_facets",
                    benchmark_facets_name + ".csv",
                ),
                delimiter=" ",
            )
        except FileNotFoundError:
            print(
                "{facets_benchmark_name} not found, have you generated the benchmark ?"
            )

    if plot:
        print("plot ", benchmark_facets_name)
        save_tags(
            mesh, os.path.join(parent_dir, benchmark_facets_name + ".xdmf"), facets_tags
        )
        levelset = generate_levelset(np)
        cg_element = element("Lagrange", mesh.topology.cell_name(), detection_degree)
        space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test = dfx.fem.Function(space)
        levelset_test.interpolate(levelset)
        with XDMFFile(
            mesh.comm,
            os.path.join(parent_dir, benchmark_facets_name + "levelset" + ".xdmf"),
            "w",
        ) as of:
            of.write_mesh(mesh)
            of.write_function(levelset_test)

    assert np.all(facets_tags.indices == facets_benchmark[0, :])
    assert np.all(facets_tags.values == facets_benchmark[1, :])


if __name__ == "__main__":
    from utils_test import save_tags

    testdata_main = [testdata[5]]
    testdegrees_main = [testdegrees[0]]
    testdiscretize = [False, True]
    testsingle_layer_cut = [False, True]
    for test_data in testdata_main:
        print(f"{test_data[0]}, {test_data[1]}")
        for test_degree in testdegrees_main:
            for test_discretize in testdiscretize:
                for single_layer_cut in testsingle_layer_cut:
                    test_compute_meshtags(
                        *test_data,
                        test_degree,
                        test_discretize,
                        single_layer_cut,
                        save_as_benchmark=True,
                        plot=True,
                    )
