import os

import dolfinx as dfx
import numpy as np
import pytest
from basix.ufl import element
from dolfinx.io import XDMFFile
from mpi4py import MPI

from phifem.mesh_scripts import _tag_cells, _tag_facets

"""
Data_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("circle_in_circle", "disk", lambda x: x[0, :] ** 2 + x[1, :] ** 2 - 0.125)

data_2 = (
    "boundary_crossing_circle",
    "disk",
    lambda x: x[0] ** 2 + (x[1] - 0.5) ** 2 - 0.125,
)

data_3 = (
    "circle_in_square",
    "square_quad",
    lambda x: x[0, :] ** 2 + x[1, :] ** 2 - 0.125,
)

data_4 = (
    "square_in_square",
    "square_tri",
    lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 1.0,
)

data_5 = (
    "ellipse_in_square",
    "square_quad",
    lambda x: x[0] ** 2 + (0.3 * x[1] + 0.1) ** 2 - 0.65,
)

data_6 = (
    "circle_near_boundary",
    "coarse_square",
    lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.2,
)

data_7 = (
    "nasty_levelset",
    "square_tri",
    lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    * (np.abs(np.arctan2(x[1], x[0])) * np.sin(1.0 / np.abs(np.arctan2(x[1], x[0]))))
    - 0.25,
)
testdata = [data_1, data_2, data_3, data_4, data_5, data_6, data_7]

testdegrees = [1, 2, 3]

parent_dir = os.path.dirname(__file__)


@pytest.mark.parametrize("discrete_levelset_degree", testdegrees)
@pytest.mark.parametrize("data_name, mesh_name, levelset", testdata)
def test_compute_meshtags(
    data_name,
    mesh_name,
    levelset,
    discrete_levelset_degree,
    save_as_benchmark=False,
    plot=False,
):
    data_name = data_name + "_" + str(discrete_levelset_degree)
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as fi:
        mesh = fi.read_mesh()

    cg_element = element(
        "Lagrange", mesh.topology.cell_name(), discrete_levelset_degree
    )
    cg_space = dfx.fem.functionspace(mesh, cg_element)
    levelset_test = dfx.fem.Function(cg_space)
    levelset_test.interpolate(levelset)
    # Test computation of cells tags
    cells_tags = _tag_cells(mesh, levelset_test, discrete_levelset_degree)

    # Test computation of facets tags when cells tags are provided
    facets_tags = _tag_facets(mesh, cells_tags, levelset_test, discrete_levelset_degree)

    # To save benchmark
    if save_as_benchmark:
        cells_benchmark = np.vstack([cells_tags.indices, cells_tags.values])
        np.savetxt(
            os.path.join(parent_dir, "tests_data", data_name + "_cells_tags.csv"),
            cells_benchmark,
            delimiter=" ",
            newline="\n",
        )

        facets_benchmark = np.vstack([facets_tags.indices, facets_tags.values])
        np.savetxt(
            os.path.join(parent_dir, "tests_data", data_name + "_facets_tags.csv"),
            facets_benchmark,
            delimiter=" ",
            newline="\n",
        )
    else:
        try:
            cells_benchmark = np.loadtxt(
                os.path.join(parent_dir, "tests_data", data_name + "_cells_tags.csv"),
                delimiter=" ",
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "{cells_benchmark_name} not found, have you generated the benchmark ?"
            )
        try:
            facets_benchmark = np.loadtxt(
                os.path.join(parent_dir, "tests_data", data_name + "_facets_tags.csv"),
                delimiter=" ",
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "{facets_benchmark_name} not found, have you generated the benchmark ?"
            )

    if plot:
        # For visualization purpose only
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(
            mesh,
            cells_tags,
            ax,
            expression_levelset=levelset,
            display_indices=False,
            linewidth=0.1,
        )
        plt.savefig(
            os.path.join(parent_dir, "tests_data", data_name + "_cells_tags.png"),
            dpi=500,
            bbox_inches="tight",
        )
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(mesh, facets_tags, ax, linewidth=0.5, display_indices=False)
        plt.savefig(
            os.path.join(parent_dir, "tests_data", data_name + "_facets_tags.png"),
            dpi=500,
            bbox_inches="tight",
        )

    assert np.all(cells_tags.indices == cells_benchmark[0, :])
    assert np.all(cells_tags.values == cells_benchmark[1, :])

    assert np.all(facets_tags.indices == facets_benchmark[0, :])
    assert np.all(facets_tags.values == facets_benchmark[1, :])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from meshtagsplot import plot_mesh_tags

    testdata_main = testdata
    testdegrees_main = testdegrees
    for test_degree in testdegrees_main:
        for test_data in testdata_main:
            test_compute_meshtags(
                *test_data, test_degree, save_as_benchmark=True, plot=True
            )
