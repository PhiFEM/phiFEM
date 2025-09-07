from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import pytest
from phifem.mesh_scripts import _tag_cells, _tag_facets
import os

"""
Data_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("circle_in_circle_-1", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, -1)
data_2 = ("circle_in_circle_1", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, 1)
data_3 = ("circle_in_circle_2", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, 2)
data_4 = ("circle_in_circle_3", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, 3)

data_5 = ("boundary_crossing_circle_-1", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, -1)
data_6 = ("boundary_crossing_circle_1", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, 1)
data_7 = ("boundary_crossing_circle_2", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, 2)
data_8 = ("boundary_crossing_circle_3", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, 3)

data_9  = ("circle_in_square_-1", "square_quad", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, -1)
data_10 = ("circle_in_square_1", "square_quad", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, 1)
data_11 = ("circle_in_square_2", "square_quad", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, 2)
data_12 = ("circle_in_square_3", "square_quad", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, 3)

data_13 = ("square_in_square_-1", "square", lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 1., -1)
data_14 = ("square_in_square_1", "square", lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 1., 1)
data_15 = ("square_in_square_2", "square", lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 1., 2)
data_16 = ("square_in_square_3", "square", lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) - 1., 3)

data_17 = ("ellipse_in_square_-1", "square_quad", lambda x: x[0]**2 + (0.3 * x[1] + 0.1)**2 - 0.65, -1)
data_18 = ("ellipse_in_square_1",  "square_quad", lambda x: x[0]**2 + (0.3 * x[1] + 0.1)**2 - 0.65, 1)
data_19 = ("ellipse_in_square_2",  "square_quad", lambda x: x[0]**2 + (0.3 * x[1] + 0.1)**2 - 0.65, 2)
data_20 = ("ellipse_in_square_3",  "square_quad", lambda x: x[0]**2 + (0.3 * x[1] + 0.1)**2 - 0.65, 3)


testdata = [data_1,  data_2,  data_3,  data_4,
            data_5,  data_6,  data_7,  data_8,
            data_9,  data_10, data_11, data_12,
            data_13, data_14, data_15, data_16,
            data_17, data_18, data_19, data_20]

parent_dir = os.path.dirname(__file__)

@pytest.mark.parametrize("data_name, mesh_name, levelset, discrete_levelset_degree", testdata)
def test_compute_meshtags(data_name, mesh_name, levelset, discrete_levelset_degree, save_as_benchmark=False):
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as fi:
        mesh = fi.read_mesh()
    
    if discrete_levelset_degree > 0:
        cg_element = element("Lagrange", mesh.topology.cell_name(), discrete_levelset_degree)
        cg_space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test = dfx.fem.Function(cg_space)
        levelset_test.interpolate(levelset)
        # Test computation of cells tags
        cells_tags = _tag_cells(mesh, levelset_test, discrete_levelset_degree)
    else:
        levelset_test = levelset
        # Test computation of cells tags
        cells_tags = _tag_cells(mesh, levelset_test, 1)

    # Test computation of facets tags when cells tags are provided
    facets_tags = _tag_facets(mesh, cells_tags)

    # To save benchmark
    if save_as_benchmark:
        cells_benchmark = np.vstack([cells_tags.indices, cells_tags.values])
        np.savetxt(os.path.join(parent_dir, "tests_data", data_name + "_cells_tags.csv"), cells_benchmark, delimiter=" ", newline="\n")

        facets_benchmark = np.vstack([facets_tags.indices, facets_tags.values])
        np.savetxt(os.path.join(parent_dir, "tests_data", data_name + "_facets_tags.csv"), facets_benchmark, delimiter=" ", newline="\n")

        # For visualization purpose only
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(mesh, cells_tags, ax, expression_levelset=levelset)
        plt.savefig(os.path.join(parent_dir, "tests_data", data_name + "_cells_tags.png"), dpi=500, bbox_inches="tight")
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(mesh, facets_tags, ax, linewidth=1.5)
        plt.savefig(os.path.join(parent_dir, "tests_data", data_name + "_facets_tags.png"), dpi=500, bbox_inches="tight")
    else:
        try:
            cells_benchmark = np.loadtxt(os.path.join(parent_dir, "tests_data", data_name + "_cells_tags.csv"), delimiter=" ")
        except FileNotFoundError:
            raise FileNotFoundError("{cells_benchmark_name} not found, have you generated the benchmark ?")
        try:
            facets_benchmark = np.loadtxt(os.path.join(parent_dir, "tests_data", data_name + "_facets_tags.csv"), delimiter=" ")
        except FileNotFoundError:
            raise FileNotFoundError("{facets_benchmark_name} not found, have you generated the benchmark ?")

    assert np.all(cells_tags.indices == cells_benchmark[0,:])
    assert np.all(cells_tags.values  == cells_benchmark[1,:])

    assert np.all(facets_tags.indices == facets_benchmark[0,:])
    assert np.all(facets_tags.values  == facets_benchmark[1,:])


if __name__=="__main__":
    from tags_plot.plot import plot_mesh_tags
    import matplotlib.pyplot as plt
    for test_data in testdata:
        test_compute_meshtags(test_data[0], test_data[1], test_data[2], test_data[3], save_as_benchmark=True)