from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import pytest
from phiFEM.phifem.mesh_scripts import _tag_cells, _tag_facets
from phiFEM.phifem.continuous_functions import Levelset
import os
from test_outward_normal import create_disk # type: ignore


"""
Data_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125), "celltags_1", "facettags_1", -1)
data_2 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125), "celltags_1", "facettags_1", 1)
data_3 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125), "celltags_1", "facettags_1", 2)
data_4 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125), "celltags_1", "facettags_1", 3)

testdata = [data_1, data_2, data_3, data_4]

parent_dir = os.path.dirname(__file__)

@pytest.mark.parametrize("data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name, discrete_levelset_degree", testdata)
def test_compute_meshtags(data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name, discrete_levelset_degree, save_as_benchmark=False):
    mesh_path = os.path.join(parent_dir, "tests_data", "disk" + ".xdmf")

    if not os.path.isfile(mesh_path):
        print(f"{mesh_path} not found, we create it.")
        create_disk(mesh_path, 0.1)
    
    with XDMFFile(MPI.COMM_WORLD, os.path.join(parent_dir, "tests_data", "disk.xdmf"), "r") as fi:
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
        np.savetxt(os.path.join(parent_dir, "tests_data", "celltags_1.csv"), cells_benchmark, delimiter=" ", newline="\n")

        facets_benchmark = np.vstack([facets_tags.indices, facets_tags.values])
        np.savetxt(os.path.join(parent_dir, "tests_data", "facettags_1.csv"), facets_benchmark, delimiter=" ", newline="\n")
    else:
        try:
            cells_benchmark = np.loadtxt(os.path.join(parent_dir, "tests_data", cells_benchmark_name + ".csv"), delimiter=" ")
        except FileNotFoundError:
            raise FileNotFoundError("{cells_benchmark_name} not found, have you generated the benchmark ?")
        try:
            facets_benchmark = np.loadtxt(os.path.join(parent_dir, "tests_data", facets_benchmark_name + ".csv"), delimiter=" ")
        except FileNotFoundError:
            raise FileNotFoundError("{facets_benchmark_name} not found, have you generated the benchmark ?")

    assert np.all(cells_tags.indices == cells_benchmark[0,:])
    assert np.all(cells_tags.values  == cells_benchmark[1,:])

    assert np.all(facets_tags.indices == facets_benchmark[0,:])
    assert np.all(facets_tags.values  == facets_benchmark[1,:])


if __name__=="__main__":
    mesh_path = os.path.join(parent_dir, "tests_data", "disk" + ".xdmf")

    if not os.path.isfile(mesh_path):
        print(f"{mesh_path} not found, we create it.")
        create_disk(mesh_path, 0.1)
    
    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as fi:
        mesh = fi.read_mesh()
    
    tilt_angle = np.pi/3.
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

    # def expression_levelset(x):
    #     def fct(x):
    #         return np.sum(np.abs(rotation(-tilt_angle + np.pi/4., x)), axis=0)
    #     return fct(x) - np.sqrt(2.)/2.
    
    def expression_levelset(x):
        return x[0, :]**2 + x[1, :]**2 - 0.125


    k = 4
    CGElement = element("Lagrange", mesh.topology.cell_name(), k)
    V = dfx.fem.functionspace(mesh, CGElement)
    discrete_levelset = dfx.fem.Function(V)
    discrete_levelset.interpolate(expression_levelset)

    cells_tags = _tag_cells(mesh,
                            discrete_levelset,
                            2)

    facets_tags = _tag_facets(mesh,
                              cells_tags)

    test_compute_meshtags("0", "disk", expression_levelset, "celltags_1", "facettags_1", 2, save_as_benchmark=False)