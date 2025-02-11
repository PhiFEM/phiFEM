from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import pytest
from phiFEM.phifem.compute_meshtags import tag_cells, tag_facets
from phiFEM.phifem.continuous_functions import Levelset
import os
from test_outward_normal import create_disk # type: ignore


"""
Data_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125), "celltags_1", "facettags_1")

testdata = [data_1]

parent_dir = os.path.dirname(__file__)

@pytest.mark.parametrize("data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name", testdata)
def test_compute_meshtags(data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name, save_as_benchmark=False):
    mesh_path = os.path.join(parent_dir, "tests_data", "disk" + ".xdmf")

    if not os.path.isfile(mesh_path):
        print(f"{mesh_path} not found, we create it.")
        create_disk(mesh_path, 0.1)
    
    with XDMFFile(MPI.COMM_WORLD, os.path.join(parent_dir, "tests_data", "disk.xdmf"), "r") as fi:
        mesh = fi.read_mesh()
    
    # Test computation of cells tags
    cells_tags = tag_cells(mesh, levelset, 1)
    # Test computation of facets tags when cells tags are provided
    facets_tags = tag_facets(mesh, cells_tags)

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

    levelset = Levelset(expression_levelset)

    k = 4
    CGElement = element("Lagrange", mesh.topology.cell_name(), k)
    V = dfx.fem.functionspace(mesh, CGElement)
    discrete_levelset = dfx.fem.Function(V)
    discrete_levelset.interpolate(levelset.expression)

    cells_tags = tag_cells(mesh,
                           discrete_levelset,
                           plot=False)

    facets_tags = tag_facets(mesh,
                             discrete_levelset,
                             cells_tags,
                             plot=True)

    test_compute_meshtags("0", "disk", levelset, "celltags_1", "facettags_1", save_as_benchmark=False)
