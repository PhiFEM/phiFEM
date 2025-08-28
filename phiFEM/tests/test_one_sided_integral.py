from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_scalar
from mpi4py import MPI
import numpy as np
import os
import pytest
import ufl

from phiFEM.phifem.mesh_scripts import compute_tags_measures
from phiFEM.tests.tests_data.utils import create_disk
from tags_plot.plot import plot_mesh_tags
import matplotlib.pyplot as plt

"""
Data_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, "celltags_1", "facettags_1", -1)
data_2 = ("Circle radius 1", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, "celltags_1", "facettags_1", 1)
data_3 = ("Circle radius 1", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, "celltags_1", "facettags_1", 2)
data_4 = ("Circle radius 1", "disk", lambda x: x[0, :]**2 + x[1, :]**2 - 0.125, "celltags_1", "facettags_1", 3)
# data_5 = ("Boundary touching", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, "celltags_2", "facettags_2", -1)
# data_6 = ("Boundary touching", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, "celltags_2", "facettags_2", 1)
# data_7 = ("Boundary touching", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, "celltags_2", "facettags_2", 2)
# data_8 = ("Boundary touching", "disk", lambda x: x[0]**2 + (x[1] - 0.5)**2 - 0.125, "celltags_2", "facettags_2", 3)

testdata = [data_1, data_2, data_3, data_4]
            # data_5, data_6, data_7, data_8]


def integrand(n):
    return ufl.dot(1.e17 * ufl.as_vector((1, 0)), n)

parent_dir = os.path.dirname(__file__)

@pytest.mark.parametrize("data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name, discrete_levelset_degree", testdata)
def test_one_sided_integral(data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name, discrete_levelset_degree, save_as_benchmark=False, plot=False):
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

    cells_tags, facets_tags, _, d_from_inside, d_from_outside = compute_tags_measures(mesh, levelset_test, detection_degree, box_mode=True)

    cells_tags_in, _, submesh_in, _, _ = compute_tags_measures(mesh, levelset_test, detection_degree, box_mode=False)

    levelset_test_out_cg = dfx.fem.Function(levelset_test_cg.function_space)
    levelset_test_out_cg.x.array[:] = -levelset_test_cg.x.array[:]
    cells_tags_out, _, submesh_out, _, _ = compute_tags_measures(mesh, levelset_test_out_cg, detection_degree, box_mode=False)

    if plot:
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(mesh, cells_tags, ax, expression_levelset=levelset)
        plt.savefig("one_sided_cells_mesh.png", dpi=500, bbox_inches="tight")
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(mesh, facets_tags, ax, expression_levelset=levelset)
        plt.savefig("one_sided_facets_mesh.png", dpi=500, bbox_inches="tight")
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(submesh_in, cells_tags_in, ax, expression_levelset=levelset)
        plt.savefig("one_sided_submesh_in.png", dpi=500, bbox_inches="tight")
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(submesh_out, cells_tags_out, ax, expression_levelset=levelset)
        plt.savefig("one_sided_submesh_out.png", dpi=500, bbox_inches="tight")

    n = ufl.FacetNormal(mesh)
    test_int_mesh_in = integrand(n) * d_from_inside
    val_test_mesh_in = assemble_scalar(dfx.fem.form(test_int_mesh_in))

    test_int_mesh_out = integrand(n) * d_from_outside
    val_test_mesh_out = assemble_scalar(dfx.fem.form(test_int_mesh_out))


    n = ufl.FacetNormal(submesh_in)
    test_int_submesh_in = integrand(n) * ufl.ds
    val_test_submesh_in = assemble_scalar(dfx.fem.form(test_int_submesh_in))
    
    assert val_test_mesh_in == val_test_submesh_in

    # Mark corresponding facets (i.e. where levelset < 0)
    fdim = submesh_out.topology.dim - 1
    facets = dfx.mesh.locate_entities_boundary(submesh_out, fdim, lambda x: levelset(x) < 0.)

    sorted_indices = np.argsort(facets)
    facets_markers = np.ones_like(facets).astype(np.int32)
    facets_tags_out = dfx.mesh.meshtags(submesh_out,
                                        fdim,
                                        facets[sorted_indices],
                                        facets_markers[sorted_indices])

    ds = ufl.Measure("ds", domain=submesh_out, subdomain_data=facets_tags_out)

    n = ufl.FacetNormal(submesh_out)
    test_int_submesh_out = integrand(n) * ds(1)
    val_test_submesh_out = assemble_scalar(dfx.fem.form(test_int_submesh_out))
    
    assert val_test_mesh_out == val_test_submesh_out

if __name__=="__main__":
    test_data = data_1
    test_one_sided_integral(data_1[0], data_1[1], data_1[2], data_1[3], data_1[4], data_1[5])