from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import pytest
import os

from phiFEM.phifem.mesh_scripts import _tag_cells, _tag_facets
from phiFEM.phifem.continuous_functions import Levelset
from phiFEM.phifem.mesh_scripts import (_reshape_facets_map,
                                        compute_outward_normal)
from tests_data.utils import create_square, create_disk

parent_dir = os.path.dirname(__file__)

def rotation(angle, x):
    return (np.cos(angle)*x[0] + np.sin(angle)*x[1], -np.sin(angle)*x[0] + np.cos(angle)*x[1])

"""
Dara_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_11 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125 * np.ones_like(x[0, :])), -1)
data_12 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125 * np.ones_like(x[0, :])), 1)
data_13 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125 * np.ones_like(x[0, :])), 2)
data_14 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125 * np.ones_like(x[0, :])), 3)

def levelset_2(x):
    def fct(x):
        return np.sum(np.abs(rotation(np.pi/6. - np.pi/4., x)), axis=0)
    return fct(x) - np.sqrt(2.)/2.

data_21 = ("Square", "square", Levelset(levelset_2), -1)
data_22 = ("Square", "square", Levelset(levelset_2), 1)
data_23 = ("Square", "square", Levelset(levelset_2), 2)
data_24 = ("Square", "square", Levelset(levelset_2), 3)

testdata = [data_11, data_12, data_13, data_14,
            data_21, data_22, data_23, data_24]

@pytest.mark.parametrize("data_name, mesh_name, levelset, discrete_levelset_degree", testdata)
def test_outward_normal(data_name, mesh_name, levelset, discrete_levelset_degree, save_normal=False):
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    if not os.path.isfile(mesh_path):
        print(f"{mesh_path} not found, we create it.")
        if mesh_name=="disk":
            create_disk(mesh_path, 0.1)
        elif mesh_name=="square":
            create_square(mesh_path, 0.1)
    
    with XDMFFile(MPI.COMM_WORLD, os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf"), "r") as fi:
        mesh = fi.read_mesh()
    
    cdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1

    cells_tags  = _tag_cells(mesh, levelset, 1)
    facets_tags = _tag_facets(mesh, cells_tags)

    if discrete_levelset_degree > 0:
        cg_element = element("Lagrange", mesh.topology.cell_name(), discrete_levelset_degree)
        cg_space = dfx.fem.functionspace(mesh, cg_element)
        levelset_test = dfx.fem.Function(cg_space)
        levelset_test.interpolate(levelset)
    else:
        levelset_test = levelset

    w0 = compute_outward_normal(mesh, levelset_test)

    if save_normal:
        with XDMFFile(mesh.comm, "./normal.xdmf", "w") as of:
            of.write_mesh(mesh)
            of.write_function(w0)

    W0 = w0.function_space

    mesh.topology.create_connectivity(cdim, cdim)
    f2v_connect = mesh.topology.connectivity(fdim, 0)

    f2v_map = np.reshape(f2v_connect.array, (-1, 2))

    points = mesh.geometry.x

    f2c_connect = mesh.topology.connectivity(fdim, cdim)
    f2c_map = _reshape_facets_map(f2c_connect)
    mask = np.where(facets_tags.values == 4)
    f2c_map[mask]

    for facet in facets_tags.indices[mask]:
        neighbor_inside_cell = np.intersect1d(f2c_map[facet], cells_tags.indices[np.where(cells_tags.values == 2)])
        dof_0 = dfx.fem.locate_dofs_topological(W0.sub(0), cdim, neighbor_inside_cell)
        dof_1 = dfx.fem.locate_dofs_topological(W0.sub(1), cdim, neighbor_inside_cell)
        verts = f2v_map[facet]
        vec_facet = [points[verts][0][0] - points[verts][1][0], points[verts][0][1] - points[verts][1][1]]
        val_normal = [w0.sub(0).x.array[dof_0][0], w0.sub(1).x.array[dof_1][0]]
        inner_pdct = np.inner(vec_facet, val_normal)
        
        # Check that the gradient from the levelset is orthogonal to the boundary facet
        assert np.isclose(inner_pdct, 0.), f"inner_pdct = {inner_pdct}"

        coords = mesh.geometry.x
        cell_vertices = mesh.topology.connectivity(cdim, 0).links(neighbor_inside_cell[0])
        cell_midpoint = coords[cell_vertices].mean(axis=0)
        facet_vertices = mesh.topology.connectivity(fdim, 0).links(facet)
        facet_midpoint = coords[facet_vertices].mean(axis=0)

        vec_midpoints = facet_midpoint - cell_midpoint

        # Check that the gradient from the levelset is pointing outward of Omega_h
        assert np.greater(np.inner(val_normal, vec_midpoints[:-1]), 0.)

        # Check that the gradient is normalized
        norm = np.sqrt(np.inner(val_normal, val_normal))
        assert np.isclose(norm, 1.), f"||normal|| = {norm}"

if __name__=="__main__":
    test_outward_normal(*data_22, save_normal=True)