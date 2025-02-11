from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
from lxml import etree
from mpi4py import MPI
import meshio # type: ignore
import numpy as np
import pygmsh as pg # type: ignore
import pytest
import os

from phiFEM.phifem.compute_meshtags import tag_cells, tag_facets
from phiFEM.phifem.continuous_functions import Levelset
from phiFEM.phifem.mesh_scripts import (reshape_facets_map,
                                        compute_outward_normal)

parent_dir = os.path.dirname(__file__)

def create_square(mesh_path, lcar):
    mesh_corners = np.array([[-1.5, -1.5],
                             [1.5, 1.5]])
    nx = int(np.abs(mesh_corners[1, 0] - mesh_corners[0, 0]) * np.sqrt(2.) / lcar)
    ny = int(np.abs(mesh_corners[1, 1] - mesh_corners[0, 1]) * np.sqrt(2.) / lcar)
    mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, mesh_corners, [nx, ny])

    with XDMFFile(mesh.comm, os.path.join("tests_data", "square.xdmf"), "w") as of:
        of.write_mesh(mesh)

def create_disk(mesh_path, lcar):
    with pg.geo.Geometry() as geom:
        # Points
        p1 = geom.add_point([0.,   0., 0.], lcar)
        p2 = geom.add_point([0.5,  0., 0.], lcar)
        p3 = geom.add_point([-0.5, 0., 0.], lcar)

        # Lines
        c1 = geom.add_circle_arc(p2, p1, p3)
        c2 = geom.add_circle_arc(p3, p1, p2)

        # Suface
        lloop = geom.add_curve_loop([c1, c2])
        surf = geom.add_plane_surface(lloop)

        mesh = geom.generate_mesh(dim=2, algorithm=6)

    mesh.points = mesh.points[:, :2]

    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            triangle_cells = [("triangle", cell_block.data)]

    meshio.write_points_cells(mesh_path, mesh.points, triangle_cells)

    # meshio and dolfinx use incompatible Grid names ("Grid" for meshio and "mesh" for dolfinx)
    # the lines below change the Grid name from "Grid" to "mesh" to ensure the compatibility between meshio and dolfinx.
    tree = etree.parse(mesh_path)
    root = tree.getroot()

    for grid in root.findall(".//Grid"):
        grid.set("Name", "mesh")
    
    tree.write(mesh_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def rotation(angle, x):
    return (np.cos(angle)*x[0] + np.sin(angle)*x[1], -np.sin(angle)*x[0] + np.cos(angle)*x[1])

"""
Dara_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", Levelset(lambda x: x[0, :]**2 + x[1, :]**2 - 0.125 * np.ones_like(x[0, :])))

def levelset_2(x):
    def fct(x):
        return np.sum(np.abs(rotation(np.pi/6. - np.pi/4., x)), axis=0)
    return fct(x) - np.sqrt(2.)/2.

data_2 = ("Square", "square", Levelset(levelset_2))

testdata = [data_1, data_2]

@pytest.mark.parametrize("data_name, mesh_name, levelset", testdata)
def test_outward_normal(data_name, mesh_name, levelset, save_normal=False):
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

    cells_tags  = tag_cells(mesh, levelset, 1)
    facets_tags = tag_facets(mesh, cells_tags)
    w0 = compute_outward_normal(mesh, levelset)

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
    f2c_map = reshape_facets_map(f2c_connect)
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
    test_outward_normal(*data_2, save_normal=True)