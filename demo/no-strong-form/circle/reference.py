import argparse
import os
import sys

import dolfinx as dfx
import numpy as np
import petsc4py.PETSc as PETSc
import ufl
from basix.ufl import element
from data import (
    MESH_SIZE,
    dirichlet_data,
    gen_mesh,
    source_term_scalar,
    source_term_vector,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile

sys.path.append("../")

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run FEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

output_dir = os.path.join(parent_dir, "fem_output")

mesh, geoModel = gen_mesh(MESH_SIZE, curved=True)
# For visualization only
linear_mesh = gen_mesh(MESH_SIZE, curved=False)[0]

tdim = mesh.topology.dim
fdim = tdim - 1
cell_name = mesh.topology.cell_name()
fe_element = element("Lagrange", cell_name, 1)
vector_element = element("Lagrange", cell_name, 1, shape=(tdim,))
fe_space = dfx.fem.functionspace(mesh, fe_element)
vector_space = dfx.fem.functionspace(mesh, vector_element)

f_scalar_h = dfx.fem.Function(fe_space)
f_scalar_h.interpolate(source_term_scalar)

f_vector_h = dfx.fem.Function(vector_space)
f_vector_h.interpolate(source_term_vector)

gh = dfx.fem.Function(fe_space)
gh.interpolate(dirichlet_data)

dx = ufl.Measure("dx", domain=mesh)

bcs = []
boundary_facets = dfx.mesh.locate_entities_boundary(
    mesh, tdim - 1, lambda x: np.ones_like(x[0]).astype(bool)
)
dofs_D = dfx.fem.locate_dofs_topological(fe_space, tdim - 1, boundary_facets)
bc = dfx.fem.dirichletbc(gh, dofs_D)
bcs = [bc]

u = ufl.TrialFunction(fe_space)
v = ufl.TestFunction(fe_space)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L = ufl.inner(f_scalar_h, v) * dx + ufl.inner(f_vector_h, ufl.grad(v)) * dx

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form, bcs=bcs)
A.assemble()
linear_form = dfx.fem.form(L)
b = assemble_vector(linear_form)
dfx.fem.apply_lifting(b, [bilinear_form], [bcs])
dfx.fem.set_bc(b, bcs)
b.assemble()

# PETSc solver
solver = PETSc.KSP().create(mesh.comm)
solver.setType("preonly")
solver.setOperators(A)
pc = solver.getPC()
pc.setType("cholesky")

solution_uh = dfx.fem.Function(fe_space)
solver.solve(b, solution_uh.x.petsc_vec)
solver.destroy()

linear_mesh_space = dfx.fem.functionspace(linear_mesh, fe_element)
cdim = linear_mesh.topology.dim
num_linear_cells = linear_mesh.topology.index_map(cdim).size_global
linear_cells = np.arange(num_linear_cells)
nmm_fe2linear = dfx.fem.create_interpolation_data(
    linear_mesh_space, fe_space, linear_cells, padding=1.0e-14
)

linear_mesh_sol = dfx.fem.Function(linear_mesh_space)
linear_mesh_sol.interpolate_nonmatching(solution_uh, linear_cells, nmm_fe2linear)
with XDMFFile(linear_mesh.comm, os.path.join(output_dir, "solution.xdmf"), "w") as of:
    of.write_mesh(linear_mesh)
    of.write_function(linear_mesh_sol)
