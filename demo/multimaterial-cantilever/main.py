import os

import dolfinx as dfx
import numpy as np
import petsc4py.PETSc as PETSc
import ufl
from basix.ufl import element, mixed_element
from data import (
    E_left,
    E_right,
    epsilon,
    exact_levelset,
    penalization_coefficient,
    sigma_left,
    sigma_right,
    stabilization_coefficient,
    traction,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI

from phifem.mesh_scripts import compute_tags_measures

"""
Create the output directory.
"""
parent_dir = os.path.dirname(__file__)
output_dir = os.path.join(parent_dir, "output")

if not os.path.isdir(output_dir):
    print(f"{output_dir} directory not found, we create it.")
    os.mkdir(os.path.join(parent_dir, output_dir))

mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [2.0, 1.0]], [50, 25])

"""
The phiFEM interface scheme uses extra variables in addition to the "primal" variables (the displacement on the left part and the displacement on the right part).
These additional variables represent the fluxes on the left and right and one auxiliary variable with no physical interpretation, used to "relax" the boundary condition penalization.
We then define the corresponding mixed element containing the element for the left and right displacement fields (primal), left and right fluxes tensors (flux) and the auxiliary variable.
We finaly define the mixed_space.
"""
gdim = mesh.geometry.dim
cell_name = mesh.topology.cell_name()
primal_degree = 1
primal_element = element("Lagrange", cell_name, primal_degree, shape=(gdim,))
flux_degree = 1
flux_element = element("Lagrange", cell_name, flux_degree, shape=(gdim, gdim))
auxiliary_degree = 0
auxiliary_element = element("DG", cell_name, auxiliary_degree, shape=(gdim,))

mixd_element = mixed_element(
    [primal_element, primal_element, flux_element, flux_element, auxiliary_element]
)

mixed_space = dfx.fem.functionspace(mesh, mixd_element)

"""
We create the FE space used to interpolate the analytical levelset.
Then, we use the interpolated levelset to compute the phiFEM tags and the phiFEM measures.
Here, we choose the detection_degree to be the same as the levelset degree.
We set box_mode to True since the levelset is here used to define an interface between two materials, we therefore want to keep the part of the mesh where the levelset is positive.
"""
levelset_degree = 1
levelset_element = element("Lagrange", cell_name, levelset_degree)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)
levelset = dfx.fem.Function(levelset_space)
levelset.interpolate(exact_levelset)

"""
We tag the facets supporting the different boundary conditions applied to the beam.
"""


def clamped_bdy(x):
    return x[0] < 0.001


def traction_bdy(x):
    right = x[0] > 1.99
    middle = np.logical_and(x[1] > 0.4, x[1] < 0.6)
    return np.logical_and(right, middle)


fdim = gdim - 1
clamped_bdy_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, clamped_bdy)
traction_bdy_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, traction_bdy)

clamped_tags = np.full_like(clamped_bdy_facets, 10)
traction_tags = np.full_like(traction_bdy_facets, 20)
constrained_facets = np.hstack([clamped_bdy_facets, traction_bdy_facets])
constrained_tags = np.hstack([clamped_tags, traction_tags])
sorted = np.argsort(constrained_facets)
sorted_facets = constrained_facets[sorted]
sorted_tags = constrained_tags[sorted]

bdy_facet_tags = dfx.mesh.meshtags(mesh, fdim, sorted_facets, sorted_tags)
tags_to_overwrite = {"facets": bdy_facet_tags}

"""
We call the phiFEM function to return the cells and facets tags as well as the phiFEM boundary measure.
Note that we had to overwrite the tags phiFEM applies to the boundary of the mesh with the tags on the traction facets.
We use the cells and facets tags to define our dx and dS measures.
Note that the ds_phifem measure, unlike standard dolfinx code, is not restricted to the boundary of the mesh but applies also to the boundaries of the union of the cut cells.
"""
cells_tags, facets_tags, _, ds_phifem, _ = compute_tags_measures(
    mesh, levelset, levelset_degree, box_mode=True, overwrite_tags=tags_to_overwrite
)
with XDMFFile(mesh.comm, os.path.join(output_dir, "cells_tags.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_meshtags(cells_tags, mesh.geometry)

all_facets = dfx.mesh.locate_entities(mesh, 1, lambda x: np.ones_like(x[0], dtype=bool))
wireframe = dfx.mesh.create_submesh(mesh, 1, all_facets)[0]
wf_cell_name = wireframe.topology.cell_name()
dg0_element = element("DG", wf_cell_name, 0)
wf_dg0_space = dfx.fem.functionspace(wireframe, dg0_element)

facets_tags_h = dfx.fem.Function(wf_dg0_space)
facets_tags_h.x.array[:] = facets_tags.values

with XDMFFile(wireframe.comm, os.path.join(output_dir, "facets_tags.xdmf"), "w") as of:
    of.write_mesh(wireframe)
    of.write_function(facets_tags_h)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

"""
We define the clamped Dirichlet boundary condition, note that the Dirichlet boundary condition only need to be applied to the left displacement, connected to the clamped boundary, this is why we use mixed_space.sub(0).
"""
left_space, map_left = mixed_space.sub(0).collapse()
clamped_bdy_dofs = dfx.fem.locate_dofs_topological(
    (mixed_space.sub(0), left_space), fdim, clamped_bdy_facets
)
dbc = dfx.fem.Function(left_space)
dirichlet_bc = dfx.fem.dirichletbc(dbc, clamped_bdy_dofs, mixed_space.sub(0))
bcs = [dirichlet_bc]

u_left, u_right, y_left, y_right, p = ufl.TrialFunctions(mixed_space)
v_left, v_right, z_left, z_right, q = ufl.TestFunctions(mixed_space)

n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)

boundary_left = ufl.inner(ufl.dot(y_left, n), v_left)
boundary_right = ufl.inner(ufl.dot(y_right, n), v_right)

stiffness_left = ufl.inner(sigma_left(u_left), epsilon(v_left))
stiffness_right = ufl.inner(sigma_right(u_right), epsilon(v_right))

coef_left = (E_left / (E_left + E_right)) ** 2
coef_right = (E_right / (E_left + E_right)) ** 2
penalization = penalization_coefficient * (
    ufl.inner(y_left + sigma_left(u_left), z_left + sigma_left(v_left)) * coef_right
    + ufl.inner(y_right + sigma_right(u_right), z_right + sigma_right(v_right))
    * coef_left
    + h ** (-2)
    * ufl.inner(
        ufl.dot(y_left, ufl.grad(levelset)) - ufl.dot(y_right, ufl.grad(levelset)),
        ufl.dot(z_left, ufl.grad(levelset)) - ufl.dot(z_right, ufl.grad(levelset)),
    )
    + h ** (-2)
    * ufl.inner(
        u_left - u_right + h ** (-1) * p * levelset,
        v_left - v_right + h ** (-1) * q * levelset,
    )
)

stabilization_cells_right = stabilization_coefficient * ufl.inner(
    ufl.div(y_left), ufl.div(z_left)
)

stabilization_cells_left = stabilization_coefficient * ufl.inner(
    ufl.div(y_right), ufl.div(z_right)
)

stabilization_facets_right = (
    stabilization_coefficient
    * ufl.avg(h)
    * ufl.inner(ufl.jump(sigma_right(u_right), n), ufl.jump(sigma_right(v_right), n))
)

stabilization_facets_left = (
    stabilization_coefficient
    * ufl.avg(h)
    * ufl.inner(ufl.jump(sigma_left(u_left), n), ufl.jump(sigma_left(v_left), n))
)

a = (
    stiffness_left * dx((1, 2))
    + stiffness_right * dx((2, 3))
    + penalization * dx(2)
    + stabilization_facets_left * dS(3)
    + stabilization_facets_right * dS(4)
    + stabilization_cells_right * dx(2)
    + stabilization_cells_left * dx(2)
    + boundary_left * ds_phifem(100)
    + boundary_right * ds_phifem(101)
)

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form, bcs=bcs)
A.assemble()

ksp = PETSc.KSP().create(mesh.comm)
ksp.setType("preonly")
solver = ksp.create(MPI.COMM_WORLD)
solver.setFromOptions()
solver.setOperators(A)

pc = solver.getPC()
pc.setType("lu")
"""
We use MUMPS to handle the nullspace during the LU solve. Note that we could have used an iterative solver instead.
"""
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

L = ufl.inner(traction, v_right) * ds_phifem(20)

linear_form = dfx.fem.form(L)
b = assemble_vector(linear_form)

# Apply the dirichlet bc to the RHS vector
dfx.fem.petsc.apply_lifting(b, [bilinear_form], bcs=[bcs])
for bc in bcs:
    bc.set(b.array_w)

"""
Solve
"""
solution_wh = dfx.fem.Function(mixed_space)

# Monitor PETSc solve time
viewer = PETSc.Viewer().createASCII(os.path.join(output_dir, "petsc_log.txt"))
PETSc.Log.begin()
ksp.solve(b, solution_wh.x.petsc_vec)
PETSc.Log.view(viewer)
ksp.destroy()

solution_left, solution_right = solution_wh.split()[:2]
displacement_left = solution_left.collapse()
displacement_right = solution_right.collapse()
displacement_left.name = "displacement_left"
displacement_right.name = "displacement_right"

mesh.topology.create_connectivity(2, 2)
right_space, map_right = mixed_space.sub(1).collapse()
map_left = np.asarray(map_left)
map_right = np.asarray(map_right)
displacement = dfx.fem.Function(left_space)

cut_dofs_left = dfx.fem.locate_dofs_topological(
    (mixed_space.sub(0), left_space), cells_tags.dim, cells_tags.find(2)
)[1]
cut_dofs_right = dfx.fem.locate_dofs_topological(
    (mixed_space.sub(1), right_space), cells_tags.dim, cells_tags.find(2)
)[1]
left_dofs = dfx.fem.locate_dofs_topological(
    (mixed_space.sub(0), left_space), cells_tags.dim, cells_tags.find(1)
)[1]
left_dofs = np.setdiff1d(left_dofs, cut_dofs_left)
right_dofs = dfx.fem.locate_dofs_topological(
    (mixed_space.sub(1), right_space), cells_tags.dim, cells_tags.find(3)
)[1]
right_dofs = np.setdiff1d(right_dofs, cut_dofs_right)
displacement.x.array[left_dofs] = solution_wh.x.array[map_left[left_dofs]]
displacement.x.array[right_dofs] = solution_wh.x.array[map_right[right_dofs]]
displacement.x.array[cut_dofs_left] = 0.5 * (
    solution_wh.x.array[map_left[cut_dofs_left]]
    + solution_wh.x.array[map_right[cut_dofs_right]]
)
displacement.name = "displacement"

with XDMFFile(mesh.comm, os.path.join(output_dir, "results.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(displacement_left)
    of.write_function(displacement_right)
    of.write_function(displacement)
