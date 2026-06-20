import argparse
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
    levelset,
    sigma_left,
    sigma_right,
    traction
)
from dolfinx.fem import assemble_scalar
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

mesh = dfx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[0., 0.], [2., 1.]], [20, 10]
)

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
cell_name = mesh.topology.cell_name()
levelset_degree = 1
levelset_element = element("Lagrange", cell_name, levelset_degree)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)
levelset_h = dfx.fem.Function(levelset_space)
levelset_h.interpolate(levelset)

"""
We tag the facets supporting the different boundary conditions applied to the beam.
"""

def clamped_bdy(x):
    return x[0] < 0.1

def traction_bdy(x):
    right = x[0] > 0.9
    middle = np.logical_and(x[1] > 0.4, x[1] < 0.6)
    return np.logical_and(right, middle)

fdim = gdim - 1
clamped_bdy_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, clamped_bdy)
traction_bdy_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, traction_bdy)

clamped_tags = np.full_like(clamped_bdy_facets, 10)
traction_tags = np.full_like(traction_bdy_facets, 20)
constrained_facets = np.vstack([clamped_bdy_facets, traction_bdy_facets])
constrained_tags = np.vstack([clamped_tags, traction_tags])
sorted_facets = np.argsort(constrained_facets)
sorted_tags = constrained_tags[sorted_facets]

bdy_meshtags = dfx.mesh.meshtags(mesh, fdim, sorted_facets, sorted_tags)

"""
We call the phiFEM function to return the cells and facets tags as well as the phiFEM boundary measure.
Note that we had to overwrite the tags phiFEM applies to the boundary of the mesh with the tags on the traction facets.
We use the cells and facets tags to define our dx and dS measures.
Note that the ds_phifem measure, unlike standard dolfinx code, is not restricted to the boundary of the mesh but applies also to the boundaries of the union of the cut cells.
"""
cells_tags, facets_tags, _, ds_phifem, _ = compute_tags_measures(
    mesh, levelset_h, levelset_degree, box_mode=True, overwrite_tags=bdy_meshtags
)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

"""
We define the clamped Dirichlet boundary condition, note that the Dirichlet boundary condition only need to be applied to the left displacement, connected to the clamped boundary, this is why we use mixed_space.sub(0).
"""
clamped_bdy_dofs = dfx.fem.locate_dofs_topological(
    mixed_space.sub(0), fdim, clamped_bdy_facets
)
dirichlet_bc = dfx.fem.dirichletbc(dfx.default_scalar_type(0), clamped_bdy_dofs, mixed_space.sub(0))
bcs = [dirichlet_bc]

u_left, u_right, y_left, y_right, p = ufl.TrialFunctions(mixed_space)
v_left, v_right, z_left, z_right, q = ufl.TestFunctions(mixed_space)



n = ufl.FacetNormal(mesh)
h_T = ufl.CellDiameter(mesh)

boundary_in = ufl.inner(ufl.dot(y_left, n), v_left)
boundary_out = ufl.inner(ufl.dot(y_right, n), v_right)

stiffness_in = ufl.inner(sigma_left(u_left), epsilon(v_left))
stiffness_out = ufl.inner(sigma_right(u_right), epsilon(v_right))

coef_in = (E_left / (E_left + E_right)) ** 2
coef_out = (E_right / (E_left + E_right)) ** 2
penalization = penalization_coefficient * (
    ufl.inner(y_left + sigma_left(u_left), z_left + sigma_left(v_left)) * coef_out
    + ufl.inner(y_right + sigma_right(u_right), z_right + sigma_right(v_right)) * coef_in
    + h_T ** (-2)
    * ufl.inner(
        ufl.dot(y_left, ufl.grad(phi_h)) - ufl.dot(y_right, ufl.grad(phi_h)),
        ufl.dot(z_left, ufl.grad(phi_h)) - ufl.dot(z_right, ufl.grad(phi_h)),
    )
    + h_T ** (-2)
    * ufl.inner(
        u_left - u_right + h_T ** (-1) * p * phi_h,
        v_left - v_right + h_T ** (-1) * q * phi_h,
    )
)

stabilization_facets_in = (
    stabilization_coefficient
    * ufl.avg(h_T)
    * ufl.inner(ufl.jump(sigma_left(u_left), n), ufl.jump(sigma_left(v_left), n))
)

stabilization_cells_in = (
    stabilization_coefficient * h_T**2 * ufl.inner(ufl.div(y_left), ufl.div(z_left))
)

stabilization_cells_out = (
    stabilization_coefficient * h_T**2 * ufl.inner(ufl.div(y_right), ufl.div(z_right))
)

stabilization_facets_out = (
    stabilization_coefficient
    * ufl.avg(h_T)
    * ufl.inner(ufl.jump(sigma_right(u_right), n), ufl.jump(sigma_right(v_right), n))
)

a = (
    stiffness_in * dx((1, 2))
    + stiffness_out * dx((2, 3))
    + penalization * dx(2)
    + stabilization_facets_in * dS(3)
    + stabilization_facets_out * dS(4)
    + stabilization_cells_in * dx(2)
    + stabilization_cells_out * dx(2)
    + boundary_in * d_bdry(100)
    + boundary_out * d_bdry(101)
)

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form, bcs=bcs)
A.assemble()

ksp = PETSc.KSP().create(mesh.comm)
ksp.setType("preonly")
solver = ksp.create(MPI.COMM_WORLD)
solver.setFromOptions()
solver.setOperators(A)

# Configure MUMPS to handle nullspace
pc = solver.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

stabilization_rhs_in = (
    stabilization_coefficient * h_T**2 * (ufl.inner(f, ufl.div(z_left)))
)
stabilization_rhs_out = (
    stabilization_coefficient * h_T**2 * (ufl.inner(f, ufl.div(z_right)))
)
rhs_in = ufl.inner(f, v_left)
rhs_out = ufl.inner(f, v_right)

L = (
    rhs_in * dx((1, 2))
    + rhs_out * dx((2, 3))
    + stabilization_rhs_in * dx(2)
    + stabilization_rhs_out * dx(2)
)

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

solution_uh_in, solution_uh_out, _, _, _ = solution_wh.split()
save_function(solution_uh_in.collapse(), f"solution_in_{str(i).zfill(2)}")
save_function(solution_uh_out.collapse(), f"solution_out_{str(i).zfill(2)}")

# Combine the in and out solutions
solution_h = dfx.fem.Function(mixed_space)
solution_uh, _, _, _, _ = solution_h.split()
solution_uh = solution_uh.collapse()

mesh.topology.create_connectivity(gdim, gdim)
dofs_to_remove_in = dfx.fem.locate_dofs_topological(
    mixed_space.sub(0), gdim, cells_tags.find(3)
)
dofs_cut_in = dfx.fem.locate_dofs_topological(
    mixed_space.sub(0), gdim, cells_tags.find(2)
)
dofs_to_remove_in = np.setdiff1d(dofs_to_remove_in, dofs_cut_in)

dofs_to_remove_out = dfx.fem.locate_dofs_topological(
    mixed_space.sub(1), gdim, cells_tags.find(1)
)
dofs_cut_out = dfx.fem.locate_dofs_topological(
    mixed_space.sub(1), gdim, cells_tags.find(2)
)
dofs_to_remove_out = np.setdiff1d(dofs_to_remove_out, dofs_cut_out)

solution_uh_out.x.array[dofs_cut_out] = solution_uh_out.x.array[dofs_cut_out] / 2.0
solution_uh_in.x.array[dofs_cut_in] = solution_uh_in.x.array[dofs_cut_in] / 2.0
solution_uh_out.x.array[dofs_to_remove_out] = 0.0
solution_uh_in.x.array[dofs_to_remove_in] = 0.0
solution_uh_out = solution_uh_out.collapse()
solution_uh_in = solution_uh_in.collapse()
solution_uh.x.array[:] = solution_uh_in.x.array[:] + solution_uh_out.x.array[:]

save_function(solution_uh, f"solution_{str(i).zfill(2)}")
save_function(phi_h, f"levelset_{str(i).zfill(2)}")

# Discretization error computation

reference_element = element("Lagrange", cell_name, primal_degree + 2, shape=(gdim,))
reference_space = dfx.fem.functionspace(mesh, reference_element)

reference_exact_solution = dfx.fem.Function(reference_space)
reference_exact_solution.interpolate(exact_solution)
reference_solution_uh = dfx.fem.Function(reference_space)
reference_solution_uh.interpolate(solution_uh)

reference_error = reference_exact_solution - reference_solution_uh

# H10 error
h10_norm_exact_solution = (
    ufl.inner(
        ufl.grad(reference_exact_solution), ufl.grad(reference_exact_solution)
    )
    * dx
)
h10_norm_exact_solution = assemble_scalar(dfx.fem.form(h10_norm_exact_solution))

h10_norm = ufl.inner(ufl.grad(reference_error), ufl.grad(reference_error))

v0 = ufl.TrialFunction(dg0_space)
h10_local_fct = dfx.fem.Function(dg0_space)

h10_local = ufl.inner(h10_norm, v0) * dx
h10_local_form = dfx.fem.form(h10_local)
h10_local_vec = assemble_vector(h10_local_form)
h10_local_fct.x.array[:] = h10_local_vec.array[:]

save_function(h10_local_fct, f"h10_local_error_{str(i).zfill(2)}")

h10_global_err = np.sqrt(np.sum(h10_local_vec.array[:]) / h10_norm_exact_solution)
results["H10 relative error"].append(h10_global_err)

# L2 error
l2_norm_exact_solution = (
    ufl.inner(reference_exact_solution, reference_exact_solution) * dx
)
l2_norm_exact_solution = assemble_scalar(dfx.fem.form(l2_norm_exact_solution))

l2_norm = ufl.inner(reference_error, reference_error)

v0 = ufl.TrialFunction(dg0_space)
l2_local_fct = dfx.fem.Function(dg0_space)

l2_local = ufl.inner(l2_norm, v0) * dx
l2_local_form = dfx.fem.form(l2_local)
l2_local_vec = assemble_vector(l2_local_form)
l2_local_fct.x.array[:] = l2_local_vec.array[:]

save_function(l2_local_fct, f"l2_local_error_{str(i).zfill(2)}")

l2_global_err = np.sqrt(np.sum(l2_local_vec.array[:]) / l2_norm_exact_solution)
results["L2 relative error"].append(l2_global_err)

df = pl.DataFrame(results)
df.write_csv(os.path.join(output_dir, "results.csv"))
print(df)

if i < num_iterations - 1:
    mesh = dfx.mesh.refine(mesh)[0]

h10_slope, _ = np.polyfit(
    np.log(results["dof"][:]), np.log(results["H10 relative error"][:]), 1
)
l2_slope, _ = np.polyfit(
    np.log(results["dof"][:]), np.log(results["L2 relative error"][:]), 1
)

print("H10 relative error slope:", h10_slope)
print("L2 relative error slope:", l2_slope)
