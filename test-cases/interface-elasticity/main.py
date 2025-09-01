import argparse
from   basix.ufl import element, mixed_element
import numpy as np
import dolfinx as dfx
from   dolfinx.fem import assemble_scalar
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import os
import petsc4py.PETSc as PETSc
import ufl
import yaml

# Import phiFEM modules
from phiFEM.phifem.mesh_scripts import compute_tags_measures

from tags_plot.plot import plot_mesh_tags
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog="Run the demo.",
                                 description="Run phiFEM on a multimaterial elasticity test case.")

parser.add_argument("parameters", type=str, help="Name of parameters file (without yaml extension).")

args = parser.parse_args()
parameters = args.parameters

parameters_path = os.path.join(parent_dir, parameters + ".yaml")
output_dir      = os.path.join(parent_dir, parameters)

if not os.path.isdir(output_dir):
	print(f"{output_dir} directory not found, we create it.")
	os.mkdir(os.path.join(parent_dir, output_dir))

def save_function(fct, file_name):
    mesh = fct.function_space.mesh
    fct_element = fct.function_space.element.basix_element
    deg = fct_element.degree
    if deg > 1:
        element_family = fct_element.family.name
        mesh = fct.function_space.mesh
        cg1_element = element(element_family, mesh.topology.cell_name(), 1, shape=fct.function_space.value_shape)
        cg1_space = dfx.fem.functionspace(mesh, cg1_element)
        cg1_fct = dfx.fem.Function(cg1_space)
        cg1_fct.interpolate(fct)
        with XDMFFile(mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(cg1_fct)
    else:
        with XDMFFile(mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(fct)

from data import levelset_1, levelset_2, sigma_in, sigma_out, epsilon

with open(parameters_path, "rb") as f:
    parameters = yaml.safe_load(f)

# Extract parameters
mesh_size                 = parameters["initial_mesh_size"]
primal_degree             = parameters["primal_degree"]
flux_degree               = parameters["flux_degree"]
auxiliary_degree          = parameters["auxiliary_degree"]
levelset_number           = parameters["levelset_number"]
levelset_degree           = parameters["levelset_degree"]
detection_degree          = parameters["boundary_detection_degree"]
bbox                      = parameters["bbox"]
penalization_coefficient  = parameters["penalization_coefficient"]
stabilization_coefficient = parameters["stabilization_coefficient"]
cell_type                 = parameters["cell_type"]

if levelset_number == 1:
    try:
        from data import detection_levelset_1
    except ImportError:
        print("detection_levelset_1 not found in data, use levelset instead.")
        detection_levelset = levelset_1
    levelset = levelset_1
elif levelset_number == 2:
    try:
        from data import detection_levelset_2
    except ImportError:
        print("detection_levelset_2 not found in data, use levelset instead.")
        detection_levelset = levelset_2
    levelset = levelset_2
else:
    raise ValueError("Parameter error levelset_number can only be 1 or 2.")

# Create the background mesh
nx = int(np.abs(bbox[0][1] - bbox[0][0]) / mesh_size)
ny = int(np.abs(bbox[1][1] - bbox[1][0]) / mesh_size)
# Quads cells
if cell_type == "triangle":
    cell_type = dfx.cpp.mesh.CellType.triangle
elif cell_type == "quadrilateral":
    cell_type = dfx.cpp.mesh.CellType.quadrilateral
else:
    raise ValueError("Parameter error cell_type can only be 'triangle' or 'quadrilateral'.")
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, np.asarray(bbox).T, [nx, ny], cell_type)

cells_tags, facets_tags, _, d_from_inside, d_from_outside, _ = compute_tags_measures(mesh, levelset, detection_degree, box_mode=True)

fig = plt.figure()
ax = fig.subplots()
plot_mesh_tags(mesh, facets_tags, ax, linewidth=1.5)
plt.savefig(os.path.join(output_dir, "facets_tags.png"), dpi=500, bbox_inches="tight")

fig = plt.figure()
ax = fig.subplots()
plot_mesh_tags(mesh, cells_tags, ax, linewidth=1.)
plt.savefig(os.path.join(output_dir, "cells_tags.png"), dpi=500, bbox_inches="tight")

gdim = mesh.geometry.dim
cell_name = mesh.topology.cell_name()
primal_element = element("Lagrange", cell_name, primal_degree, shape=(gdim,))
flux_element   = element("Lagrange", cell_name, flux_degree,   shape=(gdim, gdim))
auxiliary_element = element("Lagrange", cell_name, auxiliary_degree, shape=(gdim,))

mixd_element = mixed_element([primal_element,
                              primal_element,
                              flux_element,
                              flux_element,
                              auxiliary_element])

dg0_element = element("DG", cell_name, 0)
dg0_space = dfx.fem.functionspace(mesh, dg0_element)

levelset_element = element("Lagrange", cell_name, levelset_degree)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)

primal_space = dfx.fem.functionspace(mesh, primal_element)
mixed_space  = dfx.fem.functionspace(mesh, mixd_element)

def point_source(x):
    return np.logical_and(np.isclose(x[0, :], 1.5),
                            np.abs(x[1, :] - 0.75 * np.ones_like(x[1, :])) <= 2.0/ny)

# Defines mapping between matrices and functions
primal_space_y = primal_space.sub(1)
primal_space_y, primal_dofmap_y = primal_space_y.collapse()
dofs_primal_space_y = primal_space_y.tabulate_dof_coordinates()[:,:2]

primal_dofs_y = dfx.fem.locate_dofs_geometrical(primal_space_y, point_source)
point_source_dofs = primal_dofmap_y[primal_dofs_y[0]]

# Primal problem RHS vector
f_h = dfx.fem.Function(primal_space)
f_h.sub(1).x.array[point_source_dofs] = -1.

# Normalization of the point source function
norm_f_h_int = ufl.sqrt(ufl.inner(f_h, f_h)) * ufl.dx
norm_f_h_form = dfx.fem.form(norm_f_h_int)
norm_f_h = assemble_scalar(norm_f_h_form)
f_h /= norm_f_h

phi_h = dfx.fem.Function(levelset_space)
phi_h.interpolate(levelset)

u_in, u_out, y_in, y_out, p = ufl.TrialFunctions(mixed_space)
v_in, v_out, z_in, z_out, q = ufl.TestFunctions (mixed_space)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

def boundary_dbc(x):
    left  = np.isclose(x[0], -1.5).astype(bool)
    return left

boundary_dbc_facets = dfx.mesh.locate_entities_boundary(mesh,
                                                        gdim - 1,
                                                        boundary_dbc)

# Create a FE function from outer space
u_dbc = dfx.fem.Function(mixed_space)
u_dbc_in, u_dbc_out, _, _, _ = u_dbc.split()

bc_in_dofs = dfx.fem.locate_dofs_topological(mixed_space.sub(0), gdim - 1, boundary_dbc_facets)
bc_out_dofs = dfx.fem.locate_dofs_topological(mixed_space.sub(1), gdim - 1, boundary_dbc_facets)
dbc_in  = dfx.fem.dirichletbc(u_dbc_in,  bc_in_dofs)
dbc_out = dfx.fem.dirichletbc(u_dbc_out, bc_out_dofs)
bcs = [dbc_in, dbc_out]

n = ufl.FacetNormal(mesh)
boundary_in = ufl.inner(ufl.dot(y_in, n), v_in)
boundary_out = ufl.inner(ufl.dot(y_out, n), v_in)

h_T = ufl.CellDiameter(mesh)

stiffness_in  = ufl.inner(sigma_in(u_in),   epsilon(v_in))
stiffness_out = ufl.inner(sigma_out(u_out), epsilon(v_out))

penalization = penalization_coefficient * \
            ( ufl.inner(y_in + sigma_in(u_in), z_in + sigma_in(v_in))\
            + ufl.inner(y_out + sigma_out(u_out), z_out + sigma_out(v_out)) \
+ h_T**(-2) * ufl.inner(ufl.dot(y_in, ufl.grad(phi_h)) - ufl.dot(y_out, ufl.grad(phi_h)), ufl.dot(z_in, ufl.grad(phi_h)) - ufl.dot(z_out, ufl.grad(phi_h))) \
+ h_T**(-2) * ufl.inner(u_in - u_out + h_T**(-1) * p * phi_h, v_in - v_out + h_T**(-1) * q * phi_h))

stabilization_facets_in = stabilization_coefficient * \
                    ufl.avg(h_T) * \
                    ufl.inner(ufl.jump(sigma_in(u_in), n),
                              ufl.jump(sigma_in(v_in), n))

stabilization_cells_in = stabilization_coefficient * \
                    ufl.inner(ufl.div(y_in), ufl.div(z_in))

stabilization_cells_out = stabilization_coefficient * \
                    ufl.inner(ufl.div(y_out), ufl.div(z_out))

stabilization_facets_out = stabilization_coefficient * \
                    ufl.avg(h_T) * \
                    ufl.inner(ufl.jump(sigma_out(u_out), n),
                              ufl.jump(sigma_out(v_out), n))

a =  stiffness_in             * dx((1,2)) \
   + stiffness_out            * dx((2,3)) \
   + penalization             * dx(2) \
   + stabilization_facets_in  * dS(3) \
   + stabilization_facets_out * dS(4) \
   + stabilization_cells_in   * dx(2) \
   + stabilization_cells_out  * dx(2)

boundary_in_int  = boundary_in  * d_from_inside
boundary_out_int = boundary_out * d_from_outside

bilinear_form = dfx.fem.form(a)
bdy_in_form = dfx.fem.form(boundary_in_int)
bdy_out_form = dfx.fem.form(boundary_out_int)
A_temp    = assemble_matrix(bilinear_form, bcs=bcs)
A_bdy_in  = assemble_matrix(bdy_in_form,   bcs=bcs)
A_bdy_out = assemble_matrix(bdy_out_form,  bcs=bcs)
A_temp.assemble()
A_bdy_in.assemble()
A_bdy_out.assemble()
A = A_temp + A_bdy_in + A_bdy_out

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

stabilization_rhs_in = stabilization_coefficient * \
                    (ufl.inner(f_h, ufl.div(z_in)))
stabilization_rhs_out = stabilization_coefficient * \
                    (ufl.inner(f_h, ufl.div(z_out)))
rhs_in  = ufl.inner(f_h, v_in)
rhs_out = ufl.inner(f_h, v_out)

L = rhs_in                * dx((1,2)) \
  + rhs_out               * dx((2,3)) \
  - stabilization_rhs_in  * dx(2) \
  - stabilization_rhs_out * dx(2)

linear_form = dfx.fem.form(L)
b = assemble_vector(linear_form)

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

# Combine the in and out solutions
solution_h = dfx.fem.Function(mixed_space)
solution_uh, _, _, _, _ = solution_h.split()
solution_uh = solution_uh.collapse()

mesh.topology.create_connectivity(gdim, gdim)
dofs_cut = dfx.fem.locate_dofs_topological(mixed_space.sub(0), gdim, cells_tags.find(2))
dofs_direct_interface = dfx.fem.locate_dofs_topological(mixed_space.sub(0), gdim - 1, facets_tags.find(6))
dofs_to_turn_off = np.union1d(dofs_cut, dofs_direct_interface)

solution_uh_in.x.array[dofs_to_turn_off] = 0.
solution_uh_in = solution_uh_in.collapse()
solution_uh_out = solution_uh_out.collapse()
solution_uh.x.array[:] = solution_uh_in.x.array[:] + solution_uh_out.x.array[:]

save_function(solution_uh,     "solution")
save_function(solution_uh_in,  "solution_in")
save_function(solution_uh_out, "solution_out")
save_function(phi_h,           "levelset")