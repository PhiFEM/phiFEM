import argparse
from   basix.ufl import element, mixed_element
import numpy as np
import dolfinx as dfx
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import os
import petsc4py.PETSc as PETSc
import ufl
import yaml

# Import phiFEM modules
from phiFEM.phifem.mesh_scripts import compute_tags, compute_outward_normal

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

from data import levelset, sigma_in, sigma_out, epsilon, neumann

try:
    from data import detection_levelset
except ImportError:
    print("detection_levelset not found in data, use levelset instead.")
    detection_levelset = levelset

with open(parameters_path, "rb") as f:
    parameters = yaml.safe_load(f)

# Extract parameters
mesh_size                 = parameters["initial_mesh_size"]
primal_degree             = parameters["primal_degree"]
flux_degree               = parameters["flux_degree"]
auxiliary_degree          = parameters["auxiliary_degree"]
levelset_degree           = parameters["levelset_degree"]
detection_degree          = parameters["boundary_detection_degree"]
bbox                      = parameters["bbox"]
penalization_coefficient  = parameters["penalization_coefficient"]
stabilization_coefficient = parameters["stabilization_coefficient"]

# Create the background mesh
nx = int(np.abs(bbox[0][1] - bbox[0][0]) * np.sqrt(2.) / mesh_size)
ny = int(np.abs(bbox[1][1] - bbox[1][0]) * np.sqrt(2.) / mesh_size)
# Quads cells
cell_type = dfx.cpp.mesh.CellType(-4)
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, np.asarray(bbox).T, [nx, ny], cell_type)

cells_tags, facets_tags, _ = compute_tags(mesh, levelset, detection_degree, box_mode=True)

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

phi_h = dfx.fem.Function(levelset_space)
phi_h.interpolate(levelset)

u_in, u_out, y_in, y_out, p = ufl.TrialFunctions(mixed_space)
v_in, v_out, z_in, z_out, q = ufl.TestFunctions (mixed_space)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

def boundary_dbc(x):
    left  = np.isclose(x[0], 0.).astype(bool)
    return left

def boundary_nbc(x):
    right = np.isclose(x[0], 1.).astype(bool)
    return right

boundary_dbc_facets = dfx.mesh.locate_entities_boundary(mesh,
                                                        gdim - 1,
                                                        boundary_dbc)
boundary_nbc_facets = dfx.mesh.locate_entities_boundary(mesh,
                                                        gdim - 1,
                                                        boundary_nbc)

boundary_indices = np.hstack([boundary_dbc_facets, boundary_nbc_facets])
sorted_indices = np.argsort(boundary_indices)
dbc_markers = np.ones_like(boundary_dbc_facets).astype(np.int32)
nbc_markers = 2 * np.ones_like(boundary_nbc_facets).astype(np.int32)
boundary_markers = np.hstack([dbc_markers, nbc_markers])
boundary_tags = dfx.mesh.meshtags(mesh, gdim - 1, boundary_indices[sorted_indices], boundary_markers[sorted_indices])

ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundary_tags)

# Create a FE function from outer space
u_dbc = dfx.fem.Function(mixed_space)
u_nbc = dfx.fem.Function(mixed_space)
_, u_dbc_out, _, _, _ = u_dbc.split()
_, u_nbc_out, _, _, _ = u_nbc.split()
u_nbc_out.interpolate(neumann)

bc_dofs = dfx.fem.locate_dofs_topological(mixed_space.sub(1), gdim - 1, boundary_dbc_facets)
bc = dfx.fem.dirichletbc(u_dbc_out, bc_dofs)

# Inside domain outward normal and indicator
interface_outward_n = compute_outward_normal(mesh, levelset)
inside_indicator = dfx.fem.Function(dg0_space)
inside_indicator.x.petsc_vec.set(0.)
inside_cells = cells_tags.find(1)
cut_cells    = cells_tags.find(2)
disk_cells = np.union1d(inside_cells, cut_cells)
inside_indicator.x.array[disk_cells] = 1.

boundary_in = ufl.inner(2. * ufl.avg(ufl.dot(y_in, interface_outward_n) * inside_indicator), 2. * ufl.avg(v_in * inside_indicator))

# Outside domain inward normal and indicator
interface_inward_n = compute_outward_normal(mesh, lambda x: -levelset(x))
outside_indicator = dfx.fem.Function(dg0_space)
outside_indicator.x.petsc_vec.set(0.)
outside_cells = cells_tags.find(3)
complement_cells = np.union1d(outside_cells, cut_cells)
outside_indicator.x.array[complement_cells] = 1.

boundary_out = ufl.inner(2. * ufl.avg(ufl.dot(y_out, interface_inward_n) * outside_indicator), 2. * ufl.avg(v_out * outside_indicator))

h_T = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

stiffness_in  = ufl.inner(sigma_in(u_in),   epsilon(v_in))
stiffness_out = ufl.inner(sigma_out(u_out), epsilon(v_out))

penalization = penalization_coefficient * \
            ( ufl.inner(y_in + sigma_in(u_in), z_in + sigma_in(v_in)) \
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

a =  stiffness_in             * (dx(1) + dx(2)) \
   + stiffness_out            * (dx(2) + dx(3)) \
   + boundary_in              * dS(4) \
   + boundary_out             * dS(3) \
   + penalization             * dx(2) \
   + stabilization_facets_in  * dS(3) \
   + stabilization_facets_out * dS(4) \
   + stabilization_cells_in   * dx(2) \
   + stabilization_cells_out  * dx(2)

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()

ksp = PETSc.KSP().create(mesh.comm)
ksp.setType("preonly")
solver = ksp.create(MPI.COMM_WORLD)
solver.setFromOptions()
solver.setOperators(A)
# Configure MUMPS to handle pressure nullspace
pc = solver.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

# The RHS is not mandatory here but added for the sake of the demo (the source terms are zero)
source_term_in_h  = dfx.fem.Function(primal_space)
source_term_out_h = dfx.fem.Function(primal_space)

rhs_in  = ufl.inner(source_term_in_h,  v_in)
rhs_out = ufl.inner(source_term_out_h, v_out)
stabilization_rhs_in = stabilization_coefficient * \
                       ufl.inner(source_term_in_h, ufl.div(z_in))
stabilization_rhs_out = stabilization_coefficient * \
                        ufl.inner(source_term_out_h, ufl.div(z_out))

L = rhs_in                * (dx(1) + dx(2)) \
  + rhs_out               * (dx(2) + dx(3)) \
  + stabilization_rhs_in  * dx(2) \
  + stabilization_rhs_out * dx(2)

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
solution_uh_in = solution_uh_in.collapse()
solution_uh_out = solution_uh_out.collapse()

solution_h = dfx.fem.Function(mixed_space)
solution_uh, _, _, _, _ = solution_h.split()
solution_uh = solution_uh.collapse()
solution_uh.x.array[:] = solution_uh_in.x.array[:] + solution_uh_out.x.array[:]

save_function(solution_uh,     "solution")
save_function(solution_uh_in,  "solution_in")
save_function(solution_uh_out, "solution_out")