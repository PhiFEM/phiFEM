import argparse
from basix.ufl import element, mixed_element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import os
import petsc4py.PETSc as PETSc
import ufl

from data import detection_levelset, levelset, source_term, neumann_data, exact_solution
from phifem.mesh_scripts import compute_tags_measures

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog="main.py", description="Run neumann phiFEM demo.")

parser.add_argument(
    "mesh_type",
    type=str,
    choices=["bg", "sub"],
    help="Choose if the problem is solved on the background mesh (bg) or on a submesh (sub).",
)

args = parser.parse_args()
mesh_type = args.mesh_type

output_dir = os.path.join(parent_dir, mesh_type + "_output")

if not os.path.isdir(output_dir):
    print(f"{output_dir} directory not found, we create it.")
    os.mkdir(os.path.join(parent_dir, output_dir))


# Degree of uh
primal_degree = 1
# Degree of yh
vector_degree = 1
# Degree of ph
auxiliary_degree = 0
# Degree of φh
levelset_degree = 2
# Penalization and stabilization parameters
pen_coef = 1.0
stab_coef = 1.0

bbox = [[-1.0, -1.0], [1.0, 1.0]]
cell_type = dfx.cpp.mesh.CellType.quadrilateral
bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox, [200, 200], cell_type)

if mesh_type == "bg":
    cells_tags, facets_tags, _, ds, _, _ = compute_tags_measures(
        bg_mesh, detection_levelset, 1, box_mode=True
    )
    mesh = bg_mesh
elif mesh_type == "sub":
    cells_tags, facets_tags, mesh, _, _, _ = compute_tags_measures(
        bg_mesh, detection_levelset, 1, box_mode=False
    )
    ds = ufl.Measure("ds", domain=mesh)

cell_name = mesh.topology.cell_name()
gdim = mesh.geometry.dim
primal_element = element("Lagrange", cell_name, primal_degree)
auxiliary_element = element("DG", cell_name, auxiliary_degree)
vector_element = element("Lagrange", cell_name, vector_degree, shape=(gdim,))
levelset_element = element("Lagrange", cell_name, levelset_degree)
mxd_element = mixed_element([primal_element, vector_element, auxiliary_element])

primal_space = dfx.fem.functionspace(mesh, primal_element)
auxiliary_space = dfx.fem.functionspace(mesh, auxiliary_element)
vector_space = dfx.fem.functionspace(mesh, vector_element)
mixed_space = dfx.fem.functionspace(mesh, mxd_element)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)

"""
===================
 φ-FEM formulation
===================
"""

# Interpolation of the levelset
phi_h = dfx.fem.Function(levelset_space)
phi_h.interpolate(levelset)

# Interpolation of the source term f
f_h = dfx.fem.Function(primal_space)
f_h.interpolate(source_term)

# Neumann data
u_N = dfx.fem.Function(primal_space)
u_N.interpolate(neumann_data)

u, y, p = ufl.TrialFunctions(mixed_space)
v, z, q = ufl.TestFunctions(mixed_space)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

h_T = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Bilinear form
a = (
    (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * dx((1, 2))
    + ufl.inner(ufl.inner(y, n), v) * ds
    + (
        pen_coef
        * (
            ufl.inner(y + ufl.grad(u), z + ufl.grad(v))
            + ufl.inner(ufl.div(y) + u, ufl.div(z) + v)
            + (
                h_T ** (-2)
                * ufl.inner(
                    ufl.inner(y, ufl.grad(phi_h)) + h_T ** (-1) * ufl.inner(p, phi_h),
                    ufl.inner(z, ufl.grad(phi_h)) + h_T ** (-1) * ufl.inner(q, phi_h),
                )
            )
        )
        * dx(2)
    )
    + stab_coef * ufl.avg(h_T)
    * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n))
    * dS(3)
)

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form)
A.assemble()

# Linear form
norm_grad_phi_h = ufl.sqrt(ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h)))

L = ufl.inner(f_h, v) * dx((1, 2)) + (
    pen_coef
    * (
        -(h_T ** (-2))
        * ufl.inner(
            u_N,
            norm_grad_phi_h
            * (ufl.inner(z, ufl.grad(phi_h)) + h_T ** (-1) * ufl.inner(q, phi_h)),
        )
        + ufl.inner(f_h, ufl.div(z) + v)
    )
) * dx(2)

linear_form = dfx.fem.form(L)
b = assemble_vector(linear_form)

"""
=========================
 Set up the PETSc LU solver
=========================
"""
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
# Configure MUMPS to handle nullspace.
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

"""
===============================
 Solve the φ-FEM linear system
===============================
"""
solution_wh = dfx.fem.Function(mixed_space)
ksp.solve(b, solution_wh.x.petsc_vec)
ksp.destroy()

# Recover the primal solution from the mixed solution
solution_uh, _, _ = solution_wh.split()
solution_uh.collapse()

"""
=================================
 Save solution for visualization
=================================
"""
with XDMFFile(mesh.comm, os.path.join(output_dir, "solution.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(solution_uh)

u_exact_h = dfx.fem.Function(primal_space)
u_exact_h.interpolate(exact_solution)

with XDMFFile(mesh.comm, os.path.join(output_dir, "exact_solution.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(u_exact_h)

"""
=========================
 Local error computation
=========================
"""
ref_element = element("Lagrange", cell_name, primal_degree + 2)
ref_space = dfx.fem.functionspace(mesh, ref_element)

ref_exact_solution = dfx.fem.Function(ref_space)
ref_exact_solution.interpolate(exact_solution)
ref_solution = dfx.fem.Function(ref_space)
ref_solution.interpolate(solution_uh)
ref_error = dfx.fem.Function(ref_space)
ref_error.x.array[:] = ref_exact_solution.x.array[:] - ref_solution.x.array[:]

dg0_element = element("DG", cell_name, 0)
dg0_space = dfx.fem.functionspace(mesh, dg0_element)
v0 = ufl.TestFunction(dg0_space)

h1_error = ufl.inner(
    ufl.inner(ufl.grad(ref_error), ufl.grad(ref_error))
    + ufl.inner(ref_error, ref_error),
    v0,
) * dx((1, 2))

h1_error_form = dfx.fem.form(h1_error)
h1_error_vec = assemble_vector(h1_error_form)

h1_error_fct = dfx.fem.Function(dg0_space)
h1_error_fct.x.array[:] = h1_error_vec.array[:]

with XDMFFile(mesh.comm, os.path.join(output_dir, "h1_error.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(h1_error_fct)

h1_norm_exact_solution = (
    ufl.inner(ufl.grad(ref_exact_solution), ufl.grad(ref_exact_solution))
    + ufl.inner(ref_exact_solution, ref_exact_solution)
) * dx((1, 2))
h1_norm_exact_solution_form = dfx.fem.form(h1_norm_exact_solution)
h1_norm_exact_solution = dfx.fem.assemble_scalar(h1_norm_exact_solution_form)

print("Relative H1 error:")
print(np.sqrt(h1_error_fct.x.array.sum() / h1_norm_exact_solution))
