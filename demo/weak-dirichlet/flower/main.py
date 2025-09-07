import argparse
from basix.ufl import element, mixed_element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
import os
import petsc4py.PETSc as PETSc
import ufl

from data import detection_levelset, levelset, source_term, dirichlet_data
from phifem.mesh_scripts import compute_tags_measures

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="main.py", description="Run weak dirichlet phiFEM demo."
)

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
# Degree of φh
levelset_degree = 1
# Penalization and stabilization parameters
pen_coef = 1.0
stab_coef = 1.0

bbox = [[-4.5, -4.5], [4.5, 4.5]]
bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox, [200, 200])

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
primal_element = element("Lagrange", cell_name, primal_degree)
auxiliary_element = element("Lagrange", cell_name, primal_degree)
levelset_element = element("Lagrange", cell_name, levelset_degree)
mxd_element = mixed_element([primal_element, auxiliary_element])

primal_space = dfx.fem.functionspace(mesh, primal_element)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)
mixed_space = dfx.fem.functionspace(mesh, mxd_element)

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

# Dirichlet data (constant=0 here but added for the sake of the demo.)
u_D = dfx.fem.Function(primal_space)
u_D.interpolate(dirichlet_data)

u, p = ufl.TrialFunctions(mixed_space)
v, q = ufl.TestFunctions(mixed_space)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

h_T = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Bilinear form
a = (
    ufl.inner(ufl.grad(u), ufl.grad(v)) * dx((1, 2))
    - ufl.inner(ufl.inner(ufl.grad(u), n), v) * ds
    + (
        pen_coef
        * h_T ** (-2)
        * ufl.inner(
            u - h_T ** (-1) * ufl.inner(phi_h, p), v - h_T ** (-1) * ufl.inner(phi_h, q)
        )
        * dx(2)
    )
    + (
        stab_coef
        * h_T**2
        * ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v)))
        * dx(2)
    )
    + (
        stab_coef
        * ufl.avg(h_T)
        * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n))
        * dS((2, 3))
    )
)

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form)
A.assemble()

# Linear form
L = (
    ufl.inner(f_h, v) * dx((1, 2))
    + (
        pen_coef
        * h_T ** (-2)
        * ufl.inner(u_D, v - h_T ** (-1) * ufl.inner(phi_h, q))
        * dx(2)
    )
    - stab_coef * h_T**2 * ufl.inner(f_h, ufl.div(ufl.grad(v))) * dx(2)
)

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
solution_uh, _ = solution_wh.split()
solution_uh.collapse()

"""
=================================
 Save solution for visualization
=================================
"""
with XDMFFile(mesh.comm, os.path.join(output_dir, "solution.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(solution_uh)
