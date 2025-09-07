import argparse
from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
import os
import petsc4py.PETSc as PETSc
import ufl

from data import detection_levelset, levelset, source_term
from phifem.mesh_scripts import compute_tags_measures

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="main.py", description="Run strong dirichlet phiFEM demo."
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


# Degree of wh
fe_degree = 1
# Degree of φh
levelset_degree = 1
# Degree of uh = wh·φh
solution_degree = 1
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
primal_element = element("Lagrange", cell_name, fe_degree)
levelset_element = element("Lagrange", cell_name, levelset_degree)
solution_element = element("Lagrange", cell_name, solution_degree)

primal_space = dfx.fem.functionspace(mesh, primal_element)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)
solution_space = dfx.fem.functionspace(mesh, solution_element)

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

w = ufl.TrialFunction(primal_space)
phiw = phi_h * w
v = ufl.TestFunction(primal_space)
phiv = phi_h * v

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

h_T = ufl.CellDiameter(mesh)
h_E = ufl.FacetArea(mesh)
n = ufl.FacetNormal(mesh)

a = (
    ufl.inner(ufl.grad(phiw), ufl.grad(phiv)) * dx((1, 2))
    - ufl.inner(ufl.inner(ufl.grad(phiw), n), phiv) * ds
    + (
        stab_coef
        * h_T**2
        * ufl.inner(ufl.div(ufl.grad(phiw)), ufl.div(ufl.grad(phiv)))
        * dx(2)
    )
    + (
        stab_coef
        * ufl.avg(h_T)
        * ufl.inner(ufl.jump(ufl.grad(phiw), n), ufl.jump(ufl.grad(phiv), n))
        * dS((2, 3))
    )
)

bilinear_form = dfx.fem.form(a)
A = assemble_matrix(bilinear_form)
A.assemble()

# Linear form
L = ufl.inner(f_h, phiv) * dx((1, 2)) - stab_coef * h_T**2 * ufl.inner(
    f_h, ufl.div(ufl.grad(phiv))
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

# When solving on the background mesh, we need mumps to handle the null space of the matrix
if mesh_type == "bg":
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

"""
===============================
 Solve the φ-FEM linear system
===============================
"""
solution_wh = dfx.fem.Function(primal_space)
ksp.solve(b, solution_wh.x.petsc_vec)
ksp.destroy()

solution_uh = dfx.fem.Function(solution_space)
solution_wh_s_space = dfx.fem.Function(solution_space)
solution_wh_s_space.interpolate(solution_wh)
phi_h_s_space = dfx.fem.Function(solution_space)
phi_h_s_space.interpolate(phi_h)

solution_uh.x.array[:] = solution_wh_s_space.x.array[:] * phi_h_s_space.x.array[:]

"""
=================================
 Save solution for visualization
=================================
"""
with XDMFFile(mesh.comm, os.path.join(output_dir, "solution.xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(solution_uh)
