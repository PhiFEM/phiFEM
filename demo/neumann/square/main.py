from   basix.ufl import element, mixed_element
import numpy as np
import dolfinx as dfx
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import petsc4py.PETSc as PETSc
import ufl

from phiFEM.phifem.mesh_scripts import compute_tags, compute_outward_normal

"""
=============================
 Tilted square levelset data 
=============================
"""

# Levelset
tilt_angle = np.pi/6.
def _rotation(angle, x):
    if x.shape[0] == 3:
        R = np.array([[np.cos(angle),   np.sin(angle), 0],
                      [-np.sin(angle),  np.cos(angle), 0],
                      [            0,               0, 1]])
    elif x.shape[0] == 2:
        R = np.array([[np.cos(angle),   np.sin(angle)],
                      [-np.sin(angle),  np.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(np.asarray(x))

def detection_levelset(x):
    y = np.sum(np.abs(_rotation(tilt_angle - np.pi/4., x)), axis=0)
    return y - np.sqrt(2.)/2.

def levelset(x):
    vect = np.full_like(x, 0.5)
    val = -np.sin(np.pi * (_rotation(tilt_angle, x - _rotation(-tilt_angle, vect)))[0, :]) * \
           np.sin(np.pi * (_rotation(tilt_angle, x - _rotation(-tilt_angle, vect)))[1, :])
    return val

# Analytical solution
def exact_solution(x):
    return np.cos(2. * np.pi * _rotation(tilt_angle, x)[0, :]) * \
           np.cos(2. * np.pi * _rotation(tilt_angle, x)[1, :])

# Source term
def source_term(x):
    return 8. * np.pi**2 * exact_solution(x) + exact_solution(x)

def neumann(x):
    rx = _rotation(tilt_angle, x)

    def _dx(rx):
        return - 2. * np.pi * np.sin(2. * np.pi * rx[0,:]) * \
                              np.cos(2. * np.pi * rx[1,:])
    def _dy(rx):
        return - 2. * np.pi * np.cos(2. * np.pi * rx[0,:]) * \
                              np.sin(2. * np.pi * rx[1,:])

    vals = -_dy(rx)
    mask = np.where(np.abs(rx[1,:]) < rx[0,: ])[0]
    vals[mask] = _dx(rx[:,mask])
    mask = np.where(np.abs(rx[0,:]) < rx[1,:])[0]
    vals[mask] = _dy(rx[:, mask])
    mask = np.where(np.abs(rx[1,:]) < -rx[0,: ])[0]
    vals[mask] = -_dx(rx[:, mask])
    return vals

"""
=========================
 Initial background mesh
=========================
"""
bbox = [[-1., -1.], [1., 1.]]
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox, [200, 200])

cells_tags, facets_tags, mesh = compute_tags(mesh,
                                             detection_levelset,
                                             2,
                                             box_mode=True)

# Degree of uh
primal_degree = 1
# Degree of yh
vector_degree = 1
# Degree of ph
auxiliary_degree = 0
# Degree of φh
levelset_degree = 2

cell_name = mesh.topology.cell_name()
gdim = mesh.geometry.dim
primal_element    = element("Lagrange", cell_name, primal_degree)
auxiliary_element = element("DG",       cell_name, auxiliary_degree)
vector_element    = element("Lagrange", cell_name, vector_degree, shape=(gdim,))
levelset_element  = element("Lagrange", cell_name, levelset_degree)
mxd_element = mixed_element([primal_element, vector_element, auxiliary_element])

primal_space    = dfx.fem.functionspace(mesh, primal_element)
auxiliary_space = dfx.fem.functionspace(mesh, auxiliary_element)
vector_space    = dfx.fem.functionspace(mesh, vector_element)
mixed_space     = dfx.fem.functionspace(mesh, mxd_element)
levelset_space  = dfx.fem.functionspace(mesh, levelset_element)

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

# Dirichlet data (added here for the sake of demo)
u_N = dfx.fem.Function(primal_space)
u_N.interpolate(neumann)

u, y, p = ufl.TrialFunctions(mixed_space)
v, z, q = ufl.TestFunctions(mixed_space)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

h_T = ufl.CellDiameter(mesh)
h_E = ufl.FacetArea(mesh)
n   = ufl.FacetNormal(mesh)

# In box_mode (see l93), the boundary term needs a special treatment
omega_h_n = compute_outward_normal(mesh, levelset)
dg0_element = element("DG", cell_name, 0)
dg0_space = dfx.fem.functionspace(mesh, dg0_element)
omega_h_indicator = dfx.fem.Function(dg0_space)
inside_cells = cells_tags.find(1)
cut_cells = cells_tags.find(2)
omega_h_cells = np.union1d(inside_cells, cut_cells)
omega_h_indicator.x.array[omega_h_cells] = 1.

# Bilinear form
boundary = ufl.inner(2. * ufl.avg(ufl.inner(y, omega_h_n) * omega_h_indicator),
                     2. * ufl.avg(ufl.inner(v, omega_h_indicator)))

stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)

penalization = 1.0 * (ufl.inner(y + ufl.grad(u), z + ufl.grad(v)) \
                 + ufl.inner(ufl.div(y) + u, ufl.div(z) + v) \
                 + h_T**(-2) * ufl.inner(ufl.inner(y, ufl.grad(phi_h)) + h_T**(-1) * ufl.inner(p, phi_h), ufl.inner(z, ufl.grad(phi_h)) + h_T**(-1) * ufl.inner(q, phi_h)))


stabilization_facets = 10.0 * ufl.avg(h_E) * \
                        ufl.inner(ufl.jump(ufl.grad(u), n),
                                  ufl.jump(ufl.grad(v), n))

a = stiffness              * (dx(1) + dx(2)) \
    + boundary             * dS(4) \
    + penalization         * dx(2) \
    + stabilization_facets * dS(2)

# Linear form
rhs = ufl.inner(f_h, v)
penalization_rhs = 1.0 * (- h_T**(-2) * ufl.inner(u_N, ufl.sqrt(ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h))) * (ufl.inner(z, ufl.grad(phi_h)) + h_T**(-1) * ufl.inner(q, phi_h))) \
                + ufl.inner(f_h, ufl.div(z) + v))

L = rhs                 * (dx(1) + dx(2)) \
    + penalization_rhs  * dx(2)

bilinear_form = dfx.fem.form(a)
linear_form = dfx.fem.form(L)

A = assemble_matrix(bilinear_form)
b = assemble_vector(linear_form)
A.assemble()

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
with XDMFFile(mesh.comm, "solution.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(solution_uh)

u_exact_h = dfx.fem.Function(primal_space)
u_exact_h.interpolate(exact_solution)

with XDMFFile(mesh.comm, "exact_solution.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(u_exact_h)