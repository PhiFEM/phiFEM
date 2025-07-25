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
 Flower smooth levelset data 
=============================
"""

def atan_r(x, radius=1., slope=1.):
    r = np.sqrt(np.square(x[0, :]) + np.square(x[1, :]))
    r0 = np.full_like(r, radius)
    val = np.arctan(slope * (r - r0))
    return val

# Implementation of a graded smooth-min function inspired from: https://iquilezles.org/articles/smin/
def smin(x, y_1, y_2, kmin=0., kmax=1.):
    k = kmax * ((np.pi/2. - atan_r(x, radius=2., slope=50.))/np.pi/2.) + kmin
    return np.maximum(k, np.minimum(y_1, y_2)) - np.linalg.norm(np.maximum(np.vstack([k, k]) - np.vstack([y_1, y_2]), 0.), axis=0)

# Levelset and RHS expressions taken from: https://academic.oup.com/imajna/article-abstract/42/1/333/6041856?redirectedFrom=fulltext
# This smooth definition of the levelset is used in the phiFEM formulation only (and not during the mesh tagging).
def levelset(x):
    def phi0(x):
        r = np.full_like(x[0, :], 2.)
        return np.square(x[0, :]) + np.square(x[1, :]) - np.square(r)
    val = phi0(x)

    for i in range(1, 9):
        xi = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.)) * np.cos(i * np.pi/4.)
        yi = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.)) * np.sin(i * np.pi/4.)
        ri = np.sqrt(2.) * 2. * (np.sin(np.pi/8.) + np.cos(np.pi/8.)) * np.sin(np.pi/8.)
        def phi_i(x):
            return np.square(x[0, :] - np.full_like(x[0, :], xi)) + \
                   np.square(x[1, :] - np.full_like(x[1, :], yi)) - \
                   np.square(np.full_like(x[0, :], ri))
        
        #val *= phi_i(x)
        val = smin(x, val, phi_i(x))
    return val

# A non-smooth detection expression of the levelset is used only for mesh tagging purposes in order to avoid possible non connected sets to be selected if the smooth expression was used.
def detection_levelset(x):
    def phi0(x):
        r = np.full_like(x[0, :], 2.)
        return np.square(x[0, :]) + np.square(x[1, :]) - np.square(r)
    val = phi0(x)

    for i in range(1, 9):
        xi = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.)) * np.cos(i * np.pi/4.)
        yi = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.)) * np.sin(i * np.pi/4.)
        ri = np.sqrt(2.) * 2. * (np.sin(np.pi/8.) + np.cos(np.pi/8.)) * np.sin(np.pi/8.)
        def phi_i(x):
            return np.square(x[0, :] - np.full_like(x[0, :], xi)) + \
                   np.square(x[1, :] - np.full_like(x[1, :], yi)) - \
                   np.square(np.full_like(x[0, :], ri))
        
        val = np.minimum(val, phi_i(x))
    return val

def source_term(x):
    x1 = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.))
    y1 = 0.
    r1 =  np.sqrt(2.) * 2. * (np.sin(np.pi/8.) \
        + np.cos(np.pi/8.)) * np.sin(np.pi/8.)

    val = np.square(x[0, :] - np.full_like(x[0, :], x1)) + \
          np.square(x[1, :] - np.full_like(x[1, :], y1))
        
    return np.where(val <= np.square(r1)/2., 10., 0.)

# Not necessary (added here for the sake of the demo)
def dirichlet(x):
    return np.zeros_like(x[0,:])

"""
=========================
 Initial background mesh
=========================
"""
bbox = [[-4.5, -4.5], [4.5, 4.5]]
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox, [200, 200])

cells_tags, facets_tags, mesh = compute_tags(mesh,
                                             detection_levelset,
                                             2,
                                             box_mode=True)

# Degree of uh
primal_degree = 1
# Degree of φh
levelset_degree = 2

cell_name = mesh.topology.cell_name()
primal_element    = element("Lagrange", cell_name, primal_degree)
auxiliary_element = element("Lagrange", cell_name, primal_degree)
levelset_element  = element("Lagrange", cell_name, levelset_degree)
mxd_element = mixed_element([primal_element, auxiliary_element])

primal_space   = dfx.fem.functionspace(mesh, primal_element)
levelset_space = dfx.fem.functionspace(mesh, levelset_element)
mixed_space    = dfx.fem.functionspace(mesh, mxd_element)

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
u_D = dfx.fem.Function(primal_space)
u_D.interpolate(dirichlet)

u, p = ufl.TrialFunctions(mixed_space)
v, q = ufl.TestFunctions(mixed_space)

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
boundary = ufl.inner(2. * ufl.avg(ufl.inner(ufl.grad(u), omega_h_n) * \
    omega_h_indicator), 2. * ufl.avg(ufl.inner(v, omega_h_indicator)))

stiffness = ufl.inner(ufl.grad(u), ufl.grad(v))

penalization = 1.0 * h_T**(-2) * ufl.inner(u - h_T**(-1) * ufl.inner(phi_h, p), v - h_T**(-1) * ufl.inner(phi_h, q))

stabilization_facets = 10.0 * ufl.avg(h_E) * ufl.inner(ufl.jump(ufl.grad(u), n),
                                                       ufl.jump(ufl.grad(v), n))
stabilization_cells = 10.0 * h_T**2 * ufl.inner(ufl.div(ufl.grad(u)),
                                                ufl.div(ufl.grad(v)))

a = stiffness              * (dx(1) + dx(2)) \
    - boundary             * dS(4) \
    + penalization         * dx(2) \
    + stabilization_facets * dS(2) \
    + stabilization_cells  * dx(2)

# Linear form
rhs = ufl.inner(f_h, v)
penalization_rhs = 1.0 * h_T**(-2) * ufl.inner(u_D, v - h_T**(-1)*phi_h*q)
stabilization_rhs = 10.0 * h_T**2 * ufl.inner(f_h, ufl.div(ufl.grad(v)))

L = rhs                 * (dx(1) + dx(2)) \
    + penalization_rhs  * dx(2) \
    - stabilization_rhs * dx(2)

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
solution_uh, _ = solution_wh.split()
solution_uh.collapse()

"""
=================================
 Save solution for visualization
=================================
"""
with XDMFFile(mesh.comm, "solution.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(solution_uh)