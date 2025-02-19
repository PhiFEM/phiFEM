from   basix.ufl import element
import numpy as np
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import pandas as pd
from   petsc4py.PETSc import Options, KSP

from phiFEM.phifem.continuous_functions import ContinuousFunction, Levelset
from phiFEM.phifem.solver import PhiFEMSolver


print("==============================")
print("       Flower test case")
print("==============================")

"""
==============================
   phiFEM solver parameters
==============================
"""
mesh_size                 = 0.04                     # Initial mesh size
bbox                      = np.array([[-4.5, 4.5],
                                      [-4.5, 4.5]]) # Domain bounding box mesh
finite_element_degree     = 1                       # FE degree for w_h
levelset_degree           = 2                       # FE degree for φ_h
boundary_detection_degree = 4                       # Degree of the boundary detection
boundary_refinement_type  = 'h'                     # Refinement type used in the hierarchical approximation of the boundary error
use_fine_space            = True                    # If true, uses the proper FE degree for the product w_h*φ_h, if False use the degree of w_h

"""
==============================
          Input data
==============================
"""
# Smooth grading is taken as a radial arctan
def atan_r(x, radius=1., slope=1.):
    r = np.sqrt(np.square(x[0, :]) + np.square(x[1, :]))
    r0 = np.full_like(r, radius)
    val = np.arctan(slope * (r - r0))
    return val

# Implementation of a graded smooth-min function inspired from: https://iquilezles.org/articles/smin/
def smin(x, y_1, y_2, kmin=0., kmax=1.):
    k = kmax * ((np.pi/2. - atan_r(x, radius=3., slope=15.))/np.pi/2.) + kmin
    return np.maximum(k, np.minimum(y_1, y_2)) - np.linalg.norm(np.maximum(np.vstack([k, k]) - np.vstack([y_1, y_2]), 0.), axis=0)

# Levelset and RHS expressions taken from: https://academic.oup.com/imajna/article-abstract/42/1/333/6041856?redirectedFrom=fulltext
def expression_levelset(x):
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

def expression_detection_levelset(x):
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

# The source term is piecewise constant equal to 10 in a circle on the right of the domain and 0 elsewhere.
def expression_rhs(x):
    x1 = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.))
    y1 = 0.
    r1 = np.sqrt(2.) * 2. * (np.sin(np.pi/8.) + np.cos(np.pi/8.)) * np.sin(np.pi/8.)

    val = np.square(x[0, :] - np.full_like(x[0, :], x1)) + \
          np.square(x[1, :] - np.full_like(x[1, :], y1))
        
    return np.where(val <= np.square(r1)/2., 10., 0.)

"""
==============================
Defines continuous functions data
==============================
"""
source_term    = ContinuousFunction(expression_rhs)
levelset       = Levelset(expression_levelset)
levelset.set_detection_expression(expression_detection_levelset)

"""
==============================
Create the initial background mesh
==============================
"""
nx = int(np.abs(bbox[0, 1] - bbox[0, 0]) * np.sqrt(2.) / mesh_size)
ny = int(np.abs(bbox[1, 1] - bbox[1, 0]) * np.sqrt(2.) / mesh_size)
bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox.T, [nx, ny])

"""
==============================
   Defines finite elements
==============================
"""
fe_element       = element("Lagrange", bg_mesh.topology.cell_name(), finite_element_degree)
levelset_element = element("Lagrange", bg_mesh.topology.cell_name(), levelset_degree)

"""
==============================
   Set up the PETSc solver
==============================
"""
options = Options()
options["ksp_type"]      = "cg"
options["pc_type"]       = "hypre"
options["ksp_rtol"]      = 1e-7
options["pc_hypre_type"] = "boomeramg"
petsc_solver = KSP().create(bg_mesh.comm)
petsc_solver.setFromOptions()

"""
==============================
  Initialize phiFEM solver
==============================
"""
phiFEM_solver = PhiFEMSolver(bg_mesh,
                             fe_element,
                             petsc_solver,
                             levelset_element=levelset_element,
                             detection_degree=boundary_detection_degree,
                             use_fine_space=use_fine_space,
                             boundary_refinement_type=boundary_refinement_type)

"""
==============================
Pass the input data to the phiFEM solver
==============================
"""
phiFEM_solver.set_source_term(source_term)
phiFEM_solver.set_levelset(levelset)

"""
==============================
Locate {φ_h = 0} and tag the cells
==============================
"""
phiFEM_solver.compute_tags()

"""
==============================
Set up the phiFEM variational formulation
==============================
"""
_, _, _, num_dofs = phiFEM_solver.set_variational_formulation()

"""
==============================
Assemble the phiFEM linear system
==============================
"""
phiFEM_solver.assemble()

"""
==============================
Solve the phiFEM linear system
==============================
"""
phiFEM_solver.solve()
uh = phiFEM_solver.get_solution()

# Since we set use_fine_space = True, uh is of degree 3, we need to interpolate it to a degree 1 function as XDMF handles degree 1 only.
cg1_element = element("Lagrange", uh.function_space.mesh.topology.cell_name(), 1)
cg1_space = dfx.fem.functionspace(uh.function_space.mesh, cg1_element)
uh_cg1 = dfx.fem.Function(cg1_space)
uh_cg1.interpolate(uh)

with XDMFFile(uh_cg1.function_space.mesh.comm, "solution.xdmf", "w") as of:
    of.write_mesh(uh_cg1.function_space.mesh)
    of.write_function(uh_cg1)

"""
==============================
Compute the residual a posteriori error estimator
==============================
"""
_ = phiFEM_solver.estimate_residual()

H10_residual = phiFEM_solver.get_H10_residual()
global_H10_estimator = np.sqrt(H10_residual.x.petsc_vec.array.sum())
with XDMFFile(H10_residual.function_space.mesh.comm, "local_estimators.xdmf", "w") as of:
    of.write_mesh(H10_residual.function_space.mesh)
    of.write_function(H10_residual)

df = pd.DataFrame({"Num. dofs":      [num_dofs],
                   "H10 estimator":  [global_H10_estimator]})

print("==============================")
print("           Results")
print("==============================")
print(df)