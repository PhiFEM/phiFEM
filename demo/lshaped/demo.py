from   basix.ufl import element
import numpy as np
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import pandas as pd
from   petsc4py.PETSc import Options, KSP

from phiFEM.phifem.continuous_functions import ContinuousFunction, Levelset
from phiFEM.phifem.solver import PhiFEMSolver

from FEM.compute_exact_error import compute_exact_error

print("==============================")
print("      L-shaped test case")
print("==============================")

"""
phiFEM solver parameters
"""
mesh_size                 = 0.01                    # Initial mesh size
bbox                      = np.array([[-1.5, 1.5],
                                      [-1.5, 1.5]]) # Domain bounding box mesh
finite_element_degree     = 1                       # FE degree for w_h
levelset_degree           = 2                       # FE degree for φ_h
boundary_detection_degree = 4                       # Degree of the boundary detection
boundary_refinement_type  = 'h'                     # Refinement type used in the hierarchical approximation of the boundary error
use_fine_space            = False                   # If true, uses the proper FE degree for the product w_h*φ_h, if False use the degree of w_h

"""
Input data
"""
tilt_angle = np.pi/6.
shift = np.array([np.pi/32., np.pi/32.])

def rotation(angle, x):
    if x.shape[0] == 3:
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [            0,              0, 1]])
    elif x.shape[0] == 2:
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(np.asarray(x))

def line(x, y, a, b, c):
    rotated = rotation(tilt_angle + np.pi / 4., np.vstack([x, y]))
    return a*rotated[0,:] + b*rotated[1,:] + np.full_like(x, c)

"""
We use a different expression for the boundary detection and for the solver.
"""
def expression_detection_levelset(x):
    x_shift = x[0, :] - np.full_like(x[0, :], shift[0])
    y_shift = x[1, :] - np.full_like(x[1, :], shift[1])

    line_1 = line(x_shift, y_shift, -1.,  0.,   0.)
    line_2 = line(x_shift, y_shift,  0., -1.,   0.)
    line_3 = line(x_shift, y_shift,  1.,  0., -0.5)
    line_4 = line(x_shift, y_shift,  0.,  1., -0.5)
    line_5 = line(x_shift, y_shift,  0., -1., -0.5)
    line_6 = line(x_shift, y_shift, -1.,  0., -0.5)

    reentrant_corner = np.minimum(line_1, line_2)
    top_right_corner = np.maximum(line_3, line_4)
    corner           = np.maximum(reentrant_corner, top_right_corner)
    horizontal_leg   = np.maximum(corner, line_5)
    vertical_leg     = np.maximum(horizontal_leg, line_6)
    return vertical_leg

def expression_levelset(x):
    x_shift = x - np.tile(np.array([[shift[0]], [shift[1]], [0.]]), x.shape[1])
    x_rot = rotation(tilt_angle, x_shift)
    theta = np.arctan2(x_rot[1, :], x_rot[0, :])
    values = np.cos((2./3.) * theta)
    x_rot_sq = rotation(tilt_angle + np.pi/4., x_shift)
    values *= x_rot_sq[1, :] + np.full_like(x_rot_sq[1, :], 0.5)
    values *= x_rot_sq[1, :] - np.full_like(x_rot_sq[1, :], 0.5)
    values *= x_rot_sq[0, :] + np.full_like(x_rot_sq[0, :], 0.5)
    values *= x_rot_sq[0, :] - np.full_like(x_rot_sq[0, :], 0.5)
    return values

def expression_rhs(x):
    return np.ones_like(x[0, :])

"""
Defines continuous functions data
"""
source_term    = ContinuousFunction(expression_rhs)
levelset       = Levelset(expression_levelset)
levelset.set_detection_expression(expression_detection_levelset)

"""
Create the initial background mesh
"""
nx = int(np.abs(bbox[0, 1] - bbox[0, 0]) * np.sqrt(2.) / mesh_size)
ny = int(np.abs(bbox[1, 1] - bbox[1, 0]) * np.sqrt(2.) / mesh_size)
bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox.T, [nx, ny])

"""
Defines finite elements
"""
fe_element       = element("Lagrange", bg_mesh.topology.cell_name(), finite_element_degree)
levelset_element = element("Lagrange", bg_mesh.topology.cell_name(), levelset_degree)

"""
Set up the PETSc solver
"""
options = Options()
options["ksp_type"]      = "cg"
options["pc_type"]       = "hypre"
options["ksp_rtol"]      = 1e-7
options["pc_hypre_type"] = "boomeramg"
petsc_solver = KSP().create(bg_mesh.comm)
petsc_solver.setFromOptions()

"""
Initialize phiFEM solver
"""
phiFEM_solver = PhiFEMSolver(bg_mesh,
                             fe_element,
                             petsc_solver,
                             levelset_element=levelset_element,
                             detection_degree=boundary_detection_degree,
                             use_fine_space=use_fine_space,
                             boundary_refinement_type=boundary_refinement_type)

"""
Pass the input data to the phiFEM solver
"""
phiFEM_solver.set_source_term(source_term)
phiFEM_solver.set_levelset(levelset)

"""
Locate {φ_h = 0} and tag the cells
"""
phiFEM_solver.compute_tags()

"""
Set up the phiFEM variational formulation
"""
_, _, _, num_dofs = phiFEM_solver.set_variational_formulation()

"""
Assemble the phiFEM linear system
"""
phiFEM_solver.assemble()

"""
Solve the phiFEM linear system
"""
phiFEM_solver.solve()
uh = phiFEM_solver.get_solution()

with XDMFFile(uh.function_space.mesh.comm, "solution.xdmf", "w") as of:
    of.write_mesh(uh.function_space.mesh)
    of.write_function(uh)

"""
Compute the residual a posteriori error estimator
"""
_ = phiFEM_solver.estimate_residual()

H10_residual = phiFEM_solver.get_H10_residual()
global_H10_estimator = np.sqrt(H10_residual.x.petsc_vec.array.sum())
with XDMFFile(H10_residual.function_space.mesh.comm, "local_estimators.xdmf", "w") as of:
    of.write_mesh(H10_residual.function_space.mesh)
    of.write_function(H10_residual)

"""
Compute a higher order approximation of the error from a reference solution
"""
global_H10_error, _, _, _ = compute_exact_error(uh,
                                                source_term,
                                                output_path="./",
                                                reference_mesh_path="./reference_mesh.xdmf")

"""
Compute residual estimator efficiency
"""
H10_efficiency = global_H10_estimator/global_H10_error

df = pd.DataFrame({"Num. dofs":      [num_dofs],
                   "H10 error":      [global_H10_error],
                   "H10 estimator":  [global_H10_estimator],
                   "H10 efficiency": [H10_efficiency]})

print("==============================")
print("           Results")
print("==============================")
print(df)