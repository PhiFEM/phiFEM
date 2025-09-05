from   basix.ufl import element
import jax.numpy as jnp
import numpy as np
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import pandas as pd
from   petsc4py.PETSc import Options, KSP

from phiFEM.phifem.mesh_scripts import compute_tags_measures

print("===============================")
print("  Product of sines test case")
print("===============================")

# Set up parameters
mesh_size                 = 0.01                    # Initial mesh size
bbox                      = np.array([[-1.5, 1.5],
                                      [-1.5, 1.5]]) # Domain bounding box mesh
marking_parameter         = 0.3                     # Dörfler marking parameter
finite_element_degree     = 1                       # FE degree for w_h
levelset_degree           = 1                       # FE degree for φ_h
boundary_detection_degree = 3                       # Degree of the boundary detection
boundary_refinement_type  = 'h'                     # Refinement type used in the hierarchical approximation of the boundary error
use_fine_space            = False                   # If true, uses the proper FE degree for the product w_h*φ_h, if False use the degree of w_h

tilt_angle = np.pi/6.
def rotation(angle, x):
    if x.shape[0] == 3:
        R = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                       [jnp.sin(angle),  jnp.cos(angle), 0],
                       [             0,               0, 1]])
    elif x.shape[0] == 2:
        R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                       [jnp.sin(angle),  jnp.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(jnp.asarray(x))

def tilted_square(x):
    def fct(x):
        return jnp.sum(jnp.abs(rotation(-tilt_angle + jnp.pi/4., x)), axis=0)
    return fct(x) - jnp.sqrt(2.)/2.

# Levelset as a smooth product of sines function
def levelset(x):
    vect = np.full_like(x, 0.5)
    val = -np.sin(np.pi * (rotation(-tilt_angle, x - rotation(tilt_angle, vect))[0, :])) * \
           np.sin(np.pi * (rotation(-tilt_angle, x - rotation(tilt_angle, vect))[1, :]))

    # val_ext = tilted_square(x)
    # val[val_ext > 0.] = val_ext[val_ext > 0.]
    return val

def detection_levelset(x):
    return tilted_square(x)

def exact_solution(x):
    return jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
           jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[1, :])

# Not required since jax will compute the negative laplacian of u_exact automatically but we add it since we know the analytical expression :)
def source_term(x):
    return 8. * jnp.pi**2 * jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
                            jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[1, :])

# Create the initial background mesh
nx = int(np.abs(bbox[0, 1] - bbox[0, 0]) * np.sqrt(2.) / mesh_size)
ny = int(np.abs(bbox[1, 1] - bbox[1, 0]) * np.sqrt(2.) / mesh_size)
bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox.T, [nx, ny])

# Defines finite elements
fe_element       = element("Lagrange", bg_mesh.topology.cell_name(), finite_element_degree)
levelset_element = element("Lagrange", bg_mesh.topology.cell_name(), levelset_degree)

# Set up the PETSc solver
options = Options()
options["ksp_type"]      = "cg"
options["pc_type"]       = "hypre"
options["ksp_rtol"]      = 1e-7
options["pc_hypre_type"] = "boomeramg"
petsc_solver = KSP().create(bg_mesh.comm)
petsc_solver.setFromOptions()



# Initialize phiFEM solver
phiFEM_solver = PhiFEMSolver(bg_mesh,
                             fe_element,
                             petsc_solver,
                             levelset_element=levelset_element,
                             detection_degree=boundary_detection_degree,
                             use_fine_space=use_fine_space,
                             boundary_refinement_type=boundary_refinement_type)

phiFEM_solver.set_source_term(source_term)
phiFEM_solver.set_levelset(levelset)

# Locate {φ_h = 0} and tag the cells
phiFEM_solver.compute_tags()

# Set up the variational formulation
_, _, _, num_dofs = phiFEM_solver.set_variational_formulation()

# Assemble the linear system
phiFEM_solver.assemble()

# Solve the phiFEM linear system
phiFEM_solver.solve()
uh = phiFEM_solver.get_solution()

with XDMFFile(uh.function_space.mesh.comm, "solution.xdmf", "w") as of:
    of.write_mesh(uh.function_space.mesh)
    of.write_function(uh)

# Compute the residual a posteriori error estimator
_ = phiFEM_solver.estimate_residual()

H10_residual = phiFEM_solver.get_H10_residual()
global_H10_estimator = np.sqrt(H10_residual.x.petsc_vec.array.sum())
with XDMFFile(H10_residual.function_space.mesh.comm, "local_estimators.xdmf", "w") as of:
    of.write_mesh(H10_residual.function_space.mesh)
    of.write_function(H10_residual)

# Compute the exact error
global_H10_error, _, _, _ = compute_exact_error(uh,
                                                source_term,
                                                output_path="./",
                                                expression_u_exact=expression_u_exact,
                                                reference_mesh_path="./reference_mesh.xdmf")

# Compute residual estimator efficiency
H10_efficiency = global_H10_estimator/global_H10_error

df = pd.DataFrame({"Num. dofs":      [num_dofs],
                   "H10 error":      [global_H10_error],
                   "H10 estimator":  [global_H10_estimator],
                   "H10 efficiency": [H10_efficiency]})

print("===============================")
print("           Results")
print("===============================")
print(df)