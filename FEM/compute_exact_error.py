from   ufl import Measure
from   basix.ufl import element
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.fem import Function
from   dolfinx.fem.petsc import assemble_vector
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
import os
from   os import PathLike
from   petsc4py.PETSc import Options as PETSc_Options # type: ignore[attr-defined]
from   petsc4py.PETSc import KSP as PETSc_KSP # type: ignore[attr-defined]
from   typing import Tuple, cast
import ufl
from   ufl import inner, grad

from FEM.solver import FEMSolver
from phiFEM.phifem.continuous_functions import ExactSolution

PathStr = PathLike[str] | str
NDArrayTuple = Tuple[npt.NDArray[np.float64]]
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

def compute_exact_error(solution,
                        source_term,
                        phifem_measure: Measure | None = None,
                        output_path: PathStr = "./",
                        iteration_num: int = 0,
                        expression_u_exact: NDArrayFunction | None = None,
                        extra_ref: int = 1,
                        ref_degree: int = 2,
                        interpolation_padding: float = 1.e-14,
                        reference_mesh_path: PathStr | None = None) -> Tuple[Function]:
    """ Compute reference approximations to the exact errors in H10 and L2 norms.

    Args:
        solution:              the FE solution
        source_term:           the source term.
        output_path:           the output_path.
        iteration_num:         the iteration number.
        expression_u_exact:    the expression of the exact solution (if None, a reference solution is computed on a finer reference mesh).
        extra_ref:             the number of extra uniform refinements to get the reference mesh.
        ref_degree:            the degree of the finite element used to compute the approximations to the exact errors.
        interpolation_padding: padding for non-matching mesh interpolation.
        reference_mesh_path:   the path to the reference mesh.
    
    Returns:
        The H10 global error and the L2 global error.
    """
    print("Compute exact errors.")

    FEM_dir_list = [subdir if subdir!="output_phiFEM" else "output_FEM" for subdir in cast(str, output_path).split(sep=os.sep)]
    FEM_dir = os.path.join("/", *(FEM_dir_list))

    if reference_mesh_path is None:
        with XDMFFile(MPI.COMM_WORLD, os.path.join(FEM_dir, "meshes", f"mesh_{str(iteration_num).zfill(2)}.xdmf"), "r") as fi:
            try:
                reference_mesh = fi.read_mesh()
            except RuntimeError:
                print(f"Conforming mesh n°{str(iteration_num).zfill(2)} not found. In order to compute the exact errors, you must have run the FEM refinement loop first.")
    else:
        with XDMFFile(MPI.COMM_WORLD, reference_mesh_path, "r") as fi:
            reference_mesh = fi.read_mesh(name="Grid")

    # Computes the hmin in order to compare with reference mesh
    if solution is None:
        raise ValueError("SOLVER_NAME.solution is None, did you forget to solve ? (SOLVER_NAME.solve)")

    current_mesh = solution.function_space.mesh
    tdim = current_mesh.topology.dim
    num_cells = current_mesh.topology.index_map(tdim).size_global
    current_hmin = dfx.cpp.mesh.h(current_mesh._cpp_object, tdim, np.arange(num_cells)).min()

    for i in range(extra_ref):
        reference_mesh.topology.create_entities(reference_mesh.topology.dim - 1)
        reference_mesh, _, _ = dfx.mesh.refine(reference_mesh)

    # Computes hmin in order to ensure that the reference mesh is fine enough
    tdim = reference_mesh.topology.dim
    num_cells = reference_mesh.topology.index_map(tdim).size_global
    reference_hmin = dfx.cpp.mesh.h(reference_mesh._cpp_object, tdim, np.arange(num_cells)).min()
    while (reference_hmin > current_hmin):
        reference_mesh.topology.create_entities(reference_mesh.topology.dim - 1)
        reference_mesh, _, _ = dfx.mesh.refine(reference_mesh)
        # Computes hmin in order to ensure that the reference mesh is fine enough
        tdim = reference_mesh.topology.dim
        num_cells = reference_mesh.topology.index_map(tdim).size_global
        reference_hmin = dfx.cpp.mesh.h(reference_mesh._cpp_object, tdim, np.arange(num_cells)).min()

    CGfElement = element("Lagrange", reference_mesh.topology.cell_name(), ref_degree)
    reference_space = dfx.fem.functionspace(reference_mesh, CGfElement)

    if expression_u_exact is None:
        # Parametrization of the PETSc solver
        options = PETSc_Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = PETSc_KSP().create(reference_mesh.comm)
        petsc_solver.setFromOptions()

        FEM_solver = FEMSolver(reference_mesh, CGfElement, petsc_solver, num_step=iteration_num)
    
        if FEM_solver.FE_space is None:
            raise ValueError("SOLVER_NAME.FE_space is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
        
        if source_term is None:
            raise ValueError("SOLVER_NAME.rhs is None, did you forget to set the source term ? (SOLVER_NAME.set_source_term)")

        FEM_solver.set_source_term(source_term)
        dbc = dfx.fem.Function(FEM_solver.FE_space)
        facets = dfx.mesh.locate_entities_boundary(
                                reference_mesh,
                                1,
                                lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
        bcs = [dfx.fem.dirichletbc(dbc, dofs)]
        FEM_solver.set_boundary_conditions(bcs)
        _ = FEM_solver.set_variational_formulation()
        FEM_solver.assemble()
        FEM_solver.solve()
        u_ref = FEM_solver.solution
    else:
        u_exact = ExactSolution(expression_u_exact)
        u_ref = u_exact.interpolate(reference_space)

    uh_ref = dfx.fem.Function(reference_space)
    nmm = dfx.fem.create_interpolation_data(
                        reference_space,
                        solution.function_space,
                        np.arange(num_cells),
                        padding=interpolation_padding)
    uh_ref.interpolate_nonmatching(solution,
                                   np.arange(num_cells),
                                   interpolation_data=nmm)
    e_ref = dfx.fem.Function(reference_space)

    if u_ref is None:
        raise TypeError("u_ref is None.")
    if uh_ref is None:
        raise TypeError("uh_ref is None.")
    e_ref.x.array[:] = u_ref.x.array - uh_ref.x.array

    dx2 = ufl.Measure("dx",
                      domain=reference_mesh,
                      metadata={"quadrature_degree": 2 * (ref_degree + 1)})

    DG0Element = element("DG", reference_mesh.topology.cell_name(), 0)
    V0 = dfx.fem.functionspace(reference_mesh, DG0Element)
    w0 = ufl.TestFunction(V0)

    L2_norm_local  = inner(inner(e_ref, e_ref), w0) * dx2
    H10_norm_local = inner(inner(grad(e_ref), grad(e_ref)), w0) * dx2

    L2_error_0  = dfx.fem.Function(V0)
    H10_error_0 = dfx.fem.Function(V0)

    L2_norm_local_form = dfx.fem.form(L2_norm_local)
    L2_norm_local_form_assembled = assemble_vector(L2_norm_local_form)
    L2_error_0.x.array[:] = L2_norm_local_form_assembled.array
    global_L2_error = np.sqrt(L2_error_0.x.array.sum())

    H10_norm_local_form = dfx.fem.form(H10_norm_local)
    H10_norm_local_form_assembled = assemble_vector(H10_norm_local_form)
    H10_error_0.x.array[:] = H10_norm_local_form_assembled.array
    global_H10_error = np.sqrt(H10_error_0.x.array.sum())

    # We reinterpolate the local exact errors back to the current mesh for an easier comparison with the estimators
    # TODO: this is broken since dolfinx Commit ab79530

    # current_mesh_ref_element = element("Lagrange", current_mesh.topology.cell_name(), ref_degree)
    # current_mesh_V_ref = dfx.fem.functionspace(current_mesh, current_mesh_ref_element)
    # e_ref_current_mesh = dfx.fem.Function(current_mesh_V_ref)
    # num_cells = current_mesh.topology.index_map(tdim).size_global
    # nmm = dfx.fem.create_interpolation_data(solution.function_space,
    #                                         reference_space,
    #                                         np.arange(num_cells),
    #                                         padding=interpolation_padding)
    # e_ref_current_mesh.interpolate_nonmatching(e_ref,
    #                                            np.arange(num_cells),
    #                                            interpolation_data=nmm)
    
    # e_ref_reinterp = dfx.fem.Function(solution.function_space)
    # e_ref_reinterp.interpolate(e_ref_current_mesh)

    # with XDMFFile(current_mesh.comm, "./e_ref.xdmf", 'w') as of:
    #     of.write_mesh(current_mesh)
    #     of.write_function(e_ref_reinterp)

    # DG0Element_current_mesh = element("DG", current_mesh.topology.cell_name(), 0)
    # V0_current_mesh = dfx.fem.functionspace(current_mesh, DG0Element_current_mesh)

    # w0 = ufl.TestFunction(V0_current_mesh)

    # if phifem_measure is None:
    #     dx = ufl.Measure("dx",
    #                      domain=current_mesh,
    #                      metadata={"quadrature_degree": 2 * (ref_degree + 1)})
    # else:
    #     dx = phifem_measure(1) + phifem_measure(2)

    # L2_norm_local  = inner(inner(e_ref_current_mesh, e_ref_current_mesh), w0) * dx
    # H10_norm_local = inner(inner(grad(e_ref_current_mesh), grad(e_ref_current_mesh)), w0) * dx

    # L2_error_0_current_mesh  = dfx.fem.Function(V0_current_mesh)
    # H10_error_0_current_mesh = dfx.fem.Function(V0_current_mesh)

    # L2_norm_local_form = dfx.fem.form(L2_norm_local)
    # L2_norm_local_form_assembled = assemble_vector(L2_norm_local_form)
    # L2_error_0_current_mesh.x.array[:] = L2_norm_local_form_assembled.array

    # H10_norm_local_form = dfx.fem.form(H10_norm_local)
    # H10_norm_local_form_assembled = assemble_vector(H10_norm_local_form)
    # H10_error_0_current_mesh.x.array[:] = H10_norm_local_form_assembled.array

    return global_H10_error, global_L2_error, u_ref, e_ref