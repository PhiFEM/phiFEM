from   basix.ufl import element, _ElementBase
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.mesh import Mesh
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.fem import Form, Function, FunctionSpace, DirichletBC
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
import os
from   os import PathLike
from   petsc4py.PETSc import Options as PETSc_Options # type: ignore[attr-defined]
from   petsc4py.PETSc import KSP     as PETSc_KSP # type: ignore[attr-defined]
from   petsc4py.PETSc import Mat     as PETSc_Mat # type: ignore[attr-defined]
from   petsc4py.PETSc import Vec     as PETSc_Vec # type: ignore[attr-defined]
from   typing import Any, cast, Tuple
import ufl # type: ignore[import-untyped]
from   ufl import inner, jump, grad, div, avg

from phiFEM.phifem.continuous_functions import ContinuousFunction, ExactSolution
from phiFEM.phifem.saver import ResultsSaver

PathStr = PathLike[str] | str
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class FEMSolver:
    """ Class representing a FEM solver."""

    def __init__(self,
                 mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: PETSc_KSP,
                 num_step: int = 0,
                 ref_strat: str = "uniform",
                 save_output: bool = True) -> None:
        """ Initialize a solver.

        Args:
            mesh: the initial mesh on which the PDE is solved.
            FE_element: the finite element used to approximate the PDE solution.
            PETSc_solver: the PETSc solver used to solve the finite element linear system.
            ref_strat: the refinement strategy ('uniform' for uniform refinement, 'H10' for adaptive refinement based on the H10 residual estimator, 'L2' for adaptive refinement based on the L2 residual estimator).
            num_step: refinement iteration number.
            save_output: if True, save the functions, meshes and values to the disk.
        """
        self.A: PETSc_Mat | None                = None
        self.b: PETSc_Vec | None                = None
        self.bcs: list[DirichletBC] | list[int] = []
        self.bilinear_form: Form | None         = None
        self.eta_h_H10: Function | None         = None
        self.eta_h_L2: Function | None          = None
        self.err_H10: Function | None           = None
        self.err_L2: Function | None            = None
        self.FE_space: FunctionSpace            = dfx.fem.functionspace(mesh, FE_element)
        self.i: int                             = num_step
        self.linear_form: Form | None           = None
        self.mesh: Mesh                         = mesh
        self.petsc_solver: PETSc_KSP            = PETSc_solver
        self.ref_strat: str                     = ref_strat
        self.rhs: ContinuousFunction | None     = None
        self.save_output: bool                  = save_output
        self.solution: Function | None          = None
    
    def set_source_term(self, source_term: ContinuousFunction) -> None:
        """ Set the source term data.

        Args:
            source_term: the right-hand side data.
        """
        if source_term.expression is None:
            raise ValueError("The source term has no expression.")
        self.rhs = source_term

    def assemble(self) -> None:
        """ Assemble the linear system."""
        self.print("Assemble linear system.")

        if self.bilinear_form is None:
            raise ValueError("SOLVER_NAME.bilinear_form is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
        if self.linear_form is None:
            raise ValueError("SOLVER_NAME.linear_form is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")

        self.A = assemble_matrix(self.bilinear_form, bcs=self.bcs)
        self.A.assemble()
        self.b = assemble_vector(self.linear_form)
        dfx.fem.apply_lifting(self.b, [self.bilinear_form], [self.bcs])
        dfx.fem.set_bc(self.b, self.bcs)
    
    def print(self, str2print: str) -> None:
        """ Print the state of the solver."""
        if self.save_output:
            FE_degree = self.FE_space.element.basix_element.degree
            print(f"Solver: FEM. Refinement: {self.ref_strat}. FE degree: {FE_degree}. Iteration n° {str(self.i).zfill(2)}. {str2print}")
    
    def solve(self) -> None:
        """ Solve the FE linear system."""
        self.print("Solve linear system.")

        if self.FE_space is None:
            raise ValueError("SOLVER_NAME.FE_space is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
        if self.A is None:
            raise ValueError("SOLVER_NAME.A is None, did you forget to assemble ? (SOLVER_NAME.assemble)")
        if self.b is None:
            raise ValueError("SOLVER_NAME.b is None, did you forget to assemble ? (SOLVER_NAME.assemble)")
    
        self.solution = dfx.fem.Function(self.FE_space)
        self.petsc_solver.setOperators(self.A)
        self.petsc_solver.solve(self.b, self.solution.x.petsc_vec)
    
    def compute_exact_error(self,
                            results_saver: ResultsSaver,
                            expression_u_exact: NDArrayFunction | None = None,
                            save_output: bool = True,
                            extra_ref: int = 1,
                            ref_degree: int = 2,
                            interpolation_padding: float = 1.e-14,
                            reference_mesh_path: PathStr | None = None,
                            save_exact_solution: bool = False) -> None:
        """ Compute reference approximations to the exact errors in H10 and L2 norms.

        Args:
            results_saver:         the saver.
            expression_u_exact:    the expression of the exact solution (if None, a reference solution is computed on a finer reference mesh).
            save_output:           if True, save the functions, meshes and values to the disk.
            extra_ref:             the number of extra uniform refinements to get the reference mesh.
            ref_degree:            the degree of the finite element used to compute the approximations to the exact errors.
            interpolation_padding: padding for non-matching mesh interpolation.
            reference_mesh_path:   the path to the reference mesh.
        """
        self.print("Compute exact errors.")

        output_dir = results_saver.output_path

        FEM_dir_list = [subdir if subdir!="output_phiFEM" else "output_FEM" for subdir in cast(str, output_dir).split(sep=os.sep)]
        FEM_dir = os.path.join("/", *(FEM_dir_list))

        if reference_mesh_path is None:
            with XDMFFile(MPI.COMM_WORLD, os.path.join(FEM_dir, "meshes", f"mesh_{str(self.i).zfill(2)}.xdmf"), "r") as fi:
                try:
                    reference_mesh = fi.read_mesh()
                except RuntimeError:
                    print(f"Conforming mesh n°{str(self.i).zfill(2)} not found. In order to compute the exact errors, you must have run the FEM refinement loop first.")
        else:
            with XDMFFile(MPI.COMM_WORLD, reference_mesh_path, "r") as fi:
                reference_mesh = fi.read_mesh()
        
        # Computes the hmin in order to compare with reference mesh
        if self.solution is None:
            raise ValueError("SOLVER_NAME.solution is None, did you forget to solve ? (SOLVER_NAME.solve)")

        current_mesh = self.solution.function_space.mesh
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
            reference_mesh = dfx.mesh.refine(reference_mesh)
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

            FEM_solver = FEMSolver(reference_mesh, CGfElement, petsc_solver, num_step=self.i)
        
            if FEM_solver.FE_space is None:
                raise ValueError("SOLVER_NAME.FE_space is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
            
            if self.rhs is None:
                raise ValueError("SOLVER_NAME.rhs is None, did you forget to set the source term ? (SOLVER_NAME.set_source_term)")
            
            FEM_solver.set_source_term(self.rhs)
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
            u_exact_ref = FEM_solver.solution
        else:
            u_exact = ExactSolution(expression_u_exact)
            u_exact_ref = u_exact.interpolate(reference_space)

        if save_exact_solution:
            assert u_exact_ref is not None, "u_exact_ref is None."
            if ref_degree > 1:
                CG1Element = element("Lagrange", reference_mesh.topology.cell_name(), 1)
                reference_V = dfx.fem.functionspace(reference_mesh, CG1Element)
                u_exact_ref_V = dfx.fem.Function(reference_V)
                u_exact_ref_V.interpolate(u_exact_ref)
                results_saver.save_function(u_exact_ref_V, f"u_exact_{str(self.i).zfill(2)}")
            else:
                results_saver.save_function(u_exact_ref, f"u_exact_{str(self.i).zfill(2)}")

        uh_ref = dfx.fem.Function(reference_space)
        nmm = dfx.fem.create_interpolation_data(
                            uh_ref.function_space,
                            self.solution.function_space,
                            np.arange(num_cells),
                            padding=interpolation_padding)
        uh_ref.interpolate_nonmatching(self.solution,
                                       np.arange(num_cells),
                                       interpolation_data=nmm)
        e_ref = dfx.fem.Function(reference_space)

        if u_exact_ref is None:
            raise TypeError("u_exact_ref is None.")
        if uh_ref is None:
            raise TypeError("uh_ref is None.")
        e_ref.x.array[:] = u_exact_ref.x.array - uh_ref.x.array

        dx2 = ufl.Measure("dx",
                          domain=reference_mesh,
                          metadata={"quadrature_degree": 2 * (ref_degree + 1)})

        DG0Element = element("DG", reference_mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(reference_mesh, DG0Element)
        w0 = ufl.TestFunction(V0)

        L2_norm_local = inner(inner(e_ref, e_ref), w0) * dx2
        H10_norm_local = inner(inner(grad(e_ref), grad(e_ref)), w0) * dx2

        L2_error_0 = dfx.fem.Function(V0)
        H10_error_0 = dfx.fem.Function(V0)

        L2_norm_local_form = dfx.fem.form(L2_norm_local)
        L2_norm_local_form_assembled = assemble_vector(L2_norm_local_form)
        self.err_L2 = L2_norm_local_form_assembled
        L2_error_0.x.array[:] = L2_norm_local_form_assembled.array
        L2_error_global = np.sqrt(sum(L2_norm_local_form_assembled.array))

        # L2_norm_global_form = dfx.fem.form(inner(e_ref, e_ref) * dx2)
        # L2_error_global = np.sqrt(dfx.fem.assemble_scalar(L2_norm_global_form))

        results_saver.add_new_value("L2 error", L2_error_global)

        H10_norm_local_form = dfx.fem.form(H10_norm_local)
        H10_norm_local_form_assembled = assemble_vector(H10_norm_local_form)
        self.err_H10 = H10_norm_local_form_assembled
        H10_error_0.x.array[:] = H10_norm_local_form_assembled.array
        H10_error_global = np.sqrt(sum(H10_norm_local_form_assembled.array))

        # H10_norm_global_form = dfx.fem.form(inner(grad(e_ref), grad(e_ref)) * dx2)
        # H10_error_global = np.sqrt(dfx.fem.assemble_scalar(H10_norm_global_form))

        results_saver.add_new_value("H10 error", H10_error_global)

        # We reinterpolate the local exact errors back to the current mesh for an easier comparison with the estimators
        DG0Element_current_mesh = element("DG", current_mesh.topology.cell_name(), 0)
        V0_current_mesh = dfx.fem.functionspace(current_mesh, DG0Element_current_mesh)
        L2_error_0_current_mesh = dfx.fem.Function(V0_current_mesh)

        current_mesh_cells = np.arange(current_mesh.topology.index_map(tdim).size_global)
        nmm = dfx.fem.create_interpolation_data(L2_error_0_current_mesh.function_space,
                                                L2_error_0_current_mesh.function_space,
                                                current_mesh_cells,
                                                padding=interpolation_padding)
        L2_error_0_current_mesh.interpolate_nonmatching(L2_error_0,
                                                        current_mesh_cells,
                                                        interpolation_data=nmm)

        H10_error_0_current_mesh = dfx.fem.Function(V0_current_mesh)
        nmm = dfx.fem.create_interpolation_data(H10_error_0_current_mesh.function_space,
                                                H10_error_0_current_mesh.function_space,
                                                current_mesh_cells,
                                                padding=interpolation_padding)
        H10_error_0_current_mesh.interpolate_nonmatching(H10_error_0,
                                                         current_mesh_cells,
                                                         interpolation_data=nmm)

        if save_output:
            results_saver.save_function(L2_error_0_current_mesh,  f"L2_error_{str(self.i).zfill(2)}")
            results_saver.save_function(H10_error_0_current_mesh, f"H10_error_{str(self.i).zfill(2)}")
    
    def compute_efficiency_coef(self,
                                results_saver: ResultsSaver,
                                norm: str ="H10") -> None:
        assert norm in ["H10", "L2"], "The norm must be 'H10' or 'L2'."

        if norm=="H10":
            eta_h = self.eta_h_H10
            err   = self.err_H10
        elif norm=="L2":
            eta_h = self.eta_h_L2
            err   = self.err_L2
        
        if (eta_h is not None) and (err is not None):
            eta_global = np.sqrt(sum(eta_h.x.array))
            err_global = np.sqrt(sum(err.array))
            eff_coef = eta_global/err_global
            results_saver.add_new_value(f"{norm} efficiency", eff_coef)
        else:
            raise ValueError(f"The {norm} estimator or exact error is missing, did you forget to compute them ? (SOLVER_NAME.estimate_residual or SOLVER_NAME.compute_exact_errors)")

    def marking(self, theta: float = 0.3) -> npt.NDArray[np.float64]:
        """ Perform maximum marking strategy.

        Args:
            theta: the marking parameter (select the cells with the 100*theta% highest estimator values).
        """
        self.print("Mark mesh.")

        if self.ref_strat=="H10":
            if self.eta_h_H10 is None:
                raise ValueError("SOLVER_NAME.eta_h_H10 is None, did you forget to compute the residual estimator (SOLVER_NAME.estimate_residual)")
            eta_h = self.eta_h_H10
        elif self.ref_strat=="L2":
            if self.eta_h_L2 is None:
                raise ValueError("SOLVER_NAME.eta_h_L2 is None, did you forget to compute the residual estimator (SOLVER_NAME.estimate_residual)")
            eta_h = self.eta_h_L2
        else:
            raise ValueError("Marking has been called but the refinement strategy ref_strat is 'uniform' (must be 'H10' or 'L2').")

        mesh = eta_h.function_space.mesh
        cdim = mesh.topology.dim
        fdim = cdim - 1
        assert(mesh.comm.size == 1)

        eta_global = sum(eta_h.x.array)
        cutoff = theta * eta_global

        sorted_cells = np.argsort(eta_h.x.array)[::-1]
        rolling_sum = 0.0
        for j, e in enumerate(eta_h.x.array[sorted_cells]):
            rolling_sum += e
            if rolling_sum > cutoff:
                breakpoint = j
                break

        refine_cells = sorted_cells[0:breakpoint + 1]
        indices = np.array(np.sort(refine_cells), dtype=np.int32)
        c2f_connect = mesh.topology.connectivity(cdim, fdim)
        num_facets_per_cell = len(c2f_connect.links(0))
        c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        facets_indices: npt.NDArray[np.float64] = np.unique(np.sort(c2f_map[indices]))
        return facets_indices
    
    def get_solution(self) -> Function:
        if self.solution is None:
            raise ValueError("SOLVER_NAME.solution is None, did you forget to solve ? (SOLVER_NAME.solve)")
        return self.solution
    
    def get_H10_residual(self) -> Function:
        if self.eta_h_H10 is None:
            raise ValueError("SOLVER_NAME.eta_h_H10 is None, did you forget to compute the residual estimators ? (SOLVER_NAME.estimate_residual)")
        return self.eta_h_H10
    
    def get_L2_residual(self) -> Function:
        if self.eta_h_L2 is None:
            raise ValueError("SOLVER_NAME.eta_h_L2 is None, did you forget to compute the residual estimators ? (SOLVER_NAME.estimate_residual)")
        return self.eta_h_L2

    def set_boundary_conditions(self,
                                bcs: list[DirichletBC]) -> None:
        """ Set the boundary conditions."""
        self.bcs = bcs
    
    def set_variational_formulation(self, quadrature_degree: int | None = None) -> int:
        """ Defines the variational formulation.

        Args:
            quadrature_degree: (optional) int, the degree of quadrature.
        
        Returns:
            num_dofs: the number of degrees of freedom used in the FEM approximation.
        """
        self.print("Set variational formulation.")

        if quadrature_degree is None:
            quadrature_degree = 2 * (self.FE_space.element.basix_element.degree + 1)
        
        if self.rhs is None:
            raise ValueError("SOLVER_NAME.rhs is None, did you forget to set the source term ? (SOLVER_NAME.set_source_term)")

        f_h = self.rhs.interpolate(self.FE_space)

        num_dofs = len(f_h.x.array[:])
        u = ufl.TrialFunction(self.FE_space)
        v = ufl.TestFunction(self.FE_space)

        dx = ufl.Measure("dx",
                         domain=self.mesh,
                         metadata={"quadrature_degree": quadrature_degree})
        
        """
        Bilinear form
        """
        a = inner(grad(u), grad(v)) * dx

        """
        Linear form
        """
        L = inner(f_h, v) * dx

        self.bilinear_form = dfx.fem.form(a)
        self.linear_form = dfx.fem.form(L)
        return num_dofs

    def estimate_residual(self,
                          V0: FunctionSpace | None = None,
                          quadrature_degree: int | None = None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """ Compute the local and global contributions of the residual a posteriori error estimators for the H10 and L2 norms.

        Args:
            V0: (optional) dolfinx.fem.FunctionSpace, the function space in which the local contributions of the residual estimators are interpolated.
            quadrature_degree: (optional) int, the quadrature degree used in the integrals of the residual estimator.
        
        Returns:
            h10_residuals: dictionnary containing all the H1 semi-norm residuals.
            l2_residuals: dictionnary containing all the L2 norm residuals.
        """
        self.print("Compute estimators.")

        if quadrature_degree is None:
            if self.solution is None:
                raise ValueError("SOLVER_NAME.solution is None, did you forget to solve ? (SOLVER_NAME.solve)")
            k = self.solution.function_space.element.basix_element.degree
            quadrature_degree_cells  = max(0, k - 2)
            quadrature_degree_facets = max(0, k - 1)
        
        dx = ufl.Measure("dx",
                         domain=self.mesh,
                         metadata={"quadrature_degree": quadrature_degree_cells})
        dS = ufl.Measure("dS",
                         domain=self.mesh,
                         metadata={"quadrature_degree": quadrature_degree_facets})

        n   = ufl.FacetNormal(self.mesh)
        h_T = ufl.CellDiameter(self.mesh)
        h_E = ufl.FacetArea(self.mesh)

        if self.rhs is None:
            raise ValueError("SOLVER_NAME.rhs is None, did you forget to set the source term ? (SOLVER_NAME.set_source_term)")

        f_h = self.rhs.interpolate(self.FE_space)

        r = f_h + div(grad(self.solution))
        J_h = jump(grad(self.solution), -n)

        if V0 is None:
            DG0Element = element("DG", self.mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(self.mesh, DG0Element)

        v0 = ufl.TestFunction(V0)

        """
        H10 estimator
        """
        # Interior residual
        eta_T = h_T**2 * inner(inner(r, r), v0) * dx

        # Facets residual
        eta_E = avg(h_E) * inner(inner(J_h, J_h), avg(v0)) * dS

        eta = eta_T + eta_E

        eta_form = dfx.fem.form(eta)

        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.x.petsc_vec.setArray(eta_vec.array[:])
        self.eta_h_H10 = eta_h

        h10_residuals = {"Interior residual":       eta_T,
                         "Internal edges residual": eta_E,
                         "Geometry residual":       None,
                         "Boundary edges residual": None}

        """
        L2 estimator
        """
        eta_T = h_T**4 * inner(inner(r, r), v0) * dx
        eta_E = avg(h_E)**3 * inner(inner(J_h, J_h), avg(v0)) * dS

        eta = eta_T + eta_E
        eta_form = dfx.fem.form(eta)

        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.x.petsc_vec.setArray(eta_vec.array[:])
        self.eta_h_L2 = eta_h

        l2_residuals = {"Interior residual":       eta_T,
                        "Internal edges residual": eta_E,
                        "Geometry residual":       None,
                        "Boundary edges residual": None}
        return h10_residuals, l2_residuals