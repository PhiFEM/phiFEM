from   basix.ufl import element, _ElementBase
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.mesh import Mesh, MeshTags
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
from   ufl.classes import Measure # type: ignore[import-untyped]

from phiFEM.phifem.compute_meshtags import tag_cells, tag_facets
from phiFEM.phifem.continuous_functions import ContinuousFunction, ExactSolution, Levelset
from phiFEM.phifem.mesh_scripts import compute_outward_normal
from phiFEM.phifem.saver import ResultsSaver

PathStr = PathLike[str] | str
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class GenericSolver:
    """ Class representing a generic solver."""

    def __init__(self,
                 mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: PETSc_KSP,
                 ref_strat: str = "uniform",
                 num_step: int = 0,
                 box_mode: bool = False,
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
        self.A: PETSc_Mat | None            = None
        self.b: PETSc_Vec | None            = None
        self.bilinear_form: Form | None     = None
        self.box_mode: bool                 = box_mode
        self.eta_h_H10: Function | None     = None
        self.eta_h_L2: Function | None      = None
        self.err_H10: Function | None       = None
        self.err_L2: Function | None        = None
        self.FE_element: _ElementBase       = FE_element
        self.FE_space: FunctionSpace | None = None
        self.i: int                         = num_step
        self.linear_form: Form | None       = None
        self.mesh: Mesh                     = mesh
        self.petsc_solver: PETSc_KSP        = PETSc_solver
        self.ref_strat: str                 = ref_strat
        self.rhs: ContinuousFunction | None = None
        self.save_output: bool              = save_output
        self.solution_wh: Function | None   = None
        self.solution: Function | None      = None
        self.bcs: list[Any]                 = []
        self.solver_type: str               = "Generic"
    
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
            FE_degree = self.FE_element.basix_element.degree
            print(f"Solver: {self.solver_type}. Refinement: {self.ref_strat}. FE degree: {FE_degree}. Iteration n° {str(self.i).zfill(2)}. {str2print}")
    
    def solve(self) -> None:
        """ Solve the FE linear system."""
        self.print("Solve linear system.")

        if self.FE_space is None:
            raise ValueError("SOLVER_NAME.FE_space is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
        if self.A is None:
            raise ValueError("SOLVER_NAME.A is None, did you forget to assemble ? (SOLVER_NAME.assemble)")
        if self.b is None:
            raise ValueError("SOLVER_NAME.b is None, did you forget to assemble ? (SOLVER_NAME.assemble)")
    
        self.solution_wh = dfx.fem.Function(self.FE_space)
        self.petsc_solver.setOperators(self.A)
        self.petsc_solver.solve(self.b, self.solution_wh.x.petsc_vec)
    
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
    
    def get_eta_h_H10(self) -> Function:
        if self.eta_h_H10 is None:
            raise ValueError("SOLVER_NAME.eta_h_H10 is None, did you forget to compute the residual estimators ? (SOLVER_NAME.estimate_residual)")
        return self.eta_h_H10
    
    def get_eta_h_L2(self) -> Function:
        if self.eta_h_L2 is None:
            raise ValueError("SOLVER_NAME.eta_h_L2 is None, did you forget to compute the residual estimators ? (SOLVER_NAME.estimate_residual)")
        return self.eta_h_L2

# TODO: keep the connectivities as class objects to avoid unnecessary multiple computations.
class PhiFEMSolver(GenericSolver):
    """ Class representing a phiFEM solver as a GenericSolver object."""
    def __init__(self,
                 bg_mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: PETSc_KSP,
                 ref_strat: str = "uniform",
                 levelset_element: _ElementBase | None = None,
                 detection_degree: int = 1,
                 box_mode: bool = False,
                 boundary_refinement_type: str = 'h',
                 use_fine_space: bool = False,
                 num_step: int = 0,
                 save_output: bool = True) -> None:
        """ Initialize a phiFEM solver object.

        Args:
            bg_mesh: the background mesh.
            FE_element: the finite element used in the phiFEM discretization.
            PETSc_solver: the PETSc solver used to solve the finite element linear system.
            ref_strat: the refinement strategy ('uniform' for uniform refinement, 'H10' for adaptive refinement based on the H10 residual estimator, 'L2' for adaptive refinement based on the L2 residual estimator).
            levelset_element: the finite element used to discretize the levelset function.
            box_mode: if True, does not compute submeshes and only refines the initial background mesh.
            use_fine_space: use the proper fine space for the phiFEM solution (i.e. space of degree FE_degree + levelset_degree).
            num_step: the refinement step number.
            save_output: if True, save the functions, meshes and values to the disk.
        """
        super().__init__(bg_mesh,
                         FE_element,
                         PETSc_solver,
                         ref_strat=ref_strat,
                         num_step=num_step,
                         box_mode=box_mode,
                         save_output=save_output)

        self.bg_mesh_cells_tags: MeshTags | None  = None
        if boundary_refinement_type not in ['h', 'p']:
            raise ValueError("boundary_refinement_type must be 'h' or 'p'.")
        self.boundary_refinement_type: str        = boundary_refinement_type
        self.facets_tags: MeshTags | None         = None
        self.FE_space: FunctionSpace | None       = None
        self.levelset: Levelset | None            = None
        self.levelset_element: _ElementBase
        if levelset_element is None:
            self.levelset_element = FE_element
        else:
            self.levelset_element = levelset_element
        self.levelset_space: FunctionSpace | None = None
        self.solver_type: str                     = "phiFEM"
        self.submesh: Mesh | None                 = None
        self.submesh_cells_tags: MeshTags | None  = None
        self.use_fine_space: bool                 = use_fine_space
        self.v0: Function | None                  = None
        self.detection_degree: int                = detection_degree

    def _compute_normal(self, mesh: Mesh) -> Function:
        """ Private method used to compute the outward normal to Omega_h.

        Args:
            mesh: the mesh.
        
        Returns:
            The outward normal field as a dolfinx.fem.Function.
        """
        if self.levelset is None:
            raise ValueError("SOLVER_NAME.levelset is None, did you forget to set the levelset ? (SOLVER_NAME.set_levelset)")

        return compute_outward_normal(mesh, self.levelset)

    def _transfer_cells_tags(self, cmap: npt.NDArray[Any]) -> None:
        """ Private method used to transfer the cells tags from the background mesh to the submesh.

        Args:
            cmap: background cells to submesh cells indices map.
        """

        if self.bg_mesh_cells_tags is None:
            raise ValueError("SOLVER_NAME.bg_mesh_cells_tags is None, did you forget to compute the mesh tags ? (SOLVER_NAME.compute_tags)")
        if self.submesh is None:
            raise ValueError("SOLVER_NAME.submesh is None, did you forget to compute the mesh tags ? (SOLVER_NAME.compute_tags)")
        
        cdim = self.submesh.topology.dim
        # TODO: change this line to allow parallel computing
        bg_mesh_interior       = self.bg_mesh_cells_tags.find(1)
        bg_cells_omega_h_gamma = self.bg_mesh_cells_tags.find(2)
        bg_cells_padding       = self.bg_mesh_cells_tags.find(4)

        mask_interior      = np.in1d(cmap, bg_mesh_interior)
        mask_omega_h_gamma = np.in1d(cmap, bg_cells_omega_h_gamma)
        mask_padding       = np.in1d(cmap, bg_cells_padding)

        submesh_interior            = np.where(mask_interior)[0]
        submesh_cells_omega_h_gamma = np.where(mask_omega_h_gamma)[0]
        submesh_cells_padding       = np.where(mask_padding)[0]

        list_cells = [submesh_interior,
                      submesh_cells_omega_h_gamma,
                      submesh_cells_padding]
        list_markers = [np.full_like(submesh_interior, 1),
                        np.full_like(submesh_cells_omega_h_gamma, 2),
                        np.full_like(submesh_cells_padding, 4)]
        cells_indices = np.hstack(list_cells).astype(np.int32)
        cells_markers = np.hstack(list_markers).astype(np.int32)
        sorted_indices = np.argsort(cells_indices)

        self.submesh_cells_tags = dfx.mesh.meshtags(self.submesh,
                                                    cdim,
                                                    cells_indices[sorted_indices],
                                                    cells_markers[sorted_indices])

    def print(self, str2print: str) -> None:
        """ Print the state of the solver."""
        if self.save_output:
            FE_degree = self.FE_element.basix_element.degree

            levelset_degree = self.levelset_element.basix_element.degree
            print(f"Solver: {self.solver_type}. Refinement: {self.ref_strat}. FE degree: {FE_degree}. Levelset degree: {levelset_degree}. Use fine space: {str(self.use_fine_space)}. Iteration n° {str(self.i).zfill(2)}. {str2print}")

    def set_levelset(self, levelset: Levelset) -> None:
        """ Set the right-hand side source term (force term) and the levelset."""
        if levelset.expression is None:
            raise ValueError("The levelset has no expression.")
        self.levelset = levelset

    def compute_tags(self, plot: bool = False) -> None:
        """ Compute the mesh tags.

        Args:
            padding: if True, computes an extra padding layer of cells to increase the chances to keep the level 0 curve of the levelset inside the submesh.
            plot:    if True, plots the mesh tags (very slow).
        """
        self.print("Mesh tags computation.")

        working_mesh = self.mesh
        # Tag cells of the background mesh. Used to tag the facets and/or create the submesh.
        if self.levelset is None:
            raise ValueError("SOLVER_NAME.levelset is None, did you forget to set the levelset ? (SOLVER_NAME.set_levelset)")
        
        self.bg_mesh_cells_tags= tag_cells(self.mesh,
                                           self.levelset,
                                           self.detection_degree,
                                           plot=plot)

        working_cells_tags = self.bg_mesh_cells_tags

        # Create the submesh and transfer the cells tags from the bg mesh to the submesh.
        # Tag the facets of the submesh.
        if self.bg_mesh_cells_tags is None:
            raise TypeError("SOLVER_NAME.bg_mesh_cells_tags is None.")
        
        if self.box_mode:
            #self.submesh = dfx.mesh.Mesh(self.mesh, self.mesh.ufl_domain())
            self.submesh = self.mesh
            working_cells_tags = self.bg_mesh_cells_tags
            working_mesh = self.mesh
        else:
            omega_h_cells = np.unique(np.hstack([self.bg_mesh_cells_tags.find(1),
                                                 self.bg_mesh_cells_tags.find(2),
                                                 self.bg_mesh_cells_tags.find(4)]))
            self.submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(self.mesh,
                                                                        self.mesh.topology.dim,
                                                                        omega_h_cells) # type: ignore

            if self.submesh is None:
                raise TypeError("SOLVER_NAME.submesh is None.")

            self._transfer_cells_tags(c_map)
            if self.submesh_cells_tags is None:
                raise TypeError("SOLVER_NAME.submesh_cells_tags is None.")

            working_cells_tags = self.submesh_cells_tags
            working_mesh = self.submesh

        self.facets_tags = tag_facets(working_mesh,
                                      working_cells_tags,
                                      plot=plot)
    
    def set_variational_formulation(self,
                                    sigma: float = 1.,
                                    quadrature_degree: int | None = None) -> Tuple[Function | None, Measure, Measure, int]:
        """ Defines the variational formulation.

        Args:
            sigma: (optional) float, the phiFEM stabilization coefficient.
            quadrature_degree: (optional) int, the degree of quadrature.
        
        Returns:
            v0: dolfinx.fem.Function, the identity function of Omega_h.
            dx: ufl.Measure, the surface measure with the cells tags.
            dS: ufl.Measure, the facets measure with the facets tags.
            num_dofs: int, the number of degrees of freedom used in the phiFEM approximation.
        """
        self.print("Variational formulation set up.")

        if quadrature_degree is None:
            if self.levelset_element is None:
                raise TypeError("SOLVER_NAME.levelset_element is None.")
            quadrature_degree = 2 * (self.levelset_element.basix_element.degree + 1)

        if self.box_mode:
            if self.mesh is None:
                raise ValueError("SOLVER_NAME.mesh is None.")
            working_mesh = self.mesh
            if self.bg_mesh_cells_tags is None:
                raise ValueError("SOLVER_NAME.bg_mesh_cells_tags is None, did you forget to compute the mesh tags ? (SOLVER_NAME.compute_tags)")
            cells_tags = self.bg_mesh_cells_tags
        else:
            if self.submesh is None:
                raise ValueError("SOLVER_NAME.submesh is None.")
            working_mesh = self.submesh
            if self.submesh_cells_tags is None:
                raise ValueError("SOLVER_NAME.submesh_cells_tags is None, did you forget to compute the mesh tags ? (SOLVER_NAME.compute_tags)")
            cells_tags = self.submesh_cells_tags

        # TODO: modify to get it work in parallel
        interior_cells = cells_tags.find(1)
        cut_cells      = cells_tags.find(2)
        Omega_h_cells = np.union1d(interior_cells, cut_cells)
        self.FE_space = dfx.fem.functionspace(working_mesh, self.FE_element)
        cdim = working_mesh.topology.dim
        working_mesh.topology.create_connectivity(cdim, cdim)
        active_dofs = dfx.fem.locate_dofs_topological(self.FE_space, cdim, Omega_h_cells)
        num_dofs = len(active_dofs)
        self.levelset_space = dfx.fem.functionspace(working_mesh, self.levelset_element)

        if self.levelset is None:
            raise TypeError("SOLVER_NAME.levelset is None.")
        
        phi_h = self.levelset.interpolate(self.levelset_space)

        if self.rhs is None:
            raise ValueError("SOLVER_NAME.rhs is None, did you forget to set the source term ? (SOLVER_NAME.set_source_term)")

        f_h = self.rhs.interpolate(self.FE_space)

        h = ufl.CellDiameter(working_mesh)
        n = ufl.FacetNormal(working_mesh)

        w = ufl.TrialFunction(self.FE_space)
        v = ufl.TestFunction(self.FE_space)

        dx = ufl.Measure("dx",
                         domain=working_mesh,
                         subdomain_data=cells_tags,
                         metadata={"quadrature_degree": quadrature_degree})
        dS = ufl.Measure("dS",
                         domain=working_mesh,
                         subdomain_data=self.facets_tags,
                         metadata={"quadrature_degree": quadrature_degree})

        if self.box_mode:
            # In box_mode we need to compute the outward normal to Omega_h, this is the role of Omega_h_n.
            # In addition, we multiply by the indicator function v0 in order to remove the contributions of avg from outside Omega_h.
            Omega_h_n = self._compute_normal(working_mesh)
            
            # Set up a DG function with values 1. in Omega_h and 0. outside
            DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(working_mesh, DG0Element)
            v0 = dfx.fem.Function(V0)
            v0.x.set(0.)

            # We do not need the dofs here since cells and DG0 dofs share the same indices in dolfinx
            v0.x.array[Omega_h_cells] = 1.
            self.v0 = v0

            dBoundary = dS(4)
            boundary = inner(2. * avg(inner(grad(phi_h * w), Omega_h_n) * v0),
                             2. * avg(phi_h * v * v0))
        else:
            # When we are not in box_mode, the boundary term is simpler since we have access to the normal n and there is no need to use avg.
            dBoundary = ufl.Measure("ds",
                                    domain=working_mesh,
                                    metadata={"quadrature_degree": quadrature_degree})
            boundary = inner(inner(grad(phi_h * w), n), phi_h * v)

        """
        Bilinear form
        """
        stiffness = inner(grad(phi_h * w), grad(phi_h * v))

        # The stabilization terms
        stabilization_facets = sigma * avg(h) * inner(jump(grad(phi_h * w), n),
                                                     jump(grad(phi_h * v), n))
        stabilization_cells = sigma * h**2 * inner(div(grad(phi_h * w)),
                                                  div(grad(phi_h * v)))

        # The φ-FEM bilinear form
        a = stiffness             * (dx(1) + dx(2)) \
            - boundary            * dBoundary \
            + stabilization_facets * dS(2) \
            + stabilization_cells  * dx(2)
        
        if self.box_mode:
            # Useless term so that PETSc doesn't throw a Segfault at my face
            a += inner(w, v) * v0 * (dx(3) + dx(4))

            # In box_mode we turn all the exterior dofs to zero by setting a dummy zero Dirichlet BC on them
            if self.facets_tags is None:
                raise ValueError("SOLVER_NAME.facets_tags is None, did you forget to compute the mesh tags ? (SOLVER_NAME.compute_tags())")
            
            boundary_facets        = self.facets_tags.find(4)
            strict_exterior_facets = self.facets_tags.find(3)
            dofs_exterior = dfx.fem.locate_dofs_topological(self.FE_space, cdim - 1, strict_exterior_facets)
            dofs_boundary = dfx.fem.locate_dofs_topological(self.FE_space, cdim - 1, boundary_facets)
            inactive_dofs = np.setdiff1d(dofs_exterior, dofs_boundary)

            dbc_values = dfx.fem.Function(self.FE_space)
            dbc = dfx.fem.dirichletbc(dbc_values, inactive_dofs)
            self.bcs.append(dbc)

        if self.facets_tags is None:
            raise ValueError("SOLVER_NAME.facets_tags is None, did you forget to compute the mesh tags ? (SOLVER_NAME.compute_tags)")

        """
        Linear form
        """
        rhs = inner(f_h, phi_h * v)
        stabilization_rhs = sigma * h**2 * inner(f_h, div(grad(phi_h * v)))

        L = rhs                 * (dx(1) + dx(2))\
            - stabilization_rhs  * dx(2)

        self.bilinear_form = dfx.fem.form(a)
        self.linear_form = dfx.fem.form(L)
        return self.v0, dx, dS, num_dofs
    
    def solve(self) -> None:
        """ Solve the phiFEM linear system."""
        super().solve()
        # TODO: to be fixed when levelset_space != FE_space
        if self.levelset is None:
            raise TypeError("SOLVER_NAME.levelset is None.")
        if self.levelset_space is None:
            raise TypeError("SOLVER_NAME.levelset_space is None.")
        if self.solution_wh is None:
            raise TypeError("SOLVER_NAME.solution_wh is None.")

        if self.use_fine_space:
            assert self.FE_space is not None, "SOLVER_NAME.FE_space is None."
            FE_degree       = self.FE_space.element.basix_element.degree
            levelset_degree = self.levelset_space.element.basix_element.degree
            assert self.submesh is not None, "SOLVER_NAME.submesh is None."
            SolutionElement = element("Lagrange", self.submesh.topology.cell_name(), FE_degree + levelset_degree)
            SolutionSpace = dfx.fem.functionspace(self.submesh, SolutionElement)
        else:
            SolutionSpace = self.levelset_space

        phi_h = self.levelset.interpolate(SolutionSpace)
        wh = dfx.fem.Function(SolutionSpace)
        wh.interpolate(self.solution_wh)
        self.solution = dfx.fem.Function(SolutionSpace)
        self.solution.x.array[:] = wh.x.array[:] * phi_h.x.array[:]

    def get_solution_wh(self) -> Function:
        if self.solution_wh is None:
            raise ValueError("SOLVER_NAME.solution_wh is None, did you forget to solve ? (SOLVER_NAME.solve)")
        return self.solution_wh

    def get_levelset_space(self) -> FunctionSpace:
        if self.levelset_space is None:
            raise ValueError("SOLVER_NAME.levelset_space is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
        return self.levelset_space
    
    def _compute_boundary_correction_function(self,
                                              working_mesh: Mesh,
                                              entities_tags: MeshTags,
                                              refinement_type: str) -> Function:
        """ Compute the boundary correction function.

        Args:
            working_mesh: the current mesh.
            entities_tags: the cells tags if refinement_type=='p', the facets tags if refinement_type=='h'.
            refinement_type: 'p' for p-refinement boundary correction, 'h' for h-refinement boundary correction.
        
        Returns: the correction function.
        """
        if refinement_type not in ['p', 'h']:
            raise ValueError("refinement_type must be 'p' or 'h'.")
        
        phih = self.levelset.interpolate(self.levelset_space)
        if refinement_type=='p':
            """
            p-refinement boundary correction
            correction_function = (φ_h - φ_f) w_h
            where:
            - φ_h is the discretization of the levelset in the levelset space.
            - φ_f is the discretization of the levelset in a p-finer space (lagrange of degree levelset_degree + 1).
            """
            if entities_tags.dim != working_mesh.topology.dim:
                raise ValueError("In 'p' refinement, the entities_tags must be of same dim as the mesh (cells).")

            levelset_degree = self.levelset_space.element.basix_element.degree
            CGfElement = element("Lagrange", working_mesh.topology.cell_name(), levelset_degree + 1)
            V_correction = dfx.fem.functionspace(working_mesh, CGfElement)

            # Get the dofs except those on the cut cells
            cut_cells = entities_tags.find(2)
            cut_cells_dofs = dfx.fem.locate_dofs_topological(V_correction, 2, cut_cells)
            num_dofs_global = V_correction.dofmap.index_map.size_global * V_correction.dofmap.index_map_bs
            all_dofs = np.arange(num_dofs_global)
            uncut_cells_dofs = np.setdiff1d(all_dofs, cut_cells_dofs)

            phih_correction = dfx.fem.Function(V_correction)
            phih_correction.interpolate(phih)

            phi_correction = dfx.fem.Function(V_correction)
            phi_correction.interpolate(self.levelset)

            wh_correction = dfx.fem.Function(V_correction)
            if self.solution_wh is None:
                raise ValueError("SOLVER_NAME.solution_wh is None, did you forget to solve ?(SOLVER_NAME.solve)")

            wh_correction.interpolate(self.solution_wh)

            correction_function_V = dfx.fem.Function(V_correction)
            correction_function_V.x.array[:] = (phih_correction.x.array[:] - phi_correction.x.array[:]) * wh_correction.x.array[:]
            correction_function_V.x.array[uncut_cells_dofs] = 0.
        elif refinement_type=='h':
            """
            h-refinement boundary correction.
            correction_function = (φ_h - φ_f) w_f
            where:
            - φ_h is the discretization of the levelset in the levelset space.
            - φ_f is the interpolation of φ in the h-finer space (based on a mesh locally refined around Ω_h^Γ).
            All the functions have to be interpolated in the same space (the correction space) prior the computation of the correction function.
            Then all the functions are interpolated back to the working_mesh in a higher order space (to keep the features from the finer mesh).
            """
            if entities_tags.dim != working_mesh.topology.dim - 1:
                raise ValueError("In 'h' refinement, the entities_tags must be equal to mesh.topology.dim - 1 (facets).")

            cut_facets = entities_tags.find(2)

            # dfx.mesh.refine MODIFIES the input mesh preventing the computation of the estimator below.
            # To avoid it I follow the trick from https://fenicsproject.discourse.group/t/strange-behavior-after-using-create-mesh/14887/3
            # I create a dummy_mesh as a submesh that is in fact a copy of working_mesh and the refinement is made from dummy_mesh.
            num_cells = working_mesh.topology.index_map(working_mesh.topology.dim).size_global
            dummy_mesh = dfx.mesh.create_submesh(working_mesh, working_mesh.topology.dim, np.arange(num_cells))[0]
            dummy_mesh.topology.create_entities(dummy_mesh.topology.dim - 1)
            correction_mesh, _, _ = dfx.mesh.refine(dummy_mesh, cut_facets)

            CGhfElement = element("Lagrange",
                                  correction_mesh.topology.cell_name(),
                                  self.levelset_space.ufl_element().degree)
            V_correction = dfx.fem.functionspace(correction_mesh, CGhfElement)
            cdim = correction_mesh.topology.dim
            num_cells = correction_mesh.topology.index_map(cdim).size_global
            correction_mesh_cells = np.arange(num_cells)
            nmm = dfx.fem.create_interpolation_data(
                            V_correction,
                            self.levelset_space,
                            correction_mesh_cells,
                            padding=1.e-14)

            phih_correction = dfx.fem.Function(V_correction)
            phih_correction.interpolate_nonmatching(phih,
                                                    correction_mesh_cells,
                                                    interpolation_data=nmm)

            phif_correction = dfx.fem.Function(V_correction)
            phif_correction.interpolate(self.levelset)

            whf = dfx.fem.Function(V_correction)
            if self.solution_wh is None:
                raise ValueError("SOLVER_NAME.solution_wh is None, did you forget to solve ?(SOLVER_NAME.solve)")

            whf.interpolate_nonmatching(self.solution_wh,
                                        correction_mesh_cells,
                                        interpolation_data=nmm)

            correction_function = dfx.fem.Function(V_correction)
            correction_function.x.array[:] = (phih_correction.x.array[:] - phif_correction.x.array[:]) * whf.x.array[:]

            CGpfElement = element("Lagrange",
                                  working_mesh.topology.cell_name(),
                                  self.levelset_element.degree + 1)
            V_working = dfx.fem.functionspace(working_mesh, CGpfElement)

            cdim = working_mesh.topology.dim
            num_cells = working_mesh.topology.index_map(cdim).size_global
            working_mesh_cells = np.arange(num_cells)

            nmm = dfx.fem.create_interpolation_data(
                            V_working,
                            V_correction,
                            working_mesh_cells,
                            padding=1.e-14)
        
            correction_function_V = dfx.fem.Function(V_working)
            correction_function_V.interpolate_nonmatching(correction_function, 
                                                          working_mesh_cells,
                                                          interpolation_data=nmm)
        return correction_function_V
    
    def estimate_residual(self,
                          V0: FunctionSpace | None = None,
                          quadrature_degree: int | None = None,
                          boundary_term: bool = False) -> Tuple[dict[str, Any], dict[str, Any], Function]:
        """ Compute the local and global contributions of the residual a posteriori error estimators for the H10 and L2 norms.

        Args:
            V0: the function space in which the local contributions of the residual estimators are interpolated.
            quadrature_degree: the quadrature degree used in the integrals of the residual estimator.
            boundary_term: if True, the boundary term inner(inner(uh, n), w0) * ds is added to the residual estimator.

        Returns:
            h10_residuals: dictionnary containing all the H1 semi-norm residuals.
            l2_residuals: dictionnary containing all the L2 norm residuals.
        """
        self.print("Estimate residual.")

        if self.solution is None:
            raise ValueError("SOLVER_NAME.solution is None, did you forget to solver ? (SOLVER_NAME.solve)")
        uh = self.solution
        if self.box_mode:
            if self.bg_mesh_cells_tags is None:
                raise ValueError("SOLVER_NAME.bg_mesh_cells_tags is None, did you forget to compute tags ? (SOLVER_NAME.compute_tags)")
            working_cells_tags = self.bg_mesh_cells_tags
            if self.mesh is None:
                raise ValueError("SOLVER_NAME.mesh is None.")
            working_mesh = self.mesh
        else:
            if self.submesh_cells_tags is None:
                raise ValueError("SOLVER_NAME.submesh_cells_tags is None, did you forget to compute tags ? (SOLVER_NAME.compute_tags)")
            working_cells_tags = self.submesh_cells_tags
            if self.submesh is None:
                raise ValueError("SOLVER_NAME.submesh is None.")
            working_mesh = self.submesh

        if quadrature_degree is None:
            k = uh.function_space.element.basix_element.degree
            quadrature_degree_cells  = max(0, k - 2)
            quadrature_degree_facets = max(0, k - 1)
        
        dx = ufl.Measure("dx",
                         domain=working_mesh,
                         subdomain_data=working_cells_tags,
                         metadata={"quadrature_degree": quadrature_degree_cells})
        dS = ufl.Measure("dS",
                         domain=working_mesh,
                         subdomain_data=self.facets_tags,
                         metadata={"quadrature_degree":  quadrature_degree_facets})
        ds = ufl.Measure("ds",
                         domain=working_mesh,
                         metadata={"quadrature_degree":  quadrature_degree_facets})

        n   = ufl.FacetNormal(working_mesh)
        h_T = ufl.CellDiameter(working_mesh)
        h_E = ufl.FacetArea(working_mesh)

        if self.FE_space is None:
            raise ValueError("SOLVER_NAME.FE_space is None, did you forget to set the variational formulation ? (SOLVER_NAME.set_variational_formulation)")
        if self.rhs is None:
            raise ValueError("SOLVER_NAME.rhs is None, did you forget to set the source term ? (SOLVER_NAME.set_source_term)")

        f_h = self.rhs.interpolate(self.FE_space)

        r = f_h + div(grad(uh))
        J_h = jump(grad(uh), -n)

        # Boundary correction function as (phi_h - I_h phi) w_h, where I_h phi is an interpolation of phi into a finer space.
        if self.levelset_space is None:
            raise ValueError("SOLVER_NAME.levelset_space is None.")
        if self.levelset is None:
            raise ValueError("SOLVER_NAME.levelset is None.")

        if self.boundary_refinement_type=='p':
            correction_function = self._compute_boundary_correction_function(working_mesh, working_cells_tags, self.boundary_refinement_type)
        else:
            correction_function = self._compute_boundary_correction_function(working_mesh, self.facets_tags, self.boundary_refinement_type)

        correction_function_V = dfx.fem.Function(self.FE_space)
        correction_function_V.interpolate(correction_function)

        geometry_correction = inner(grad(correction_function),
                                    grad(correction_function))

        if V0 is None:
            DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(working_mesh, DG0Element)

        w0 = ufl.TestFunction(V0)

        """
        H10 estimator
        """
        if self.box_mode:
            # Interior residual
            eta_T = h_T**2 * inner(inner(r, r), w0) * self.v0 * (dx(1) + dx(2))

            # Facets residual
            eta_E = avg(h_E) * inner(inner(J_h, J_h), avg(w0)) * avg(self.v0) * (dS(1) + dS(2))

            eta_geometry = inner(geometry_correction, w0) * self.v0 * (dx(1) + dx(2))
        else:
            # Interior residual
            eta_T = h_T**2 * inner(inner(r, r), w0) * (dx(1) + dx(2))

            # Facets residual
            eta_E = avg(h_E) * inner(inner(J_h, J_h), avg(w0)) * (dS(1) + dS(2))

            eta_geometry = inner(geometry_correction, w0) * (dx(1) + dx(2))

        eta = eta_T + eta_E + eta_geometry

        eta_boundary = None
        if boundary_term:
            if self.box_mode:
                eta_boundary = h_E * inner(inner(grad(uh), n), inner(grad(uh), n)) * w0 * self.v0 * ds
            else:
                eta_boundary = h_E * inner(inner(grad(uh), n), inner(grad(uh), n)) * w0 * ds

            eta += eta_boundary

        eta_form = dfx.fem.form(eta)
        eta_vec = assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.x.petsc_vec.setArray(eta_vec.array[:])
        self.eta_h_H10 = eta_h

        h10_residuals = {"Interior residual":       eta_T,
                         "Internal edges residual": eta_E,
                         "Geometry residual":       eta_geometry,
                         "Boundary edges residual": eta_boundary}

        """
        L2 estimator
        """
        geometry_correction = inner(correction_function,
                                    correction_function)

        if self.box_mode:
            eta_T = h_T**4 * inner(inner(r, r), w0) * self.v0 * (dx(1) + dx(2))
            eta_E = avg(h_E)**3 * inner(inner(J_h, J_h), avg(w0)) * avg(self.v0) * (dS(1) + dS(2))
            eta_geometry = inner(geometry_correction, w0) * self.v0 * (dx(1) + dx(2))
        else:
            eta_T = h_T**4 * inner(inner(r, r), w0) * (dx(1) + dx(2))
            eta_E = avg(h_E)**3 * inner(inner(J_h, J_h), avg(w0)) * (dS(1) + dS(2))
            eta_geometry = inner(geometry_correction, w0) * (dx(1) + dx(2))

        eta = eta_T + eta_E + eta_geometry

        eta_boundary = None
        if boundary_term:
            if self.box_mode:
                eta_boundary = h_E**3 * inner(inner(grad(uh), n), inner(grad(uh), n)) * w0 * self.v0 * ds
            else:
                eta_boundary = h_E**3 * inner(inner(grad(uh), n), inner(grad(uh), n)) * w0 * ds
            eta += eta_boundary
        eta_form = dfx.fem.form(eta)

        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.x.petsc_vec.setArray(eta_vec.array[:])
        self.eta_h_L2 = eta_h

        l2_residuals = {"Interior residual":       eta_T,
                        "Internal edges residual": eta_E,
                        "Geometry residual":       eta_geometry,
                        "Boundary edges residual": eta_boundary}
        return h10_residuals, l2_residuals, correction_function_V

class FEMSolver(GenericSolver):
    """ Class representing a FEM solver as a GenericSolver object."""
    def __init__(self,
                 mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: PETSc_KSP,
                 ref_strat: str = "uniform",
                 num_step: int = 0,
                 save_output: bool = True) -> None:
        """ Initialize a FEM solver object.

        Args:
            mesh: dolfinx.mesh.Mesh object, the conformal mesh.
            FE_element: basix.ufl.element object, the finite element used in the phiFEM discretization.
            PETSc_solver: petsc4py.PETSc.KSP object, the PETSc solver used to solve the finite element linear system.
            ref_strat: (optional) str, the refinement strategy ('uniform' for uniform refinement, 'H10' for adaptive refinement based on the H10 residual estimator, 'L2' for adaptive refinement based on the L2 residual estimator).
            num_step: (optional) int, the refinement step number.
            save_output: (optional) bool, if True, save the functions, meshes and values to the disk.
        """
        super().__init__(mesh,
                         FE_element,
                         PETSc_solver,
                         ref_strat=ref_strat,
                         num_step=num_step,
                         save_output=save_output)

        self.bcs: list[DirichletBC] | list[int] = []
        self.FE_space: FunctionSpace            = dfx.fem.functionspace(mesh, FE_element)
        self.solver_type: str                   = "FEM"
        self.mesh: Mesh                         = mesh
    
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
        super().print("Set variational formulation.")

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

    def solve(self) -> None:
        super().solve()
        self.solution = dfx.fem.Function(self.FE_space)
        assert self.solution_wh is not None, "SOLVER_NAME.solution_wh is None, did you forget to solve ? (SOLVER_NAME.solve)"
        self.solution.x.array[:] = self.solution_wh.x.array[:]

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
        super().print("Compute estimators.")

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