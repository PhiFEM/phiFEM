from   basix.ufl import element
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.mesh import Mesh
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
from   numpy.typing import NDArray
import os
from   os import PathLike
from   petsc4py.PETSc import Options, KSP # type: ignore[attr-defined]
from   typing import Tuple, Any

from FEM.compute_exact_error import compute_exact_error

from phiFEM.phifem.solver import PhiFEMSolver
from phiFEM.phifem.continuous_functions import Levelset, ExactSolution, ContinuousFunction
from phiFEM.phifem.saver import ResultsSaver
from phiFEM.phifem.utils import assemble_and_save_residual

PathStr = PathLike[str] | str
NDArrayTuple = Tuple[npt.NDArray[np.float64]]
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class PhiFEMRefinementLoop:
    def __init__(self,
                 initial_mesh_size: float,
                 iteration_number: int,
                 refinement_method: str,
                 levelset: Levelset,
                 stabilization_parameter: float,
                 source_dir: PathStr):
        
        if refinement_method not in ["uniform", "H10", "L2"]:
            raise ValueError("refinement_method must be 'uniform', 'H10' or 'L2'.")

        self.bbox: NDArray                        = np.array([[-1.0, 1.0],
                                                              [-1.0, 1.0]])
        self.boundary_detection_degree: int       = 1
        self.box_mode: bool                       = False
        self.exact_error: bool                    = False
        self.exact_solution: ExactSolution | None = None
        self.finite_element_degree: int           = 1
        self.initial_mesh_size: float             = initial_mesh_size
        self.initial_bg_mesh: Mesh | None         = None
        self.iteration_number: int                = iteration_number
        self.levelset: Levelset                   = levelset
        self.levelset_degree: int                 = 1
        self.marking_parameter: float             = 0.3
        self.quadrature_degree: int | None        = None
        self.rhs: ContinuousFunction | None       = None
        self.ref_degree: float                    = 2
        self.refinement_method: str               = refinement_method
        self.results_saver: ResultsSaver          = ResultsSaver(os.path.join(source_dir,
                                                                              "output_phiFEM",
                                                                              refinement_method))
        self.stabilization_parameter: float       = stabilization_parameter
        self.use_fine_space: bool                 = False

    def set_parameters(self, parameters: dict[str, Any], expressions: dict[str, NDArrayFunction]):
        self.bbox                      = np.asarray(parameters["bbox"])
        self.boundary_detection_degree = parameters["boundary_detection_degree"]
        self.box_mode                  = parameters["box_mode"] 
        self.boundary_refinement_type  = parameters["boundary_refinement_type"] 
        self.exact_error               = parameters["exact_error"]
        self.finite_element_degree     = parameters["finite_element_degree"]
        self.levelset_degree           = parameters["levelset_degree"]
        self.marking_parameter         = parameters["marking_parameter"]
        self.quadrature_degree         = parameters["quadrature_degree"]
        self.use_fine_space            = parameters["use_fine_space"]
        self.save_output               = parameters["save_output"]

        if expressions["expression_rhs"] is not None:
            self.rhs = ContinuousFunction(expressions["expression_rhs"])
        else:
            self.rhs = None
        
        if expressions["expression_u_exact"] is not None:
            self.exact_solution = ExactSolution(expressions["expression_u_exact"])
        else:
            self.exact_solution = None

    def set_bbox(self, bbox: NDArray):
        self.bbox = bbox

    def set_boundary_detection_degree(self, detection_degree: int):
        self.boundary_detection_degree = detection_degree
    
    def set_box_mode(self, box_mode: bool):
        self.box_mode = box_mode
    
    def set_boundary_refinement_type(self, boundary_refinement_type: str):
        if boundary_refinement_type not in ['h', 'p']:
            raise ValueError("boundary_refinement_type must be 'h' or 'p'.")
        self.boundary_refinement_type = boundary_refinement_type
    
    def set_exact_error_on(self, exact_error: bool):
        self.exact_error = exact_error
    
    def set_exact_solution(self, expression_exact_solution: NDArrayFunction):
        self.exact_solution = ExactSolution(expression_exact_solution)
    
    def set_finite_element_degree(self, finite_element_degree: int):
        self.finite_element_degree = finite_element_degree
    
    def set_levelset_degree(self, levelset_degree: int):
        self.levelset_degree = levelset_degree
    
    def set_marking_parameter(self, marking_parameter: float):
        self.marking_parameter = marking_parameter
    
    def set_quadrature_degree(self, quadrature_degree: int):
        self.quadrature_degree = quadrature_degree
    
    def set_use_fine_space(self, use_fine_space: bool):
        self.use_fine_space = use_fine_space

    def set_results_saver(self, output_dir: PathStr):
        self.results_saver = ResultsSaver(output_dir)
    
    def set_rhs(self, expression_rhs: NDArrayFunction):
        self.rhs = ContinuousFunction(expression_rhs)

    def set_save_output(self, save_output: bool):
        self.save_output = save_output
    
    def set_stabilization_parameter(self, stabilization_parameter: float):
        self.stabilization_parameter = stabilization_parameter
    
    def create_initial_bg_mesh(self):
        """Create the initial background mesh
        """
        print("Create initial mesh.")
        nx = int(np.abs(self.bbox[0, 1] - self.bbox[0, 0]) * np.sqrt(2.) / self.initial_mesh_size)
        ny = int(np.abs(self.bbox[1, 1] - self.bbox[1, 0]) * np.sqrt(2.) / self.initial_mesh_size)
        self.initial_bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, self.bbox.T, [nx, ny])
        if self.results_saver is not None:
            self.results_saver.save_mesh(self.initial_bg_mesh, "initial_bg_mesh")
    
    def run(self):
        if self.rhs is None:
            self.rhs: ContinuousFunction
            if self.exact_solution is not None:
                self.exact_solution.compute_negative_laplacian()
                if self.exact_solution.nlap is None:
                    raise ValueError("exact_solution.nlap is None.")
                self.rhs = self.exact_solution.nlap
            else:
                raise ValueError("phiFEMRefinementLoop need expression_rhs or expression_u_exact not to be None.")

        if self.initial_bg_mesh is None:
            raise ValueError("REFINEMENT_LOOP.initial_bg_mesh is None, did you forget to create the initial mesh ? (REFINEMENT_LOOP.create_initial_bg_mesh())")
        working_mesh = self.initial_bg_mesh
        for i in range(self.iteration_number):
            whElement        = element("Lagrange", working_mesh.topology.cell_name(), self.finite_element_degree)
            levelsetElement  = element("Lagrange", working_mesh.topology.cell_name(), self.levelset_degree)

            # Parametrization of the PETSc solver
            options = Options()
            options["ksp_type"] = "cg"
            options["pc_type"] = "hypre"
            options["ksp_rtol"] = 1e-7
            options["pc_hypre_type"] = "boomeramg"
            petsc_solver = KSP().create(working_mesh.comm)
            petsc_solver.setFromOptions()

            phiFEM_solver = PhiFEMSolver(working_mesh,
                                         whElement,
                                         petsc_solver,
                                         levelset_element=levelsetElement,
                                         detection_degree=self.boundary_detection_degree,
                                         use_fine_space=self.use_fine_space,
                                         box_mode=self.box_mode,
                                         boundary_refinement_type=self.boundary_refinement_type,
                                         num_step=i,
                                         ref_strat=self.refinement_method)

            phiFEM_solver.set_source_term(self.rhs)
            phiFEM_solver.set_levelset(self.levelset)
            phiFEM_solver.compute_tags()
            v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation(sigma=self.stabilization_parameter,
                                                                             quadrature_degree=self.quadrature_degree)
            phiFEM_solver.assemble()
            phiFEM_solver.solve()
            uh = phiFEM_solver.get_solution()
            wh = phiFEM_solver.get_solution_wh()

            if phiFEM_solver.submesh is None:
                raise TypeError("phiFEM_solver.submesh is None.")

            working_mesh = phiFEM_solver.submesh

            h10_residuals, l2_residuals, correction_function = phiFEM_solver.estimate_residual()
            eta_h_H10 = phiFEM_solver.get_H10_residual()
            global_eta_H10 = np.sqrt(sum(eta_h_H10.x.array[:]))
            self.results_saver.add_new_value("H10 estimator", global_eta_H10)
            eta_h_L2 = phiFEM_solver.get_L2_residual()
            global_eta_L2 = np.sqrt(sum(eta_h_L2.x.array[:]))
            self.results_saver.add_new_value("L2 estimator", global_eta_L2)

            CG1Element = element("Lagrange", working_mesh.topology.cell_name(), 1)
            V = dfx.fem.functionspace(working_mesh, CG1Element)
            phiV = self.levelset.interpolate(V)
            # Save results
            if self.save_output:
                for dict_res, norm in zip([h10_residuals, l2_residuals], ["H10", "L2"]):
                    for key, res_letter in zip(dict_res.keys(), ["T", "E", "G", "Eb"]):
                        eta = dict_res[key]
                        if eta is not None:
                            res_name = "eta_" + res_letter + "_" + norm
                            assemble_and_save_residual(working_mesh, self.results_saver, eta, res_name, i)

                self.results_saver.add_new_value("dofs", num_dofs)
                self.results_saver.save_function(eta_h_H10,           f"eta_h_H10_{str(i).zfill(2)}")
                self.results_saver.save_function(eta_h_L2,            f"eta_h_L2_{str(i).zfill(2)}")
                self.results_saver.save_function(phiV,                f"phi_V_{str(i).zfill(2)}")
                self.results_saver.save_function(uh,                  f"uh_{str(i).zfill(2)}")
                self.results_saver.save_function(wh,                  f"wh_{str(i).zfill(2)}")
                self.results_saver.save_function(correction_function, f"boundary_correction_{str(i).zfill(2)}")
                self.results_saver.save_mesh    (working_mesh,        f"mesh_{str(i).zfill(2)}")
                if v0 is not None:
                    self.results_saver.save_function(v0, f"v0_{str(i).zfill(2)}")

            if self.exact_error:
                expression_u_exact = self.exact_solution.expression
                global_H10_error, global_L2_error, H10_local_errors, L2_local_errors, _, _ = compute_exact_error(uh,
                                                                                                                 self.rhs,
                                                                                                                 output_path=self.results_saver.output_path,
                                                                                                                 iteration_num=i,
                                                                                                                 phifem_measure=dx,
                                                                                                                 expression_u_exact=expression_u_exact)
                self.results_saver.save_function(H10_local_errors, f"H10_error_{str(i).zfill(2)}")
                self.results_saver.add_new_value("H10 error", global_H10_error)
                self.results_saver.save_function(L2_local_errors,  f"L2_error_{str(i).zfill(2)}")
                self.results_saver.add_new_value("L2 error", global_L2_error)
                self.results_saver.add_new_value("H10 efficiency", global_eta_H10/global_H10_error)
                self.results_saver.add_new_value("L2 efficiency",  global_eta_L2/global_L2_error)

            # Marking
            if i < self.iteration_number - 1:
                # Uniform refinement (Omega_h only)
                if self.refinement_method == "uniform":
                    working_mesh, _, _ = dfx.mesh.refine(working_mesh)

                # Adaptive refinement
                if self.refinement_method in ["H10", "L2"]:
                    facets2ref = phiFEM_solver.marking()
                    working_mesh, _, _ = dfx.mesh.refine(working_mesh, facets2ref)

            if self.save_output:
                self.results_saver.save_values("results.csv")
                print("\n")