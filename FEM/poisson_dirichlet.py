from   basix.ufl import element
from   collections.abc import Callable
from   contourpy import contour_generator
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   lxml import etree
import meshio # type: ignore
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
from   numpy.typing import NDArray
import os
from   os import PathLike
from   petsc4py.PETSc import Options, KSP # type: ignore[attr-defined]
import pygmsh # type: ignore
from   typing import Tuple, Any, cast

from FEM.solver import FEMSolver
from FEM.compute_exact_error import compute_exact_error

from phiFEM.phifem.continuous_functions import Levelset, ExactSolution, ContinuousFunction
from phiFEM.phifem.saver import ResultsSaver
from phiFEM.phifem.utils import assemble_and_save_residual

PathStr = PathLike[str] | str
NDArrayTuple = Tuple[npt.NDArray[np.float64]]
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class FEMRefinementLoop:
    def __init__(self,
                 initial_mesh_size: float,
                 iteration_number: int,
                 refinement_method: str,
                 levelset: Levelset,
                 source_dir: PathStr,
                 geometry_vertices: NDArray | None = None,
                 save_output: bool = True):
    
        if refinement_method not in ["uniform", "H10", "L2"]:
            raise ValueError("refinement_method must be 'uniform', 'H10' or 'L2'.")

        self.exact_error: bool                    = False
        self.exact_solution: ExactSolution | None = None
        self.finite_element_degree: int           = 1
        self.geometry_vertices: NDArray | None    = geometry_vertices
        self.initial_mesh_size: float             = initial_mesh_size
        self.iteration_number: int                = iteration_number
        self.levelset: Levelset                   = levelset
        self.marking_parameter: float             = 0.3
        self.quadrature_degree: int | None        = None
        self.rhs: ContinuousFunction | None       = None
        self.refinement_method: str               = refinement_method
        self.save_output: bool                    = save_output
        self.bbox: NDArray | None                 = None
        self.ref_degree: int                      = 2

        self.results_saver: ResultsSaver | None
        if save_output:
            self.results_saver = ResultsSaver(os.path.join(source_dir,
                                                           "output_FEM",
                                                           refinement_method))
        else:
            self.results_saver = None
    
    def set_parameters(self, parameters: dict[str, Any], expressions: dict[str, NDArrayFunction]):
        self.bbox                      = np.asarray(parameters["bbox"])
        self.exact_error               = parameters["exact_error"]
        self.exact_solution            = ExactSolution(expressions["expression_u_exact"])
        self.finite_element_degree     = parameters["finite_element_degree"]
        self.marking_parameter         = parameters["marking_parameter"]
        self.quadrature_degree         = parameters["quadrature_degree"]
        self.save_output               = parameters["save_output"]

        if expressions["expression_rhs"] is not None:
            self.rhs = ContinuousFunction(expressions["expression_rhs"])
        else:
            self.rhs = None
    
    def set_bbox(self, bbox: NDArray):
        self.bbox = bbox
    
    def set_ref_degree(self, ref_degree: float):
        self.ref_degree = ref_degree
    
    def mesh2d_from_levelset(self, interior_vertices: NDArray | None = None) -> npt.NDArray[np.float64]:
        """ Generate a 2D conforming mesh from a levelset function and saves it as an xdmf mesh.

        Args:
            interior_vertices: vertices inside the geometry. 

        Returns:
            The coordinates of the boundary vertices.
        """

        # TODO: is there a way to combine geom_vertices and contour generated vertices ?
        boundary_vertices: npt.NDArray[np.float64]
        if self.geometry_vertices is None:
            step = self.initial_mesh_size/np.sqrt(2.)

            if interior_vertices is None:
                x = np.arange(self.bbox[0,0], self.bbox[0,1] + step, step=step, dtype=np.float64)
                y = np.arange(self.bbox[1,0], self.bbox[1,1] + step, step=step, dtype=np.float64)
                X, Y = np.meshgrid(x, y, indexing="ij")
                X_flat, Y_flat = X.flatten(), Y.flatten()
            else:
                X_flat = interior_vertices[0,:]
                Y_flat = interior_vertices[1,:]

            arr = np.vstack([X_flat, Y_flat])
            detection = self.levelset.get_detection_expression()
            Z_flat = detection(arr)
            Z = np.reshape(Z_flat, X.shape)
            cg = contour_generator(x=X, y=Y, z=Z, line_type="ChunkCombinedCode")
            lines = np.asarray(cg.lines(0.)[0][0])

            # Removes points that are too close from each other
            lines_shifted = np.zeros_like(lines)
            lines_shifted[1:,:] = lines[:-1,:]
            lines_shifted[0,:] = lines[-1,:]
            diff = lines - lines_shifted
            dists = np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1]))
            lines = lines[dists>self.initial_mesh_size/2,:]

            boundary_vertices = np.unique(cast(npt.NDArray[np.float64], lines).T, axis=0)
        else:
            boundary_vertices = self.geometry_vertices
        
        if boundary_vertices.shape[0] == 1:
            boundary_vertices = np.vstack((boundary_vertices,
                                           np.zeros_like(boundary_vertices),
                                           np.zeros_like(boundary_vertices)))
        elif boundary_vertices.shape[0] == 2:
            boundary_vertices = np.vstack((boundary_vertices,
                                           np.zeros_like(boundary_vertices[0, :])))
        elif boundary_vertices.shape[0] == 3:
            boundary_vertices = boundary_vertices
        else:
            raise ValueError("The geometry vertices must have at most 3 coordinates, not more.")
        
        with pygmsh.geo.Geometry() as geom:
            # The boundary vertices are correctly ordered by matplotlib.
            geom.add_polygon(boundary_vertices.T, mesh_size=self.initial_mesh_size)
            # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
            # algorithm=9 for structured mesh (packing of parallelograms)
            mesh = geom.generate_mesh(dim=2, algorithm=1)

        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                triangular_cells = [("triangle", cell_block.data)]

        if self.results_saver is not None:
            meshio.write_points_cells(os.path.join(self.results_saver.output_path, "conforming_mesh.xdmf"), mesh.points, triangular_cells)
        
            # meshio and dolfinx use incompatible Grid names ("Grid" for meshio and "mesh" for dolfinx)
            # the lines below change the Grid name from "Grid" to "mesh" to ensure the compatibility between meshio and dolfinx.
            tree = etree.parse(os.path.join(self.results_saver.output_path, "conforming_mesh.xdmf"))
            root = tree.getroot()

            for grid in root.findall(".//Grid"):
                grid.set("Name", "mesh")
            
            tree.write(os.path.join(self.results_saver.output_path, "conforming_mesh.xdmf"), pretty_print=True, xml_declaration=True, encoding="UTF-8")
        
            with XDMFFile(MPI.COMM_WORLD, os.path.join(self.results_saver.output_path, "conforming_mesh.xdmf"), "r") as fi:
                self.mesh = fi.read_mesh()

        return boundary_vertices
    
    def run(self):
        if self.rhs is None:
            self.rhs: ContinuousFunction
            if self.exact_solution is not None:
                self.exact_solution.compute_negative_laplacian()
                if self.exact_solution.nlap is None:
                    raise ValueError("exact_solution.nlap is None.")
                self.rhs = self.exact_solution.nlap
            else:
                raise ValueError("FEMRefinementLoop need expression_rhs or expression_u_exact not to be None.")

        if self.mesh is None:
            raise ValueError("REFINEMENT_LOOP.mesh is None, did you forget to compute the mesh ? (REFINEMENT_LOOP.mesh2d_from_levelset())")        
        for i in range(self.iteration_number):
            uhElement = element("Lagrange", self.mesh.topology.cell_name(), self.finite_element_degree)

            # Parametrization of the PETSc solver
            options = Options()
            options["ksp_type"] = "cg"
            options["pc_type"] = "hypre"
            options["ksp_rtol"] = 1e-7
            options["pc_hypre_type"] = "boomeramg"
            petsc_solver = KSP().create(self.mesh.comm)
            petsc_solver.setFromOptions()

            FEM_solver = FEMSolver(self.mesh,
                                   uhElement,
                                   petsc_solver,
                                   num_step=i,
                                   ref_strat=self.refinement_method,
                                   save_output=self.save_output)
            
            FEM_solver.set_source_term(self.rhs)
            dbc = dfx.fem.Function(FEM_solver.FE_space)
            facets = dfx.mesh.locate_entities_boundary(
                                    self.mesh,
                                    1,
                                    lambda x: np.ones(x.shape[1], dtype=bool))
            dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
            bcs = [dfx.fem.dirichletbc(dbc, dofs)]
            FEM_solver.set_boundary_conditions(bcs)
            num_dofs = FEM_solver.set_variational_formulation()
            self.results_saver.add_new_value("dofs", num_dofs)
            FEM_solver.assemble()
            FEM_solver.solve()
            uh = FEM_solver.get_solution()

            h10_residuals, l2_residuals = FEM_solver.estimate_residual()
            eta_h_H10 = FEM_solver.get_H10_residual()
            global_eta_H10 = np.sqrt(sum(eta_h_H10.x.array[:]))
            self.results_saver.add_new_value("H10 estimator", global_eta_H10)
            eta_h_L2 = FEM_solver.get_L2_residual()
            global_eta_L2 = np.sqrt(sum(eta_h_L2.x.array[:]))
            self.results_saver.add_new_value("L2 estimator", global_eta_L2)

            # Save results
            if self.save_output:
                for dict_res, norm in zip([h10_residuals, l2_residuals], ["H10", "L2"]):
                    for key, res_letter in zip(dict_res.keys(), ["T", "E", "G", "Eb"]):
                        eta = dict_res[key]
                        if eta is not None:
                            res_name = "eta_" + res_letter + "_" + norm
                            assemble_and_save_residual(self.mesh, self.results_saver, eta, res_name, i)

                self.results_saver.save_function(eta_h_H10, f"eta_h_H10_{str(i).zfill(2)}")
                self.results_saver.save_function(eta_h_L2,  f"eta_h_L2_{str(i).zfill(2)}")
                self.results_saver.save_function(uh,        f"uh_{str(i).zfill(2)}")
                self.results_saver.save_mesh    (self.mesh, f"mesh_{str(i).zfill(2)}")

            if self.exact_error:
                expression_u_exact = self.exact_solution.expression
                global_H10_error, global_L2_error, H10_local_errors, L2_local_errors, _, _ = compute_exact_error(uh,
                                                                                                                 self.rhs,
                                                                                                                 output_path = self.results_saver.output_path,
                                                                                                                 iteration_num = i,
                                                                                                                 expression_u_exact=expression_u_exact)
                self.results_saver.save_function(H10_local_errors, f"H10_error_{str(i).zfill(2)}")
                self.results_saver.add_new_value("H10 error", global_H10_error)
                self.results_saver.save_function(L2_local_errors,  f"L2_error_{str(i).zfill(2)}")
                self.results_saver.add_new_value("L2 error", global_L2_error)
                self.results_saver.add_new_value("H10 efficiency", global_eta_H10/global_H10_error)
                self.results_saver.add_new_value("L2 efficiency", global_eta_L2/global_L2_error)

            if i < self.iteration_number - 1:
                # Uniform refinement (Omega_h only)
                if self.refinement_method == "uniform":
                    self.mesh, _, _ = dfx.mesh.refine(self.mesh)

                # Adaptive refinement
                if self.refinement_method in ["H10", "L2"]:
                    # Marking
                    facets2ref = FEM_solver.marking()
                    self.mesh, _, _ = dfx.mesh.refine(self.mesh, facets2ref)

                # if remesh_boundary:
                #     vertices_coordinates = self.mesh.geometry.x
                #     boundary_vertices = dfx.mesh.locate_entities_boundary(self.mesh,
                #                                                         0,
                #                                                         lambda x: np.full(x.shape[1], True, dtype=bool))
                #     boundary_vertices_coordinates = vertices_coordinates[boundary_vertices].T
                #     _ = mesh2d_from_levelset(1.,
                #                             phi,
                #                             output_dir=output_dir,
                #                             interior_vertices=boundary_vertices_coordinates)
            if self.save_output:
                with XDMFFile(MPI.COMM_WORLD, os.path.join(self.results_saver.output_path, "conforming_mesh.xdmf"), "w") as of:
                    of.write_mesh(self.mesh)
                
                self.results_saver.save_values("results.csv")
            
            print("\n")