import argparse
import jax.numpy as jnp
import numpy as np
import os
from phiFEM.phifem.poisson_dirichlet import poisson_dirichlet_phiFEM, poisson_dirichlet_FEM

parent_dir = os.path.dirname(__file__)

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Sine-sine test case.",
                                     description="Run iterations of FEM or phiFEM with uniform or adaptive refinement.")

    parser.add_argument("solver", 
                        type=str,
                        choices=["FEM", "phiFEM"],
                        help="Finite element solver.")
    parser.add_argument("char_length",
                        type=float,
                        help="Size of the initial mesh.")
    parser.add_argument("num_it",
                        type=int,
                        help="Number of refinement iterations.")
    parser.add_argument("ref_mode",
                        type=str,
                        choices=["uniform", "H10", "L2"],
                        help="Refinement strategy.")
    parser.add_argument("--exact_error",
                        default=False,
                        action='store_true',
                        help="Compute the exact errors.")

    args = parser.parse_args()
    solver               = args.solver
    cl                   = args.char_length
    num_it               = args.num_it
    ref_method           = args.ref_mode
    compute_exact_errors = args.exact_error

    if solver=="phiFEM":
        poisson_dirichlet_phiFEM(cl,
                                 num_it,
                                 expression_levelset,
                                 parent_dir,
                                 expression_u_exact=expression_u_exact,
                                 expression_rhs=expression_rhs,
                                 bbox_vertices=np.array([[-1., 1.],
                                                         [-1., 1.]]),
                                 ref_method=ref_method,
                                 compute_exact_error=compute_exact_errors)
    
    if solver=="FEM":
        point_1 = rotation(tilt_angle,
                            np.array([-0.5, -0.5]))
        point_2 = rotation(tilt_angle,
                            np.array([0.5,  -0.5]))
        point_3 = rotation(tilt_angle,
                            np.array([0.5,  0.5]))
        point_4 = rotation(tilt_angle,
                            np.array([-0.5, 0.5]))

        geom_vertices = np.vstack([point_1, point_2, point_3, point_4]).T
        poisson_dirichlet_FEM(cl,
                              num_it,
                              expression_levelset,
                              parent_dir,
                              expression_u_exact=expression_u_exact,
                              expression_rhs=expression_rhs,
                              quadrature_degree=4,
                              ref_method=ref_method,
                              geom_vertices=geom_vertices,
                              compute_exact_error=compute_exact_errors)