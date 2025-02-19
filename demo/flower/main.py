import argparse
import jax.numpy as jnp
import numpy as np
import os
from phiFEM.phifem.poisson_dirichlet import poisson_dirichlet_phiFEM, poisson_dirichlet_FEM

parent_dir = os.path.split(os.path.abspath(__file__))[0]

# Levelset and RHS expressions taken from: https://academic.oup.com/imajna/article-abstract/42/1/333/6041856?redirectedFrom=fulltext
def expression_levelset(x):
    def phi0(x):
        r = jnp.full_like(x[0, :], 2.)
        return jnp.square(x[0, :]) + jnp.square(x[1, :]) - jnp.square(r)
    val = phi0(x)

    for i in range(1, 9):
        xi = 2. * (jnp.cos(jnp.pi/8.) + jnp.sin(jnp.pi/8.)) * jnp.cos(i * jnp.pi/4.)
        yi = 2. * (jnp.cos(jnp.pi/8.) + jnp.sin(jnp.pi/8.)) * jnp.sin(i * jnp.pi/4.)
        ri = jnp.sqrt(2.) * 2. * (jnp.sin(jnp.pi/8.) + jnp.cos(jnp.pi/8.)) * jnp.sin(jnp.pi/8.)
        def phi_i(x):
            return jnp.square(x[0, :] - jnp.full_like(x[0, :], xi)) + \
                   jnp.square(x[1, :] - jnp.full_like(x[1, :], yi)) - \
                   jnp.square(jnp.full_like(x[0, :], ri))
        
        val = jnp.minimum(val, phi_i(x))
    return val

def expression_rhs(x):
    x1 = 2. * (np.cos(np.pi/8.) + np.sin(np.pi/8.))
    y1 = 0.
    r1 = np.sqrt(2.) * 2. * (np.sin(np.pi/8.) + np.cos(np.pi/8.)) * np.sin(np.pi/8.)

    val = np.square(x[0, :] - np.full_like(x[0, :], x1)) + \
          np.square(x[1, :] - np.full_like(x[1, :], y1))
        
    return np.where(val <= np.square(r1)/2., 10., 0.)


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="flower test case.",
                                     description="Run iterations of FEM or phiFEM with uniform or adaptive refinement.")

    parser.add_argument("solver",        type=str, choices=["FEM", "phiFEM"])
    parser.add_argument("char_length",   type=float)
    parser.add_argument("num_it",        type=int)
    parser.add_argument("ref_mode",      type=str, choices=["uniform", "H10", "L2"])
    parser.add_argument("--exact_error", default=False, action='store_true', help="Compute the exact errors.")
    args = parser.parse_args()
    solver = args.solver
    cl = args.char_length
    num_it = args.num_it
    ref_method = args.ref_mode
    compute_exact_errors = args.exact_error

    output_dir = os.path.join(parent_dir, "output_" + solver, ref_method)

    bbox_vertices = np.array([[-4.5,  4.5],
                              [-4.5,  4.5]])

    if solver=="phiFEM":
        poisson_dirichlet_phiFEM(cl,
                                 num_it,
                                 expression_levelset,
                                 parent_dir,
                                 expression_rhs=expression_rhs,
                                 bbox_vertices=bbox_vertices,
                                 ref_method=ref_method,
                                 compute_exact_error=compute_exact_errors)
    
    if solver=="FEM":
        poisson_dirichlet_FEM(cl,
                              num_it,
                              expression_levelset,
                              parent_dir,
                              expression_rhs=expression_rhs,
                              quadrature_degree=4,
                              ref_method=ref_method,
                              compute_exact_error=compute_exact_errors,
                              bbox_vertices=bbox_vertices,
                              remesh_boundary=True)