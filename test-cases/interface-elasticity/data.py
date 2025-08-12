import numpy as np
import jax
import jax.numpy as jnp
import ufl

def gradient(func, dim=3):
    # Wrap a single point to feed to func as a (3, 1) array
    def single_point_func(x_single):
        x = x_single[:, None]  # shape (3,) → (3,1)
        return func(x).squeeze()  # take scalar from shape (1,)
    
    # Gradient of the wrapped function
    grad_single = jax.jacrev(single_point_func)
    
    # Vectorize over N points
    def grad_func(x0):
        if x0.ndim == 1:
            return grad_single(x0[:dim]).T
        else:
            return jax.vmap(grad_single, in_axes=1)(x0[:dim, :]).T  # output shape: (dim, N)
    
    return grad_func

def sym(func):
    def sym_func(x):
        print(jax.vmap(func, in_axes=1)(x))
        return 0.5 * (jax.vmap(func, in_axes=1)(x) + jax.vmap(func, in_axes=1)(x).transpose(0, 2, 1))
    return sym_func

def div(func, dim=3, matrix=None):
    grad_func = gradient(func, dim=dim)
    def div_func(x0):
        grad_val = grad_func(x0).T
        ndim = grad_val.ndim
        grad_val_shape_0 = grad_val.shape[0]
        if ndim == 3:
            grad_val = grad_val.reshape((grad_val_shape_0,dim*dim))
        if dim>3:
            raise ValueError("dim can only be smaller or equal to 3.")
        if ndim == 3:
            unit_vec = jnp.zeros((dim, dim))
            unit_vec = unit_vec.at[0,0].set(1.)
            unit_vec = unit_vec.reshape((dim*dim,))
        else:
            unit_vec = jnp.zeros((dim,))
            unit_vec = unit_vec.at[0].set(1.)
        vals = jnp.inner(grad_val, unit_vec)
        for i in range(1,dim):
            if ndim == 3:
                unit_vec = jnp.zeros((dim, dim))
                unit_vec = unit_vec.at[i,i].set(1.)
                unit_vec = unit_vec.reshape((dim*dim,))
            else:
                unit_vec = jnp.zeros((dim,))
                unit_vec = unit_vec.at[i].set(1.)
            vals += jnp.inner(grad_val, unit_vec)
        
        if matrix is not None:
            vals = vals[:, jnp.newaxis, jnp.newaxis] * matrix
        return vals
    return div_func

"""
Interface linear elasticity test case data taken from  	
https://doi.org/10.48550/arXiv.2110.05072
"""

DISK_RADIUS = 0.3
# Disk center
X0, Y0 = [0.5, 0.5]

# Lamé coefficients
def lmbda(E, nu):
    return E * nu/(1.0+nu)/(1.+2.*nu)
def mu(E, nu):
    return E/2.0/(1.0+nu)

# Material parameters inside the disk
E_in = 7.
nu_in = 0.3
lmbda_in = lmbda(E_in, nu_in)
mu_in = mu(E_in, nu_in)
# Material parameters outside the disk
E_out = 2.28
nu_out = 0.3
lmbda_out = lmbda(E_out, nu_out)
mu_out = mu(E_out, nu_out)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma_in(u):
    return lmbda_in * ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.0 * mu_in * epsilon(u)

def sigma_out(u):
    return lmbda_out * ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.0 * mu_out * epsilon(u)

def levelset(x):
    return (x[0,:] - X0)**2 + (x[1,:] - Y0)**2 - DISK_RADIUS**2

def exact_solution(x):
    r = jnp.sqrt((x[0, :] - X0)**2 + (x[1, :] - Y0)**2)
    vals_x = (jnp.cos(r) - jnp.cos(DISK_RADIUS)) / E_in
    vals_x = jnp.where(r < DISK_RADIUS, vals_x, vals_x * E_in / E_out)
    return jnp.vstack([vals_x, vals_x])

def source_term_in(x):
    vals = div(exact_solution, dim=2, matrix=lmbda_in * jnp.eye(2))(x) + 2. * mu_in * sym(gradient(exact_solution, dim=2))(x)
    return vals

def source_term_out(x):
    vals = div(exact_solution, dim=2, matrix=lmbda_out * jnp.eye(2))(x) + 2. * mu_out * sym(gradient(exact_solution, dim=2))(x)
    return vals