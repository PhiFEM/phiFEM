import numpy as np
import ufl

# Lamé coefficients
def lmbda(E, nu):
    return E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)

def mu(E, nu):
    return E / 2.0 / (1.0 + nu)


# Left material parameters 
E_left = 0.1
nu_left = 0.3
lmbda_left = lmbda(E_left, nu_left)
mu_left = mu(E_left, nu_left)
# Right material parameters
E_right = 1.0
nu_right = 0.3
lmbda_right = lmbda(E_right, nu_right)
mu_right = mu(E_right, nu_right)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma_left(u):
    return lmbda_left * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu_left * epsilon(u)


def sigma_right(u):
    return lmbda_right * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu_right * epsilon(
        u
    )


def levelset(x):
    return 0.5 * ufl.sin(np.pi * x[1] - 1.5) + x[0] - 1.

traction = ufl.as_vector((0., -1.))
