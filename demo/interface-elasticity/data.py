import numpy as np
import ufl

# Lamé coefficients
def lmbda(E, nu):
    return E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)


def mu(E, nu):
    return E / 2.0 / (1.0 + nu)


# Material parameters outside the hole
E_in = 1.0
nu_in = 0.3
lmbda_in = lmbda(E_in, nu_in)
mu_in = mu(E_in, nu_in)
# Material parameters inside the hole
E_out = 0.001
nu_out = 0.3
lmbda_out = lmbda(E_out, nu_out)
mu_out = mu(E_out, nu_out)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma_in(u):
    return lmbda_in * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu_in * epsilon(u)


def sigma_out(u):
    return lmbda_out * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu_out * epsilon(
        u
    )


def levelset(x):
    return 1.0 - (x[0] ** 2 + x[1] ** 2)


def exact_solution(x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    val = np.cos(r) - np.cos(1.0) / E_in
    mask = r < 1.0
    val[mask] = val[mask] * (E_in / E_out)
    return np.vstack([val, val])


def cos_vec(mode):
    return lambda x: ufl.as_vector(
        [
            mode.cos(mode.sqrt(x[0] ** 2 + x[1] ** 2)),
            mode.cos(mode.sqrt(x[0] ** 2 + x[1] ** 2)),
        ]
    )