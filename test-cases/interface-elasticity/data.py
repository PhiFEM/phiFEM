import numpy as np
import ufl

# Lamé coefficients
def lmbda(E, nu):
    return E * nu/(1.0+nu)/(1.-2.*nu)
def mu(E, nu):
    return E/2.0/(1.0+nu)

# Material parameters inside the disk
E_in = 1.
nu_in = 0.3
lmbda_in = lmbda(E_in, nu_in)
mu_in = mu(E_in, nu_in)
# Material parameters outside the disk
E_out = 0.001
nu_out = 0.3
lmbda_out = lmbda(E_out, nu_out)
mu_out = mu(E_out, nu_out)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma_in(u):
    return lmbda_in * ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.0 * mu_in * epsilon(u)

def sigma_out(u):
    return lmbda_out * ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.0 * mu_out * epsilon(u)

def levelset_1(x):
    return 0.65 - (x[0]**2 + (0.6 * x[1])**2)

def levelset_2(x):
    return 1. - ((x[0])**2 + (x[1])**2)

def levelset_3(x):
    p = 1.8
    return 1.1 - (np.abs(x[0] + 0.05)**p + np.abs(x[1] + 0.05)**p)**(1/p)