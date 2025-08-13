import numpy as np
import ufl

DISK_RADIUS = np.pi/10.
# Disk center
X0, Y0 = [0.5, 0.5]

# Lamé coefficients
def lmbda(E, nu):
    return E * nu/(1.0+nu)/(1.+2.*nu)
def mu(E, nu):
    return E/2.0/(1.0+nu)

# Material parameters inside the disk
E_in = 10.
nu_in = 0.3
lmbda_in = lmbda(E_in, nu_in)
mu_in = mu(E_in, nu_in)
# Material parameters outside the disk
E_out = 0.7
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
    return (x[0] - X0)**2 + (x[1] - Y0)**2 - DISK_RADIUS**2

def dirichlet(x):
    vals_y = np.ones_like(x[1])
    mask = np.where(x[0] < 0.5)
    vals_y[mask] = -1.
    vals = np.vstack([np.zeros_like(x[0]), vals_y])
    return vals