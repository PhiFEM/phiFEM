import jax.numpy as jnp
import numpy as np

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