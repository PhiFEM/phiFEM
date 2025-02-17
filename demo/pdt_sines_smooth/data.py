import jax.numpy as jnp
import numpy as np

tilt_angle = np.pi/6.
def rotation(angle, x):
    if x.shape[0] == 3:
        R = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                       [jnp.sin(angle),  jnp.cos(angle), 0],
                       [             0,               0, 1]])
    elif x.shape[0] == 2:
        R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                       [jnp.sin(angle),  jnp.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(jnp.asarray(x))

def tilted_square(x):
    def fct(x):
        return jnp.sum(jnp.abs(rotation(-tilt_angle + jnp.pi/4., x)), axis=0)
    return fct(x) - jnp.sqrt(2.)/2.

# Levelset as a smooth product of sines function
def expression_levelset(x):
    vect = np.full_like(x, 0.5)
    val = -np.sin(np.pi * (rotation(-tilt_angle, x - rotation(tilt_angle, vect))[0, :])) * \
           np.sin(np.pi * (rotation(-tilt_angle, x - rotation(tilt_angle, vect))[1, :]))

    # val_ext = tilted_square(x)
    # val[val_ext > 0.] = val_ext[val_ext > 0.]
    return val

def expression_detection_levelset(x):
    return tilted_square(x)

def expression_u_exact(x):
    return jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
           jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[1, :])

# Not required since jax will compute the negative laplacian of u_exact automatically but we add it since we know the analytical expression :)
def expression_rhs(x):
    return 8. * jnp.pi**2 * jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
                            jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[1, :])

# FEM data
point_1 = rotation(tilt_angle,
                    np.array([-0.5, -0.5]))
point_2 = rotation(tilt_angle,
                    np.array([0.5,  -0.5]))
point_3 = rotation(tilt_angle,
                    np.array([0.5,  0.5]))
point_4 = rotation(tilt_angle,
                    np.array([-0.5, 0.5]))

geom_vertices = np.vstack([point_1, point_2, point_3, point_4]).T