from basix.ufl import element
from   collections.abc import Callable
from   copy import deepcopy
import dolfinx as dfx
from   dolfinx.mesh import Mesh
import functools
import inspect
import numpy as np
import os
from   typing import Any

from   phiFEM.phifem.saver import ResultsSaver

debug_env = os.getenv("DEBUG")
if debug_env=="True":
    debug = True
else:
    debug = False

def immutable(*arg_names: str) -> Callable:
    """ Decorator that check if the arguments specified have been modified (only if the DEBUG environment variable is set to True).

    Args:
        arg_names: the names of the wanted parameters to be immutable.
    """ 
    def decorator_mutable(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            if debug:
                # Get func's signature (needed because pytest messes up signatures)
                sig = inspect.signature(func)
                func_args = sig.bind(*args, **kwargs)
                func_args.apply_defaults()

                arg_copies = []
                for arg_name in arg_names:
                    arg_copies.append(deepcopy(func_args.arguments[arg_name]))
            
            result = func(*args, **kwargs)

            if debug:
                for i, arg_name in enumerate(arg_names):
                    if isinstance(func_args.arguments[arg_name], np.ndarray):
                        if not np.array_equal(arg_copies[i], func_args.arguments[arg_name]):
                            raise ValueError(f"Immutable parameter {arg_name} has been modified.")
                    else:
                        if arg_copies[i] != func_args.arguments[arg_name]:
                            raise ValueError(f"Immutable parameter {arg_name} has been modified.")
    
            return result
        return decorated
    return decorator_mutable

def assemble_and_save_residual(mesh: Mesh,
                               saver: ResultsSaver,
                               eta: Any | None,
                               name: str,
                               iteration: int) -> None:
    """ Assemble and save a residual (linear form) as a DG0 function.

    Args:
        mesh: the mesh supporting the DG0 space.
        saver: the saver object.
        eta: the unassembled residual.
        name: the name of the residual.
        iteration: the refinement iteration number.
    """
    if eta is not None:
        DG0Element = element("DG", mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(mesh, DG0Element)

        eta_form = dfx.fem.form(eta)
        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_global = np.sqrt(sum(eta_vec.array[:]))
        eta_h = dfx.fem.Function(V0)
        eta_h.x.petsc_vec.setArray(eta_vec.array[:])

        saver.save_function(eta_h, name + "_" + str(iteration).zfill(2))
        saver.add_new_value(name, eta_global)