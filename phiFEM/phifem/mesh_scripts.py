from   basix.ufl import element
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.cpp.graph import AdjacencyList_int32 # type: ignore
from   dolfinx.fem import Function
from   dolfinx.mesh import Mesh
import numpy as np
import numpy.typing as npt
from   os import PathLike
import ufl # type: ignore
from   ufl import inner, grad

from   phiFEM.phifem.continuous_functions import Levelset

PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

def compute_outward_normal(mesh: Mesh, levelset: Levelset) -> Function:
    """ Compute the outward normal to Omega_h.

    Args:
        mesh: the mesh on which the levelset is discretized.
        levelset: the levelset defining Omega_h.
    
    Returns:
        w0: the vector field defining the outward normal.
    """
    # This function is used to define the unit outward pointing normal to Gamma_h
    CG1Element = element("Lagrange", mesh.topology.cell_name(), 1)
    V = dfx.fem.functionspace(mesh, CG1Element)
    DG0VecElement = element("DG", mesh.topology.cell_name(), 0, shape=(mesh.topology.dim,))
    W0 = dfx.fem.functionspace(mesh, DG0VecElement)
    ext = dfx.fem.Function(V)
    ext.interpolate(levelset.exterior(0.))

    # Compute the unit outwards normal, but the scaling might create NaN where grad(ext) = 0
    normal_Omega_h = grad(ext) / (ufl.sqrt(inner(grad(ext), grad(ext))))

    # In order to remove the eventual NaNs, we interpolate into a vector functions space and enforce the values of the gradient to 0. in the cells that are not cut
    w0 = dfx.fem.Function(W0)
    w0.sub(0).interpolate(dfx.fem.Expression(normal_Omega_h[0], W0.sub(0).element.interpolation_points()))
    w0.sub(1).interpolate(dfx.fem.Expression(normal_Omega_h[1], W0.sub(1).element.interpolation_points()))

    w0.sub(0).x.array[:] = np.nan_to_num(w0.sub(0).x.array, nan=0.0)
    w0.sub(1).x.array[:] = np.nan_to_num(w0.sub(1).x.array, nan=0.0)
    return w0

def reshape_facets_map(f2c_connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
    """ Reshape the facets-to-cells indices mapping.

    Args:
        f2c_connect: the facets-to-cells connectivity.
    
    Returns:
        The facets-to-cells mapping as a ndarray.
    """
    f2c_array = f2c_connect.array
    num_cells_per_facet = np.diff(f2c_connect.offsets)
    max_cells_per_facet = num_cells_per_facet.max()
    f2c_map = -np.ones((len(f2c_connect.offsets) - 1, max_cells_per_facet), dtype=int)

    # Mask to select the boundary facets
    mask = np.where(num_cells_per_facet == 1)
    f2c_map[mask, 0] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    f2c_map[mask, 1] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    # Mask to select the interior facets
    mask = np.where(num_cells_per_facet == 2)
    f2c_map[mask, 0] = f2c_array[num_cells_per_facet.cumsum()[mask] - 2]
    f2c_map[mask, 1] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    return f2c_map