import dolfinx as dfx
import numpy as np
import numpy.typing as npt
from dolfinx.cpp.graph import AdjacencyList_int32  # type: ignore
from packaging.version import Version
from typing import Any

def reshape_map(connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
    """Reshape the connected entities mapping.

    Args:
        connect: the connectivity.

    Returns:
        The mapping as a ndarray.
    """
    array = connect.array
    num_e1_per_e2 = np.diff(connect.offsets)
    max_offset = num_e1_per_e2.max()
    emap = -np.ones((len(connect.offsets) - 1, max_offset), dtype=int)

    # Mask to select the boundary facets
    for num in np.unique(num_e1_per_e2):
        mask = np.where(num_e1_per_e2 == num)[0]
        for n in range(num):
            emap[mask, n] = array[num_e1_per_e2.cumsum()[mask] - n - 1]
    return emap, max_offset


def interpolate_to_surface_submesh(
    u_volume: dfx.fem.Function,
    u_surface: dfx.fem.Function,
    submesh_facets: npt.NDArray[np.int32],
    integration_entities: npt.NDArray[np.int32],
    entity_maps: list[Any] | None = None,
):
    """
    This script has been borrowed from
    Interpolate a function `u_volume` into the function `u_surface`.
    Note:
        Does not work for DG as no dofs are associated with the facets in versions of DOLFINx
        prior to https://github.com/FEniCS/dolfinx/pull/4140, which is included in version
        0.11.0 and later.

    Args:
        u_volume: Function to interpolate data from
        u_surface: Function to interpolate data to
        submesh_facets: Cells in facet mesh
        integration_entities: Integration entities on the parent mesh
            corresponding to the facets in `submesh_facets`
    """
    if Version(dfx.__version__) < Version("0.10.0"):
        raise RuntimeError(
            "interpolate_to_submesh requires dolfinx version 0.10.0 or higher"
        )
    
    V_vol = u_volume.function_space
    mesh = V_vol.mesh

    V_surf = u_surface.function_space
    submesh = V_surf.mesh
    ip = V_surf.element.interpolation_points

    try:
        expr = dfx.fem.Expression(u_volume, ip, entity_maps=entity_maps)
    except TypeError:
        expr = dfx.fem.Expression(u_volume, ip)
    mesh.topology.create_connectivity(mesh.topology.dim, submesh.topology.dim)
    mesh.topology.create_connectivity(submesh.topology.dim, mesh.topology.dim)

    data = expr.eval(mesh, integration_entities)
    submesh.topology.create_entity_permutations()
    mesh.topology.create_entity_permutations()
    # ft = V_surf.element.basix_element.cell_type
    # if Version(dfx.__version__) < Version("0.11.0.dev0"):
    #     V_vol = u_volume.function_space
    #     mesh = V_vol.mesh
    #     # Before the introduction of https://github.com/FEniCS/dolfinx/pull/4140
    #     # one needed to permute the data according to the facet permutations.
    #     cell_info = mesh.topology.get_cell_permutation_info()
    #     for i in range(integration_entities.shape[0]):
    #         perm = np.arange(data.shape[1], dtype=np.int32)
    #         V_vol.element.basix_element.permute_subentity_closure_inv(
    #             perm,
    #             cell_info[integration_entities[i, 0]],
    #             ft,
    #             int(integration_entities[i, 1]),
    #         )
    #         data[i] = data[i][perm]
    if len(data.shape) == 3:
        # Data is now (num_cells, value_size,num_points)
        data = data.swapaxes(1, 2)
        # Data is now (value_size, num_cells, num_points)
        data = data.swapaxes(0, 1)

    if expr.value_size == 1:
        shaped_data = data.flatten()
    else:
        shaped_data = data.reshape(expr.value_size, -1)

    if hasattr(u_surface._cpp_object, "interpolate_f"):
        interpolate_func = u_surface._cpp_object.interpolate_f
    else:
        interpolate_func = u_surface._cpp_object.interpolate

    interpolate_func(shaped_data, submesh_facets)
    u_surface.x.scatter_forward()
