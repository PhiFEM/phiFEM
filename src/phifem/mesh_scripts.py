from collections.abc import Callable
from os import PathLike
from typing import Tuple

import dolfinx as dfx
import numpy as np
import numpy.typing as npt
import ufl  # type: ignore
from dolfinx.fem import Function
from dolfinx.mesh import Mesh, MeshTags

from phifem import measures, tags

PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def compute_tags_measures(
    mesh: Mesh,
    discrete_levelset: Function,
    detection_degree: int,
    box_mode: bool = False,
    single_layer_cut: bool = False,
    overwrite_tags: dict[str, MeshTags] | dict = {},
) -> Tuple[
    MeshTags,
    MeshTags,
    Mesh | None,
    ufl.Measure,
    list[npt.NDArray[np.int32]] | None,
]:
    """Compute the mesh (cells and facets) tags as well as the discrete boundary measures.

    Args:
        mesh: the mesh on which we compute the tags.
        levelset: the levelset function used to discriminate the cells.
        detection_degree: the degree used in the custom quadrature rule of the detection form.
        box_mode: if False (default), create a submesh and return the cells tags on the submesh, if True, returns cells tags on the input mesh.
        single_layer_cut: boolean, if True force a single layer of cut cells.

    Returns
        The mesh/submesh cells tags.
        The mesh/submesh facets tags.
        The mesh/submesh (input mesh if box_mode is True).
        The boundaries measure.
        Submesh c-map, v-map and n-map.
    """
    cells_tags = _tag_cells(
        mesh, discrete_levelset, detection_degree, single_layer_cut=single_layer_cut
    )
    facets_tags = _tag_facets(mesh, cells_tags, discrete_levelset, detection_degree)

    if "cells" in overwrite_tags.keys():
        ow_cells_tags = overwrite_tags["cells"]
        if np.any(np.isin([1, 2, 3], ow_cells_tags.values)):
            raise ValueError("Cannot overwrite cells tags with values 1, 2 or 3.")
        cells_tags = _overwrite_tags(mesh, cells_tags, ow_cells_tags)
    if "facets" in overwrite_tags.keys():
        ow_facets_tags = overwrite_tags["facets"]
        if np.any(np.isin([1, 2, 3, 4, 5, 6, 100, 101], ow_facets_tags.values)):
            raise ValueError(
                "Cannot overwrite facets tags with values 1, 2, 3, 4, 5, 6, 100 or 101."
            )
        facets_tags = _overwrite_tags(mesh, facets_tags, ow_facets_tags)

    if box_mode:
        submesh = None
        integration_cells = np.union1d(cells_tags.find(2), cells_tags.find(1))
        integration_entities_outside = _compute_integration_entities(
            mesh, integration_cells, facets_tags.find(4), 100
        )
        integration_cells = np.union1d(cells_tags.find(2), cells_tags.find(3))
        integration_entities_inside = _compute_integration_entities(
            mesh, integration_cells, facets_tags.find(3), 101
        )
        combined_integration_entities = (
            integration_entities_outside + integration_entities_inside
        )

        boundaries_measure = ufl.Measure(
            "ds", domain=mesh, subdomain_data=combined_integration_entities
        )
        submesh_maps = None
    else:
        # We create the submesh
        omega_h_cells = np.unique(np.hstack([cells_tags.find(1), cells_tags.find(2)]))
        submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(
            mesh, mesh.topology.dim, omega_h_cells
        )  # type: ignore

        cells_tags = tags.transfer(cells_tags, submesh, c_map)
        facets_tags = tags.transfer(facets_tags, submesh, c_map, source_mesh=mesh)
        submesh_maps = [c_map, v_map, n_map]

    return (
        cells_tags,
        facets_tags,
        submesh,
        boundaries_measure,
        submesh_maps,
    )
