import numpy as np
import ufl
from dolfinx.mesh import Mesh

from phifem import quadratures as quad
from phifem.utils import reshape_map


def _compute_integration_entities(
    mesh: Mesh, integration_cells: list[int], integration_facets: list[int], ind: int
) -> ufl.Measure:
    """Compute the integration entities in order to build a one-sided integral over a set of given edges. This script is inspired from https://github.com/jorgensd/dolfinx-tutorial/issues/158.

    Args:
        mesh: the mesh on which we compute the measure.
        integration_cells: list of cells indices from which the integral is computed.
        integration_facets: list of facets indices on which the integral is computed.
        ind: index used in the measure.
    Returns: the integration entities.
    """
    cdim = mesh.topology.dim
    fdim = cdim - 1
    mesh.topology.create_connectivity(fdim, cdim)
    f2c_connect = mesh.topology.connectivity(fdim, cdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    f2c_map = reshape_map(f2c_connect)[0]

    # Omega_h^Gamma one-sided boundary integral
    connected_cells = f2c_map[integration_facets]
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))

    # We select the cut cells among the connected cells
    mask = np.isin(connected_cells, integration_cells)
    right_side_cells = np.reshape(
        connected_cells[mask], (connected_cells[mask].shape[0], 1)
    )

    # Removing duplicate cells while preserving the ordering
    right_side_cells = right_side_cells[
        np.sort(np.unique(right_side_cells, return_index=True)[1])
    ]

    # We compute the local indices of the integration facets connected to the cells
    facets_mask = np.isin(
        c2f_map[right_side_cells].reshape(
            right_side_cells.shape[0], num_facets_per_cell
        ),
        integration_facets,
    )
    local_indices = np.tile(np.arange(num_facets_per_cell), (facets_mask.shape[0], 1))
    local_indices[np.logical_not(facets_mask)] = -1

    # We repeat the cells indices if a cell has several facets in the integration_facets
    num_rep = (local_indices >= 0).astype(np.int32).sum(axis=1)
    right_side_cells_rep = np.repeat(right_side_cells, num_rep)
    local_indices = local_indices[np.where(local_indices != -1)]

    # We ravel the cells (global) indices and facets (local) indices in order to obtain something like: [cell_1, facet_1, cell_1, facet_2, cell_2, facet_1, cell_3, facet_1]
    integration_entities = np.ravel(
        np.column_stack((right_side_cells_rep, local_indices))
    ).astype(np.int32)

    return [(ind, integration_entities)]


def detection(mesh: Mesh, detection_degree: int, cell_type: str, integral_type: str):
    """
    Compute the detection measure. If not fallback, the quadrature points are evenly spaced on the boundary of the reference cell, the weights are all 1.

    Args:
        mesh: the mesh.
        detection_degree: the degree of the quadrature rule used in the detection measure.
        cell_type: the name of the cell type.
        integral_type: the type of integral for the measure.

    Returns:
        The detection measure as a ufl.Measure object.
    """
    fallback_quadrature = False
    if cell_type == "interval":
        points = quad.segment(detection_degree)
    elif cell_type == "triangle":
        points = quad.triangle(detection_degree)
    elif cell_type == "quadrilateral":
        points = quad.square(detection_degree)
    else:
        fallback_quadrature = True


    if fallback_quadrature:
        detection_quadrature = {
            "quadrature_rule": "default",
            "quadrature_degree": detection_degree,
        }
    else:
        weights = np.ones_like(points[:, 0])
        detection_quadrature = {
            "quadrature_rule": "custom",
            "quadrature_points": points,
            "quadrature_weights": weights,
        }

    detection_measure = ufl.Measure(
        integral_type, domain=mesh, metadata=detection_quadrature
    )
    return detection_measure


def one_sided_boundary(mesh, cells_tags, facets_tags, box_mode):
    if box_mode:
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
    else:
        combined_integration_entities = None

    boundaries_measure = ufl.Measure(
        "ds", domain=mesh, subdomain_data=combined_integration_entities
    )
    return boundaries_measure
