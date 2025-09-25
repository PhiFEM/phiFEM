import os
import warnings
from collections.abc import Callable
from os import PathLike
from typing import Any, Tuple

import dolfinx as dfx
import numpy as np
import numpy.typing as npt
import ufl  # type: ignore
from basix.ufl import element
from dolfinx.cpp.graph import AdjacencyList_int32  # type: ignore
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import Mesh, MeshTags
from ufl import inner

PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def _reference_segment_points(N: int) -> npt.NDArray[np.float64]:
    """Generate quadrature points on the reference segment.

    Args:
        N: int, N + 1 is the number of points on the segment.

    Returns: A numpy array (2, N + 1) that contains the coordinates of the quadrature points.
    """
    if N > 0:
        points = np.linspace(0, 1, N + 1).astype(np.float64)
    else:
        points = np.array([0.5]).astype(np.float64)
    return np.atleast_2d(points).T


def _reference_triangle_boundary_points(N: int) -> npt.NDArray[np.float64]:
    """Generate boundary quadrature points on the reference triangle cell.

    Args:
        N: int the number of points on each edge (if N=0, there is only one point at the center of the cell).

    Returns: A numpy array (2, 3N) that contains the coordinates of the quadrature points.
    """
    if N > 0:
        t1 = np.linspace(0, 1, N + 1)
        edge1 = np.stack((t1, np.zeros_like(t1)), axis=-1).astype(np.float64)
        t2 = t1[1:]
        edge2 = np.stack((1 - t2, t2), axis=-1).astype(np.float64)
        t3 = t1[1:-1]
        edge3 = np.stack((np.zeros_like(t3), 1 - t3), axis=-1).astype(np.float64)

        if N > 1:
            points = np.concatenate((edge1, edge2, edge3), axis=0)
        else:
            points = np.concatenate((edge1, edge2), axis=0)
    else:
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]]).astype(np.float64)
    return points


def _reference_square_boundary_points(N: int) -> npt.NDArray[np.float64]:
    """Generate boundary quadrature points on the reference square cell.

    Args:
        N: int the number of points on each edge (if N=0, there is only one point at the center of the cell).

    Returns: A numpy array (2, 4N) that contains the coordinates of the quadrature points.
    """
    if N > 0:
        t1 = np.linspace(0, 1, N + 1)
        edge1 = np.stack((t1, np.zeros_like(t1)), axis=-1).astype(np.float64)
        t2 = t1[1:]
        edge2 = np.stack((np.ones_like(t2), t2), axis=-1).astype(np.float64)
        t3 = t1[1:]
        edge3 = np.stack((1.0 - t3, np.ones_like(t3)), axis=-1).astype(np.float64)
        t4 = t1[1:-1]
        edge4 = np.stack((np.zeros_like(t4), 1.0 - t4), axis=-1).astype(np.float64)

        if N > 1:
            points = np.concatenate((edge1, edge2, edge3, edge4), axis=0)
        else:
            points = np.concatenate((edge1, edge2, edge3), axis=0)
    else:
        points = np.array([[1.0 / 2.0, 1.0 / 2.0]]).astype(np.float64)
    return points


def _compute_detection_vector(
    discrete_levelset: Function, detection_measure: ufl.Measure
):
    mesh = discrete_levelset.function_space.mesh
    # We localize at each cell via a DG0 test function.
    dg_0_element = element("DG", mesh.topology.cell_name(), 0)
    dg_0_space = dfx.fem.functionspace(mesh, dg_0_element)
    v0 = ufl.TestFunction(dg_0_space)

    # Assemble the numerator of detection
    detection_num = inner(discrete_levelset, v0) * detection_measure
    detection_num_form = dfx.fem.form(detection_num)
    detection_num_vec = assemble_vector(detection_num_form)
    # Assemble the denominator of detection
    detection_denom = inner(ufl.algebra.Abs(discrete_levelset), v0) * detection_measure
    detection_denom_form = dfx.fem.form(detection_denom)
    detection_denom_vec = assemble_vector(detection_denom_form)

    # detection_denom_vec is not supposed to be zero, this would mean that the levelset is zero at all dofs in a cell.
    # However, in practice it can happen that for a very small cut triangle, detection_denom_vec is of the order of the machine precision.
    # In this case, we set the value of detection_vector to 0.5, meaning we consider the cell as cut.
    mask = np.where(detection_denom_vec.array > 0.0)
    detection_vector = np.full_like(detection_num_vec.array, 0.5)
    detection_vector[mask] = (
        detection_num_vec.array[mask] / detection_denom_vec.array[mask]
    )
    if np.any(detection_denom_vec.array == 0.0):
        warnings.warn(
            "The detection function is zero everywhere on a cell. We mark it as 'cut' but this can be incorrect and should be carefully checked.",
            RuntimeWarning,
        )
    return detection_vector


def _one_sided_edge_measure(
    mesh: Mesh, integration_cells: list[int], integration_facets: list[int], ind: int
) -> ufl.Measure:
    """Compute a one-sided integral over a set of given edges. This script is inspired from https://github.com/jorgensd/dolfinx-tutorial/issues/158.

    Args:
        mesh: the mesh on which we compute the measure.
        integration_cells: list of cells indices from which the integral is computed.
        integration_facets: list of facets indices on which the integral is computed.
        ind: index used in the measure.
    Returns:
        measure: the integration measure of the one-sided integral.
    """
    cdim = mesh.topology.dim
    fdim = cdim - 1
    mesh.topology.create_connectivity(fdim, cdim)
    f2c_connect = mesh.topology.connectivity(fdim, cdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    f2c_map = _reshape_facets_map(f2c_connect)

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

    # We compute the one-sided measure
    measure = ufl.Measure(
        "ds", domain=mesh, subdomain_data=[(ind, integration_entities)]
    )
    return measure(ind)


def _reshape_facets_map(f2c_connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
    """Reshape the facets-to-cells indices mapping.

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


def _transfer_cells_tags(
    source_mesh_cells_tags: MeshTags, dest_mesh: Mesh, cmap: npt.NDArray[Any]
) -> MeshTags:
    """Given a cells tags from a source mesh, a destination mesh and the source mesh-destination mesh cells mapping, transfers the cells tags to the destination mesh.

    Args:
        source_mesh_cells_tags: the cells tags on the source mesh.
        dest_mesh: the destination mesh.
        cmap: the source mesh-destination mesh cells mapping.

    Returns:
        Cells tags on the destination mesh.
    """

    cdim = dest_mesh.topology.dim
    # TODO: change this line to allow parallel computing
    tag_values = np.unique(source_mesh_cells_tags.values)

    list_dest_cells = []
    list_markers = []
    for value in tag_values:
        source_cells = source_mesh_cells_tags.find(value)
        mask = np.isin(cmap, source_cells)
        dest_mesh_masked = np.where(mask)[0]
        list_dest_cells.append(dest_mesh_masked)
        list_markers.append(np.full_like(dest_mesh_masked, value))

    dest_cells_indices = np.hstack(list_dest_cells).astype(np.int32)
    dest_cells_markers = np.hstack(list_markers).astype(np.int32)
    sorted_indices = np.argsort(dest_cells_indices)

    dest_cells_tags = dfx.mesh.meshtags(
        dest_mesh,
        cdim,
        dest_cells_indices[sorted_indices],
        dest_cells_markers[sorted_indices],
    )

    return dest_cells_tags


def _tag_cells(
    mesh: Mesh, discrete_levelset: Function, detection_degree: int
) -> MeshTags:
    """Tag the mesh cells by computing detection = Σ f(dof)/Σ|f(dof)| where 'dof' are coming from a custom quadrature rule with points on the boundary of the cell only.
        Strictly inside cell  => tag 1
        Cut cell              => tag 2
        Strictly outside cell => tag 3

    Args:
        mesh: the background mesh.
        discrete_levelset: the discretization of the levelset.
        detection_degree: the degree of the custom quadrature rule used to detect cut entities.

    Returns:
        The cells tags as a MeshTags object.
    """

    # Create the custom quadrature rule.
    # The quadrature points are evenly spaced on the boundary of the reference cell.
    # The weights are 1.
    cell_type = mesh.topology.cell_type.name

    if cell_type == "triangle":
        points = _reference_triangle_boundary_points(detection_degree)
    elif cell_type == "quadrilateral":
        points = _reference_square_boundary_points(detection_degree)
    else:
        raise NotImplementedError(
            "Mesh tags computation does not support other cell types than 'triangle' or 'quadrilateral'"
        )
    weights = np.ones_like(points[:, 0])

    detection_quadrature = {
        "quadrature_rule": "custom",
        "quadrature_points": points,
        "quadrature_weights": weights,
    }

    detection_measure = ufl.Measure("dx", domain=mesh, metadata=detection_quadrature)

    detection_vector = _compute_detection_vector(discrete_levelset, detection_measure)

    cut_indices = np.where(
        np.logical_and(detection_vector > -1.0, detection_vector < 1.0)
    )[0]

    exterior_indices = np.where(detection_vector == 1.0)[0]
    interior_indices = np.where(detection_vector == -1.0)[0]

    if not os.environ["MODE"] == "debug":
        if len(interior_indices) == 0:
            raise ValueError("No interior cells (1)!")
        if len(cut_indices) == 0:
            print("WARNING: no cut cells computed in the partition.")

        assert np.logical_not(np.isin(exterior_indices, cut_indices).any()), (
            "The sets of outside cells and cut cells have a non-empty intersection"
        )
        assert np.logical_not(np.isin(interior_indices, cut_indices).any()), (
            "The sets of inside cells and cut cells have a non-empty intersection"
        )
        assert np.logical_not(np.isin(exterior_indices, interior_indices).any()), (
            "The sets of outside cells and inside cells have a non-empty intersection"
        )

    # Create the meshtags from the indices.
    indices = np.hstack([exterior_indices, interior_indices, cut_indices]).astype(
        np.int32
    )
    interior_marker = np.full_like(interior_indices, 1).astype(np.int32)
    exterior_marker = np.full_like(exterior_indices, 3).astype(np.int32)
    cut_marker = np.full_like(cut_indices, 2).astype(np.int32)
    markers = np.hstack([exterior_marker, interior_marker, cut_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    cells_tags = dfx.mesh.meshtags(
        mesh, mesh.topology.dim, indices[sorted_indices], markers[sorted_indices]
    )

    return cells_tags


def _tag_facets(
    mesh: Mesh, cells_tags: MeshTags, discrete_levelset: Function, detection_degree: int
) -> MeshTags:
    """Tag the mesh facets.
    Strictly interior facets  => tag 1
    Cut facets                => tag 2
    Interior boundary facets  => tag 3
    Boundary facets (Gamma_h) => tag 4
    Strictly exterior facets  => tag 5
    Direct interface facets   => tag 6

    Args:
        mesh: the background mesh.
        cells_tags: the MeshTags object containing cells tags.
        discrete_levelset: the discretization of the levelset.
        detection_degree: the degree of the custom quadrature rule used to detect cut entities.

    Returns:
        The facets tags as a MeshTags object.
    """
    cdim = mesh.topology.dim
    fdim = cdim - 1
    # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
    mesh.topology.create_connectivity(cdim, fdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))

    # Get tagged cells
    interior_cells = cells_tags.find(1)
    cut_cells = cells_tags.find(2)
    exterior_cells = cells_tags.find(3)

    # Check which background mesh boundary facets are cut by the interface
    background_mesh_boundary_facets = dfx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.ones_like(x[0]).astype(bool)
    )

    points = _reference_segment_points(detection_degree)
    weights = np.ones_like(points[:, 0])

    detection_quadrature = {
        "quadrature_rule": "custom",
        "quadrature_points": points,
        "quadrature_weights": weights,
    }

    detection_measure = ufl.Measure("ds", domain=mesh, metadata=detection_quadrature)

    detection_vector = _compute_detection_vector(discrete_levelset, detection_measure)
    mask_cut_indices_cells = np.logical_and(
        detection_vector > -1.0, detection_vector < 1.0
    )
    cut_indices_cells = np.where(mask_cut_indices_cells)[0]
    comp_indices_cells = np.where(np.logical_not(mask_cut_indices_cells))[0]

    cut_boundary_facets = np.intersect1d(
        c2f_map[cut_indices_cells], background_mesh_boundary_facets
    )
    uncut_boundary_facets = np.intersect1d(
        c2f_map[comp_indices_cells], background_mesh_boundary_facets
    )
    uncut_boundary_facets = np.setdiff1d(uncut_boundary_facets, c2f_map[exterior_cells])
    uncut_boundary_facets = np.setdiff1d(uncut_boundary_facets, c2f_map[interior_cells])

    # Facets shared by an interior cell and a cut cell
    interior_boundary_facets = np.intersect1d(
        c2f_map[interior_cells], c2f_map[cut_cells]
    )

    # If there is no exterior_cells, the boundary facets are just the facets on the boundary of Ω_h
    if len(exterior_cells) == 0:
        boundary_facets = background_mesh_boundary_facets
    else:
        # Facets shared by an exterior cell and a cut cell
        boundary_facets = np.intersect1d(c2f_map[exterior_cells], c2f_map[cut_cells])
        boundary_facets = np.union1d(boundary_facets, uncut_boundary_facets)

    direct_interface_facets = np.intersect1d(
        c2f_map[exterior_cells], c2f_map[interior_cells]
    )
    # Cut facets F_h^Γ
    facets_to_remove = np.union1d(boundary_facets, interior_boundary_facets)
    facets_to_remove = np.union1d(facets_to_remove, direct_interface_facets)
    facets_to_remove = np.union1d(facets_to_remove, uncut_boundary_facets)
    cut_facets = np.setdiff1d(c2f_map[cut_cells], facets_to_remove)
    cut_facets = np.union1d(cut_facets, cut_boundary_facets)

    # Interior facets
    facets_to_remove = np.union1d(interior_boundary_facets, boundary_facets)
    facets_to_remove = np.union1d(facets_to_remove, direct_interface_facets)
    interior_facets = np.setdiff1d(c2f_map[interior_cells], facets_to_remove)

    # Exterior facets
    facets_to_remove = np.union1d(interior_boundary_facets, boundary_facets)
    facets_to_remove = np.union1d(facets_to_remove, direct_interface_facets)
    exterior_facets = np.setdiff1d(c2f_map[exterior_cells], facets_to_remove)

    boundary_facets = np.setdiff1d(boundary_facets, cut_facets)

    # Only exterior_facets might be empty
    if not os.environ["MODE"] == "debug":
        if len(interior_facets) == 0:
            raise ValueError("No interior facets (1)!")
        if len(cut_facets) == 0:
            print("WARNING: no cut facet computed in the partition.")
        if len(boundary_facets) == 0:
            raise ValueError("No boundary facets (4)!")

        # The lists must not intersect
        names = ["interior facets (1)", "cut facets (2)", "boundary facets (4)"]
        for i, facets_list_1 in enumerate(
            [interior_facets, cut_facets, boundary_facets]
        ):
            for j, facets_list_2 in enumerate(
                [interior_facets, cut_facets, boundary_facets]
            ):
                if i != j and len(np.intersect1d(facets_list_1, facets_list_2)) > 0:
                    raise ValueError(
                        names[i]
                        + " and "
                        + names[j]
                        + " have a non-empty intersection!"
                    )

    # Create the meshtags from the indices.
    indices = np.hstack(
        [
            exterior_facets,
            interior_facets,
            interior_boundary_facets,
            cut_facets,
            boundary_facets,
            direct_interface_facets,
        ]
    ).astype(np.int32)
    interior_marker = np.full_like(interior_facets, 1).astype(np.int32)
    cut_marker = np.full_like(cut_facets, 2).astype(np.int32)
    interior_boundary_marker = np.full_like(interior_boundary_facets, 3).astype(
        np.int32
    )
    boundary_marker = np.full_like(boundary_facets, 4).astype(np.int32)
    exterior_marker = np.full_like(exterior_facets, 5).astype(np.int32)
    direct_interface_marker = np.full_like(direct_interface_facets, 6).astype(np.int32)
    markers = np.hstack(
        [
            exterior_marker,
            interior_marker,
            interior_boundary_marker,
            cut_marker,
            boundary_marker,
            direct_interface_marker,
        ]
    ).astype(np.int32)
    sorted_indices = np.argsort(indices)

    facets_tags = dfx.mesh.meshtags(
        mesh, fdim, indices[sorted_indices], markers[sorted_indices]
    )

    return facets_tags


def compute_tags_measures(
    mesh: Mesh,
    detection_levelset: NDArrayFunction,
    detection_degree: int,
    box_mode: bool = False,
) -> Tuple[
    MeshTags,
    MeshTags,
    Mesh | None,
    ufl.Measure | None,
    ufl.Measure | None,
    list[npt.NDArray[np.int32]] | None,
]:
    """Compute the mesh (cells and facets) tags as well as the discrete boundary measures.

    Args:
        mesh: the mesh on which we compute the tags.
        levelset: the levelset function used to discriminate the cells.
        detection_degree: the degree of the piecewise-polynomial approximation to the levelset.
        box_mode: if False (default), create a submesh and return the cells tags on the submesh, if True, returns cells tags on the input mesh.

    Returns
        The mesh/submesh cells tags.
        The mesh/submesh facets tags.
        The mesh/submesh (input mesh if box_mode is True).
        The one-sided measure from inside.
        The one-sided measure from outside.
        Submesh c-map, v-map and n-map.
    """
    detection_element = element("Lagrange", mesh.topology.cell_name(), detection_degree)
    detection_space = dfx.fem.functionspace(mesh, detection_element)
    discrete_levelset = dfx.fem.Function(detection_space)
    discrete_levelset.interpolate(detection_levelset)

    cells_tags = _tag_cells(mesh, discrete_levelset, detection_degree)

    if box_mode:
        submesh = None
        facets_tags = _tag_facets(mesh, cells_tags, discrete_levelset, detection_degree)
        integration_cells = np.union1d(cells_tags.find(2), cells_tags.find(1))
        d_boundary_outside = _one_sided_edge_measure(
            mesh, integration_cells, facets_tags.find(4), 100
        )
        integration_cells = np.union1d(cells_tags.find(2), cells_tags.find(3))
        d_boundary_inside = _one_sided_edge_measure(
            mesh, integration_cells, facets_tags.find(3), 101
        )
        submesh_maps = None
    else:
        # We create the submesh
        omega_h_cells = np.unique(np.hstack([cells_tags.find(1), cells_tags.find(2)]))
        submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(
            mesh, mesh.topology.dim, omega_h_cells
        )  # type: ignore

        detection_space_submesh = dfx.fem.functionspace(submesh, detection_element)
        discrete_levelset_submesh = dfx.fem.Function(detection_space_submesh)
        discrete_levelset_submesh.interpolate(detection_levelset)

        cells_tags = _transfer_cells_tags(cells_tags, submesh, c_map)
        facets_tags = _tag_facets(
            submesh, cells_tags, discrete_levelset_submesh, detection_degree
        )
        d_boundary_outside = None
        d_boundary_inside = None
        submesh_maps = [c_map, v_map, n_map]

    return (
        cells_tags,
        facets_tags,
        submesh,
        d_boundary_outside,
        d_boundary_inside,
        submesh_maps,
    )
