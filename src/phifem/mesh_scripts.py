from __future__ import annotations

import os
import typing
import warnings
from collections.abc import Callable
from os import PathLike

import dolfinx as dfx
import numpy as np
import numpy.typing as npt
import ufl  # type: ignore
from basix.ufl import _ElementBase, element
from dolfinx.cpp.graph import AdjacencyList_int32  # type: ignore
from dolfinx.fem import Expression, Function, FunctionSpace
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import Mesh, MeshTags
from ufl import inner

PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

debug_mode = False
if "MODE" in os.environ and os.environ["MODE"] == "debug":
    debug_mode = True


# Borrowed from: https://github.com/scientificcomputing/scifem/blob/main/src/scifem/mesh.py#L173
def _reverse_mark_entities(
    entity_map: dfx.common.IndexMap, entities: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """Communicate entities marked on a single process to all processes that ghosts or owns this entity.

    Args:
        entity_map: Index-map describing entity ownership
        entities: Local indices of entities to communicate
    Returns:
        Local indices marked on any process sharing this entity
    """
    comm_vec = dfx.la.vector(entity_map, dtype=np.int32)
    comm_vec.array[:] = 0
    comm_vec.array[entities] = 1
    comm_vec.scatter_reverse(dfx.la.InsertMode.add)
    comm_vec.scatter_forward()
    return np.flatnonzero(comm_vec.array).astype(np.int32)


# Borrowed from: https://github.com/scientificcomputing/scifem/blob/main/src/scifem/mesh.py#L57
def _get_entity_map(
    entity_map: dfx.mesh.EntityMap | npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Get an entity map from the sub-topology to the topology.

    This function handles both the deprecated construction of an entity map as a numpy array and the newer `EntityMap` class from `dolfinx.mesh`.

    Args:
        entity_map: An `EntityMap` object or a numpy array representing the mapping.
    Returns:
        Mapped indices of entities.
    """
    try:
        sub_top = entity_map.sub_topology
        assert isinstance(sub_top, dfx.mesh.Topology)
        sub_map = sub_top.index_map(entity_map.dim)
        indices = np.arange(sub_map.size_local + sub_map.num_ghosts, dtype=np.int32)
        return entity_map.sub_topology_to_topology(indices, inverse=False)
    except AttributeError:
        return entity_map


# Borrowed from: https://github.com/scientificcomputing/scifem/blob/main/src/scifem/mesh.py#L287
def _compute_subdomain_exterior_facets(
    mesh: Mesh, ct: MeshTags, markers: typing.Sequence[int]
) -> npt.NDArray[np.int32]:
    """Find the the facets that are considered to be on the "exterior" boundary of a subdomain.

    The subdomain is defined as the collection of cells in ``ct`` that is marked with any of the
    ``markers``. The exterior boundary of the subdomain is defined as the collection of facets
    that are only connected to a single cell within the subdomain.

    Note:
        Ghosted facets are included in the resulting array.

    Args:
        mesh: Mesh to extract subdomains from
        ct: MeshTags object marking subdomains
        markers: The tags making up the "new" mesh
    Returns:
        The exterior facets
    """
    # Create submesh to find the exterior facet of subdomain
    # Accumulate all entities, including ghosts, for the specfic set of tagged entities
    edim = ct.dim
    mesh.topology.create_connectivity(edim, mesh.topology.dim)
    tags_as_arr = np.asarray(markers, dtype=ct.values.dtype)
    all_tagged_indices = np.isin(ct.values, tags_as_arr)
    entities = ct.indices[all_tagged_indices]
    sub_mesh, cell_map = dfx.mesh.create_submesh(
        mesh,
        edim,
        entities,
    )[:2]

    new_et = _transfer_tags(ct, sub_mesh, cell_map, mesh)
    new_et.name = ct.name

    sub_mesh.topology.create_connectivity(
        sub_mesh.topology.dim - 1, sub_mesh.topology.dim
    )
    sub_facets = dfx.mesh.exterior_facet_indices(sub_mesh.topology)

    # Map exterior facet to (submesh_cell, local_facet_index) tuples
    try:
        integration_entities = dfx.fem.compute_integration_domains(
            dfx.fem.IntegralType.exterior_facet, sub_mesh.topology, sub_facets
        )
    except TypeError:
        integration_entities = dfx.fem.compute_integration_domains(
            dfx.fem.IntegralType.exterior_facet,
            sub_mesh.topology,
            sub_facets,
            sub_mesh.topology.dim - 1,
        )
    integration_entities = integration_entities.reshape(-1, 2)
    submap_array = _get_entity_map(cell_map)
    integration_entities[:, 0] = submap_array[integration_entities[:, 0]]

    # Get cell to facet connectivity (parent mesh)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    num_facets_per_cell = dfx.cpp.mesh.cell_num_entities(
        mesh.topology.cell_type, mesh.topology.dim - 1
    )
    c_to_f = mesh.topology.connectivity(
        mesh.topology.dim, mesh.topology.dim - 1
    ).array.reshape(-1, num_facets_per_cell)
    # Map (parent_cell, local_facet_index) to facet index (local to process)
    parent_facets = c_to_f[integration_entities[:, 0], integration_entities[:, 1]]
    facet_map = mesh.topology.index_map(mesh.topology.dim - 1)
    # Accumulate ghost facets
    return _reverse_mark_entities(facet_map, parent_facets)


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
    mesh: Mesh, discrete_levelset: Function, detection_measure: ufl.Measure
):
    """Computes the detection vector used to discriminate inside from cut from outside cells.

    Args:
        mesh: the mesh on which the detection is performed.
        discrete_levelset: the levelset used for the detection.
        detection_measure: the integration measure used to evaluate the levelset on the cells.

    Return: the detection vector as a numpy array.
    """
    # We localize at each cell via a DG0 test function.
    dg_0_element = element("DG", mesh.topology.cell_name(), 0)
    dg_0_space = dfx.fem.functionspace(mesh, dg_0_element)
    v0 = ufl.TestFunction(dg_0_space)

    # Assemble the numerator of detection
    detection_num = inner(discrete_levelset, v0) * detection_measure
    detection_num_form = dfx.fem.form(detection_num)
    detection_num_vec = assemble_vector(detection_num_form)
    # Assemble the denominator of detection
    detection_denom = inner(abs(discrete_levelset), v0) * detection_measure
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
    if np.any(np.isclose(detection_denom_vec.array, 0.0)):
        warnings.warn(
            "The detection function is zero everywhere on a cell. We mark it as 'cut' but this can be incorrect and should be carefully checked.",
            RuntimeWarning,
        )
    return detection_vector


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
    f2c_map = _reshape_map(f2c_connect)[0]

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


def _reshape_map(connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
    """Reshape the connected entities mapping. The reshaped mapping cannot be used to deduce the number of neighbors.

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


def _transfer_tags(
    source_mesh_tags: MeshTags,
    dest_mesh: Mesh,
    cmap: npt.NDArray,
    source_mesh: Mesh = None,
) -> MeshTags:
    """Given entities tags (cells or facets) from a source mesh, a destination mesh and the source mesh-destination mesh cells mapping, transfers the entities tags to the destination mesh.

    Args:
        source_mesh_tags: the tags on the source mesh.
        dest_mesh: the destination mesh.
        cmap: the source mesh-destination mesh cells mapping.
        source_mesh: the source mesh mandatory to transfer facets tags.

    Returns:
        Cells tags on the destination mesh.
    """
    cdim = dest_mesh.topology.dim
    fdim = cdim - 1
    edim = source_mesh_tags.dim

    if edim == cdim:
        emap = cmap
    elif edim == fdim:
        if source_mesh is None:
            raise ValueError("You must pass a source_mesh to transfer facets tags.")

        source_mesh.topology.create_connectivity(cdim, fdim)
        c2f_connect = source_mesh.topology.connectivity(cdim, fdim)
        num_facets_per_cell = len(c2f_connect.links(0))
        source_c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        dest_mesh.topology.create_connectivity(cdim, fdim)
        c2f_connect = dest_mesh.topology.connectivity(cdim, fdim)
        num_facets_per_cell = len(c2f_connect.links(0))
        dest_c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        source_c2f_dest_map = source_c2f_map[cmap]
        source_c2f_dest_map = source_c2f_dest_map.reshape(
            -1,
        )
        dest_c2f_map = dest_c2f_map.reshape(
            -1,
        )
        sorted_indices = np.unique(dest_c2f_map, return_index=True)[1]
        emap = source_c2f_dest_map[sorted_indices]
    else:
        raise ValueError("The source_mesh_tags can only be cells tags or facets tags.")

    # TODO: change this line to allow parallel computing
    source_tags = source_mesh_tags.values

    dest_entities = np.arange(len(emap))
    dest_tags = source_tags[emap]

    dest_entities_indices = np.hstack(dest_entities).astype(np.int32)
    dest_entities_markers = np.hstack(dest_tags).astype(np.int32)
    sorted_indices = np.argsort(dest_entities_indices)

    dest_entities_tags = dfx.mesh.meshtags(
        dest_mesh,
        edim,
        dest_entities_indices[sorted_indices],
        dest_entities_markers[sorted_indices],
    )

    return dest_entities_tags


def _tag_cells(
    mesh: Mesh,
    levelset_expression: Expression,
    single_layer_cut: bool = False,
) -> MeshTags:
    """Tag the mesh cells by computing detection = Σ f(dof)/Σ|f(dof)| where 'dof' are coming from a custom quadrature rule with points on the boundary of the cell only.
        Strictly inside cell  => tag 1
        Cut cell              => tag 2
        Strictly outside cell => tag 3

    Args:
        mesh: the background mesh.
        discrete_levelset: the discretization of the levelset.
        detection_degree: the degree of the custom quadrature rule used to detect cut entities.
        single_layer_cut: boolean, if True force a single layer of cut cells.

    Returns:
        The cells tags as a MeshTags object.
    """
    cdim = mesh.topology.dim
    all_cells = dfx.mesh.locate_entities(
        mesh, cdim, lambda x: np.ones_like(x[0]).astype(bool)
    )
    levelset_eval = levelset_expression.eval(mesh, all_cells)

    exterior = np.min(levelset_eval, axis=1) > 0.0
    interior = np.max(levelset_eval, axis=1) < 0.0
    cut = np.logical_not(np.logical_or(exterior, interior))

    exterior_indices = np.where(exterior)[0]
    interior_indices = np.where(interior)[0]
    cut_indices = np.where(cut)[0]

    if single_layer_cut:
        vdim = 0
        # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
        mesh.topology.create_connectivity(cdim, vdim)
        c2v_connect = mesh.topology.connectivity(cdim, vdim)
        num_vertices_per_cell = len(c2v_connect.links(0))
        c2v_map = np.reshape(c2v_connect.array, (-1, num_vertices_per_cell))

        mesh.topology.create_connectivity(vdim, cdim)
        v2c_connect = mesh.topology.connectivity(vdim, cdim)
        v2c_map, max_offset = _reshape_map(v2c_connect)

        neighbor_cells = np.reshape(
            v2c_map[c2v_map[cut_indices]], (-1, num_vertices_per_cell * max_offset)
        )
        mask_connected_cut_cells = np.any(
            np.isin(neighbor_cells, interior_indices), axis=1
        )
        isolated_cut_cells = cut_indices[~mask_connected_cut_cells]
        cut_indices = np.setdiff1d(cut_indices, isolated_cut_cells)
        exterior_indices = np.union1d(exterior_indices, isolated_cut_cells)

    if debug_mode:
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


def _cells_facets_pairs(f2c_map, c2f_map, facets):
    """Get integration entities cells-facets pairs for the corresponding facets indices.

    Args:
        f2c_map: the facet to cell connectivity mapping.
        c2f_map: the cell to facet connectivity mapping.
        facets: the facets indices to get the pairs from.

    Returns: the local indices of facets in their corresponding cells ordered as [cell_1 local_facet cell_2 local_facet cell_3 local_facet ...]
    """
    connected_cells = f2c_map[facets][:, 0]
    facets_connected_cells = c2f_map[connected_cells]
    facets_tiled = np.tile(facets[..., None], facets_connected_cells.shape[1])
    mask = facets_tiled == facets_connected_cells
    local_indices = np.where(mask)[1]
    pairs = np.ravel([connected_cells.T, local_indices.T], "F")
    return pairs


def _tag_facets(
    mesh: Mesh,
    levelset_expression_facets: Expression,
    cells_tags: MeshTags | None = None,
    levelset_expression_cells: Expression | None = None,
    single_layer_cut: bool = False,
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
    if cells_tags is None:
        assert levelset_expression_cells is not None, (
            "You must either pass cells_tags or a levelset expression over the cells of the domain."
        )
        cells_tags = _tag_cells(
            mesh, levelset_expression_cells, single_layer_cut=single_layer_cut
        )

    cdim = mesh.topology.dim
    fdim = cdim - 1
    mesh.topology.create_connectivity(fdim, cdim)
    f2c_connect = mesh.topology.connectivity(fdim, cdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    f2c_map = _reshape_map(f2c_connect)[0]
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))

    marker = lambda x: np.ones_like(x[0]).astype(bool)
    all_facets = dfx.mesh.locate_entities(mesh, fdim, marker)
    boundary_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, marker)
    interior_facets = np.setdiff1d(all_facets, boundary_facets)

    cell_facet_pairs = _cells_facets_pairs(f2c_map, c2f_map, interior_facets)
    levelset_eval_int = levelset_expression_facets.eval(mesh, cell_facet_pairs)

    exterior_int = np.min(levelset_eval_int, axis=1) > 0.0
    interior_int = np.max(levelset_eval_int, axis=1) < 0.0
    direct_int = np.isclose(np.sum(np.abs(levelset_eval_int), axis=1), 0.0)

    connected_cells = f2c_map[boundary_facets][:, 0]

    cell_facet_pairs = _compute_integration_entities(
        mesh, connected_cells, boundary_facets, 0
    )[0][1]
    levelset_eval_bdy = levelset_expression_facets.eval(mesh, cell_facet_pairs)
    exterior_bdy = np.min(levelset_eval_bdy, axis=1) > 0.0
    interior_bdy = np.max(levelset_eval_bdy, axis=1) < 0.0
    direct_bdy = np.isclose(np.sum(np.abs(levelset_eval_bdy), axis=1), 0.0)

    facets_indices = np.hstack([interior_facets, boundary_facets])
    exterior = np.hstack([exterior_int, exterior_bdy])
    interior = np.hstack([interior_int, interior_bdy])
    direct = np.hstack([direct_int, direct_bdy])
    exterior_indices = facets_indices[exterior]
    interior_indices = facets_indices[interior]
    direct_indices = facets_indices[direct]

    cut = np.logical_not(np.logical_or(exterior, interior))
    cut_indices = facets_indices[cut]
    cut_indices = np.setdiff1d(cut_indices, direct_indices)

    # Compute the list of facets on the boundary of the union of cut cells
    boundary_cut_indices = _compute_subdomain_exterior_facets(mesh, cells_tags, [2])
    boundary_exterior_indices = np.intersect1d(boundary_cut_indices, interior_indices)
    boundary_exterior_indices = np.setdiff1d(boundary_exterior_indices, direct_indices)
    interior_indices = np.setdiff1d(interior_indices, boundary_exterior_indices)

    boundary_interior_indices = np.intersect1d(boundary_cut_indices, exterior_indices)
    boundary_interior_indices = np.setdiff1d(boundary_interior_indices, direct_indices)
    exterior_indices = np.setdiff1d(exterior_indices, boundary_interior_indices)

    # Only exterior_facets might be empty
    if debug_mode:
        if len(interior_indices) == 0:
            raise ValueError("No interior facets (1)!")
        if len(cut_indices) == 0:
            print("WARNING: no cut facet computed in the partition.")
        if len(boundary_interior_indices) == 0:
            raise ValueError("No boundary facets (4)!")

        # The lists must not intersect
        names = ["interior facets (1)", "cut facets (2)", "boundary facets (4)"]
        for i, facets_list_1 in enumerate(
            [interior_indices, cut_indices, boundary_interior_indices]
        ):
            for j, facets_list_2 in enumerate(
                [interior_indices, cut_indices, boundary_interior_indices]
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
            exterior_indices,
            interior_indices,
            boundary_exterior_indices,
            cut_indices,
            boundary_interior_indices,
            direct_indices,
        ]
    ).astype(np.int32)
    interior_marker = np.full_like(interior_indices, 1).astype(np.int32)
    cut_marker = np.full_like(cut_indices, 2).astype(np.int32)
    interior_boundary_marker = np.full_like(boundary_exterior_indices, 3).astype(
        np.int32
    )
    boundary_marker = np.full_like(boundary_interior_indices, 4).astype(np.int32)
    exterior_marker = np.full_like(exterior_indices, 5).astype(np.int32)
    direct_interface_marker = np.full_like(direct_indices, 6).astype(np.int32)
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


def _overwrite_tags(mesh, tags_to_overwrite, new_tags):
    stack_indices = np.hstack([new_tags.indices, tags_to_overwrite.indices])
    stack_values = np.hstack([new_tags.values, tags_to_overwrite.values])
    overwritten_indices, ind = np.unique(stack_indices, return_index=True)
    overwritten_values = stack_values[ind]

    overwritten_tags = dfx.mesh.meshtags(
        mesh, tags_to_overwrite.dim, overwritten_indices, overwritten_values
    )
    return overwritten_tags


def _compute_codim_interpolation_points(
    space: dfx.fem.FunctionSpace,
) -> np.ndarray[np.float64]:
    """Compute the same FunctionSpace but defined over a submesh of codim 1.

    Args:
        space: the original space.

    Return: a new space defined on a mesh of codim 1.
    """
    family_name = space.ufl_element().basix_element.family.name
    degree = space.ufl_element().basix_element.degree
    tdim = space.mesh.topology.dim
    codim_mesh = dfx.mesh.create_submesh(space.mesh, tdim - 1, np.array([0.0]))[0]
    codim_cell_name = codim_mesh.topology.cell_name()
    new_elmt = element(family_name, codim_cell_name, degree)
    new_space = dfx.fem.functionspace(codim_mesh, new_elmt)
    return new_space.element.interpolation_points()


def _levelset_expression(
    mesh: dfx.mesh.Mesh,
    levelset: Function | Callable,
    interpolation_points: np.NDarray[np.float64],
) -> Expression:
    """Sanitize the levelset input by turning it into an Expression.

    Args:
        levelset: the levelset function.
        detection_space: the detection space.

    Return: the levelset as an Expression object.
    """

    # Sanitize levelset input
    try:
        # Test if levelset is a dolfinx.fem.Function
        _ = levelset.function_space
        levelset_expression = dfx.fem.Expression(
            levelset, interpolation_points, comm=mesh.comm
        )
    except AttributeError:
        try:
            x = ufl.SpatialCoordinate(mesh)
            levelset_expression = dfx.fem.Expression(
                levelset(x), interpolation_points, comm=mesh.comm
            )
        except TypeError:
            print(
                "Invalid levelset type: must be either a dolfinx.fem.Function or a UFL based Callable."
            )
    return levelset_expression


def compute_tags_measures(
    mesh: Mesh,
    levelset: Function | Callable,
    detection_space: FunctionSpace,
    box_mode: bool = False,
    single_layer_cut: bool = False,
    overwrite_tags: dict[str, MeshTags] | None = None,
) -> tuple[
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
    interpolation_points = detection_space.element.interpolation_points()
    levelset_expression = _levelset_expression(mesh, levelset, interpolation_points)

    cells_tags = _tag_cells(
        mesh, levelset_expression, single_layer_cut=single_layer_cut
    )

    codim_interpolation_points = _compute_codim_interpolation_points(detection_space)
    levelset_expression_facets = _levelset_expression(
        mesh, levelset, codim_interpolation_points
    )

    facets_tags = _tag_facets(
        mesh,
        levelset_expression_facets,
        cells_tags,
        levelset_expression,
        single_layer_cut=single_layer_cut,
    )

    if overwrite_tags is not None:
        if "cells" in overwrite_tags:
            ow_cells_tags = overwrite_tags["cells"]
            if np.any(np.isin([1, 2, 3], ow_cells_tags.values)):
                raise ValueError("Cannot overwrite cells tags with values 1, 2 or 3.")
            cells_tags = _overwrite_tags(mesh, cells_tags, ow_cells_tags)
        if "facets" in overwrite_tags:
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

        cells_tags = _transfer_tags(cells_tags, submesh, c_map)
        facets_tags = _transfer_tags(facets_tags, submesh, c_map, source_mesh=mesh)
        boundaries_measure = ufl.Measure("ds", domain=submesh)
        submesh_maps = [c_map, v_map, n_map]

    return (
        cells_tags,
        facets_tags,
        submesh,
        boundaries_measure,
        submesh_maps,
    )
