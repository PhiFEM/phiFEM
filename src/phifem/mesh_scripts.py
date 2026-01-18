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

debug_mode = False
if "MODE" in os.environ:
    if os.environ["MODE"] == "debug":
        debug_mode = True


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

    # We compute the one-sided measure
    measure = ufl.Measure(
        "ds", domain=mesh, subdomain_data=[(ind, integration_entities)]
    )
    return measure(ind)


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
    cmap: npt.NDArray[Any],
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
    edges_tags: MeshTags,
) -> MeshTags:
    """Tag the mesh cells.
        If in a cell:
            ∃ cut edge => Cut cell => tag 2
        else:
            ∃ inside edge => Strictly inside cell => tag 1
        else:
            ∃ outside edge => Strictly outside cell => tag 3

    Args:
        mesh: the background mesh.
        edges_tags: the mesh edges tags as a MeshTags object.

    Returns:
        The cells tags as a MeshTags object.
    """

    inside_edges = edges_tags.find(1)
    cut_edges = edges_tags.find(2)
    outside_edges = edges_tags.find(5)

    cdim = 2
    edim = 1

    # mesh.topology.create_connectivity(cdim, edim)
    mesh.topology.create_connectivity(cdim, edim)
    # e2c_connect = mesh.topology.connectivity(cdim, edim)
    # e2c_map = _reshape_facets_map(e2c_connect)
    c2e_connect = mesh.topology.connectivity(cdim, edim)

    num_e_per_c = len(c2e_connect.links(0))
    c2e_map = np.reshape(c2e_connect.array, (-1, num_e_per_c))

    mask_cut_edges = np.isin(c2e_map, cut_edges)
    mask_inside_edges = np.isin(c2e_map, inside_edges)
    mask_outside_edges = np.isin(c2e_map, outside_edges)

    mask_cut_cells = mask_cut_edges[:, 0] + mask_cut_edges[:, 1] + mask_cut_edges[:, 2]
    mask_inside_cells = (
        mask_inside_edges[:, 0] + mask_inside_edges[:, 1] + mask_inside_edges[:, 2]
    ) * np.logical_not(mask_cut_cells)
    mask_outside_cells = (
        (mask_outside_edges[:, 0] + mask_outside_edges[:, 1] + mask_outside_edges[:, 2])
        * np.logical_not(mask_cut_cells)
        * np.logical_not(mask_inside_cells)
    )
    cut_cells = np.where(mask_cut_cells)[0]
    inside_cells = np.where(mask_inside_cells)[0]
    outside_cells = np.where(mask_outside_cells)[0]

    if debug_mode:
        if len(inside_cells) == 0:
            raise ValueError("No interior cells (1)!")
        if len(cut_cells) == 0:
            print("WARNING: no cut cells computed in the partition.")

        assert np.logical_not(np.isin(outside_cells, cut_cells).any()), (
            "The sets of outside cells and cut cells have a non-empty intersection"
        )
        assert np.logical_not(np.isin(inside_cells, cut_cells).any()), (
            "The sets of inside cells and cut cells have a non-empty intersection"
        )
        assert np.logical_not(np.isin(outside_cells, inside_cells).any()), (
            "The sets of outside cells and inside cells have a non-empty intersection"
        )

    # # Create the meshtags from the indices.
    indices = np.hstack([outside_cells, inside_cells, cut_cells]).astype(np.int32)
    inside_marker = np.full_like(inside_cells, 1).astype(np.int32)
    outside_marker = np.full_like(outside_cells, 3).astype(np.int32)
    cut_marker = np.full_like(cut_cells, 2).astype(np.int32)
    markers = np.hstack([outside_marker, inside_marker, cut_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    cells_tags = dfx.mesh.meshtags(
        mesh, mesh.topology.dim, indices[sorted_indices], markers[sorted_indices]
    )

    return cells_tags


def _tag_edges(
    wireframe: Mesh,
    discrete_levelset: Function,
    detection_degree: int,
    single_cut_layer: bool = False,
) -> MeshTags:
    """Tag the mesh edges.
    Strictly inside edges    => tag 1
    Cut edges                => tag 2
    Strictly outside edges   => tag 5

    Args:
        wireframe: the wireframe of the background mesh.
        discrete_levelset: the discretization of the levelset on the wireframe.
        detection_degree: the degree of the custom quadrature rule used to detect cut entities.
        single_cut_layer: boolean, if True force cut edges to be connected to at least one interior edge.

    Returns:
        The edges tags as a MeshTags object.
    """
    points = _reference_segment_points(detection_degree)
    weights = np.ones_like(points[:, 0])

    detection_quadrature = {
        "quadrature_rule": "custom",
        "quadrature_points": points,
        "quadrature_weights": weights,
    }

    detection_measure = ufl.Measure(
        "dx", domain=wireframe, metadata=detection_quadrature
    )

    detection_vector = _compute_detection_vector(
        wireframe, discrete_levelset, detection_measure
    )

    outside_indices = np.where(detection_vector == 1.0)[0]
    inside_indices = np.where(detection_vector == -1.0)[0]

    cut_indices = np.where(
        np.logical_and(detection_vector > -1.0, detection_vector < 1.0)
    )[0]

    if single_cut_layer:
        edim = 1
        vdim = 0

        # Create the edges to vertices and vertices to edges mappings
        mesh.topology.create_connectivity(edim, vdim)
        mesh.topology.create_connectivity(vdim, edim)
        e2v_connect = mesh.topology.connectivity(edim, vdim)
        v2e_connect = mesh.topology.connectivity(vdim, edim)
        num_v_per_e = len(e2v_connect.links(0))
        e2v_map = np.reshape(e2v_connect.array, (-1, num_v_per_e))
        v2e_map, max_offset = _reshape_map(v2e_connect)

        # Create the edges to edges mapping
        neighbor_map = v2e_map[e2v_map[cut_indices]]
        neighbor_edges = np.reshape(neighbor_map, (-1, 2 * max_offset))

        # Mask telling if neighbor edges are inside edges
        mask_inside_neighbor = np.any(np.isin(neighbor_edges, inside_indices), axis=1)

        # Add the cut edges with no neighbor inside to the outside edges
        outside_indices = np.union1d(
            outside_indices, cut_indices[~mask_inside_neighbor]
        )
        # Remove the cut edges with no neighbor inside from cut edges
        cut_indices = cut_indices[mask_inside_neighbor]

    if debug_mode:
        if len(inside_indices) == 0:
            raise ValueError("No interior edges (1)!")
        if len(cut_indices) == 0:
            print("WARNING: no cut edges computed in the partition.")

        assert np.logical_not(np.isin(outside_indices, cut_indices).any()), (
            "The sets of outside edges and cut edges have a non-empty intersection"
        )
        assert np.logical_not(np.isin(inside_indices, cut_indices).any()), (
            "The sets of inside edges and cut edges have a non-empty intersection"
        )
        assert np.logical_not(np.isin(outside_indices, inside_indices).any()), (
            "The sets of outside edges and inside edges have a non-empty intersection"
        )

    # Create the meshtags from the indices.
    indices = np.hstack([outside_indices, inside_indices, cut_indices]).astype(np.int32)
    interior_marker = np.full_like(inside_indices, 1).astype(np.int32)
    exterior_marker = np.full_like(outside_indices, 5).astype(np.int32)
    cut_marker = np.full_like(cut_indices, 2).astype(np.int32)
    markers = np.hstack([exterior_marker, interior_marker, cut_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    edges_tags = dfx.mesh.meshtags(
        wireframe,
        wireframe.topology.dim,
        indices[sorted_indices],
        markers[sorted_indices],
    )

    return edges_tags


def _complete_edges_tags(
    mesh: Mesh,
    cells_tags: MeshTags,
    edges_tags: MeshTags,
) -> MeshTags:
    """Complete the edges tags.
    Inside boundary edges            => tag 3
    Outside boundary edges (Gamma_h) => tag 4
    Direct interface edges           => tag 6

    Args:
        mesh: the background mesh.
        cells_tags: the MeshTags object containing cells tags.
        edges_tags: the MeshTags object containing edges tags.

    Returns:
        The completed edges tags as a MeshTags object.
    """
    cdim = mesh.topology.dim
    edim = 1
    # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
    mesh.topology.create_connectivity(cdim, edim)
    c2e_connect = mesh.topology.connectivity(cdim, edim)
    num_e_per_c = len(c2e_connect.links(0))
    c2e_map = np.reshape(c2e_connect.array, (-1, num_e_per_c))

    mesh.topology.create_connectivity(edim, cdim)
    e2c_connect = mesh.topology.connectivity(edim, cdim)
    e2c_map = _reshape_map(e2c_connect)[0]

    bg_mesh_boundary_edges = dfx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    cut_edges = edges_tags.find(2)
    uncut_bg_mesh_boundary_edges = np.setdiff1d(bg_mesh_boundary_edges, cut_edges)

    # Get tagged cells
    inside_cells = cells_tags.find(1)
    cut_cells = cells_tags.find(2)
    outside_cells = cells_tags.find(3)

    direct_interface_edges = np.intersect1d(
        c2e_map[outside_cells], c2e_map[inside_cells]
    )
    inside_boundary_edges = np.intersect1d(c2e_map[inside_cells], c2e_map[cut_cells])
    edges_to_remove = np.union1d(direct_interface_edges, inside_boundary_edges)
    outside_boundary_edges = np.intersect1d(c2e_map[outside_cells], c2e_map[cut_cells])

    uncut_edge_boundary_cells = e2c_map[uncut_bg_mesh_boundary_edges][:, 0]
    cut_cells_boundary = np.intersect1d(uncut_edge_boundary_cells, cut_cells)
    bdy_edges_cut_cells_boundary = np.intersect1d(
        c2e_map[cut_cells_boundary], bg_mesh_boundary_edges
    )
    outside_boundary_edges = np.union1d(
        outside_boundary_edges, bdy_edges_cut_cells_boundary
    )

    edges_to_remove = np.union1d(edges_to_remove, outside_boundary_edges)

    inside_edges = np.setdiff1d(edges_tags.find(1), edges_to_remove)
    outside_edges = np.setdiff1d(edges_tags.find(5), edges_to_remove)

    # Create the meshtags from the indices.
    indices = np.hstack(
        [
            outside_edges,
            inside_edges,
            inside_boundary_edges,
            cut_edges,
            outside_boundary_edges,
            direct_interface_edges,
        ]
    ).astype(np.int32)
    inside_marker = np.full_like(inside_edges, 1).astype(np.int32)
    cut_marker = np.full_like(cut_edges, 2).astype(np.int32)
    inside_boundary_marker = np.full_like(inside_boundary_edges, 3).astype(np.int32)
    outside_boundary_marker = np.full_like(outside_boundary_edges, 4).astype(np.int32)
    outside_marker = np.full_like(outside_edges, 5).astype(np.int32)
    direct_interface_marker = np.full_like(direct_interface_edges, 6).astype(np.int32)
    markers = np.hstack(
        [
            outside_marker,
            inside_marker,
            inside_boundary_marker,
            cut_marker,
            outside_boundary_marker,
            direct_interface_marker,
        ]
    ).astype(np.int32)
    sorted_indices = np.argsort(indices)

    completed_edges_tags = dfx.mesh.meshtags(
        mesh, 1, indices[sorted_indices], markers[sorted_indices]
    )

    return completed_edges_tags


def compute_tags_measures(
    mesh: Mesh,
    discrete_levelset: Any,
    detection_degree: int,
    box_mode: bool = False,
    single_cut_layer: bool = False,
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
        detection_degree: the degree used in the custom quadrature rule of the detection form.
        box_mode: if False (default), create a submesh and return the cells tags on the submesh, if True, returns cells tags on the input mesh.
        single_cut_layer: if True, force a single layer of cut edges.

    Returns
        The mesh/submesh cells tags.
        The mesh/submesh facets tags.
        The mesh/submesh (input mesh if box_mode is True).
        The one-sided measure from inside.
        The one-sided measure from outside.
        Submesh c-map, v-map and n-map.
    """

    # Create a wireframe mesh from mesh
    mesh_edges = dfx.mesh.locate_entities(
        mesh, 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    wireframe, fmap = dfx.mesh.create_submesh(mesh, 1, mesh_edges)[:2]

    # Create a levelset element and function space on wireframe
    wf_cell_name = wireframe.topology.cell_name()

    if hasattr(discrete_levelset, "function_space"):
        levelset_space = discrete_levelset.function_space
        levelset_element = levelset_space.ufl_element()
        wf_levelset_element = element(
            levelset_element.family_name,
            wf_cell_name,
            levelset_element.degree,
        )

        wf_levelset_space = dfx.fem.functionspace(wireframe, wf_levelset_element)

        num_wf_cells = wireframe.topology.index_map(1).size_global
        wf_cells = np.arange(num_wf_cells)
        nmm_mesh2wireframe = dfx.fem.create_interpolation_data(
            wf_levelset_space, levelset_space, wf_cells
        )
        wf_levelset = dfx.fem.Function(wf_levelset_space)
        wf_levelset.interpolate_nonmatching(
            discrete_levelset, wf_cells, nmm_mesh2wireframe
        )
    else:
        wf_levelset_element = element("Lagrange", wf_cell_name, detection_degree)
        wf_levelset_space = dfx.fem.functionspace(wireframe, wf_levelset_element)
        wf_levelset = dfx.fem.Function(wf_levelset_space)
        x_ufl = ufl.SpatialCoordinate(wireframe)
        discrete_levelset_expr = dfx.fem.Expression(
            discrete_levelset(x_ufl), wf_levelset_space.element.interpolation_points()
        )
        wf_levelset.interpolate(discrete_levelset_expr)

    incomplete_edges_tags = _tag_edges(
        wireframe,
        wf_levelset,
        detection_degree,
        single_cut_layer=single_cut_layer,
    )
    mesh_edges_tags = dfx.mesh.meshtags(
        mesh, 1, fmap[incomplete_edges_tags.indices], incomplete_edges_tags.values
    )
    cells_tags = _tag_cells(mesh, mesh_edges_tags)
    mesh_edges_tags = _complete_edges_tags(mesh, cells_tags, mesh_edges_tags)

    if box_mode:
        submesh = None
        integration_cells = np.union1d(cells_tags.find(2), cells_tags.find(1))
        d_boundary_outside = _one_sided_edge_measure(
            mesh, integration_cells, mesh_edges_tags.find(4), 100
        )
        integration_cells = np.union1d(cells_tags.find(2), cells_tags.find(3))
        d_boundary_inside = _one_sided_edge_measure(
            mesh, integration_cells, mesh_edges_tags.find(3), 101
        )
        submesh_maps = None
    else:
        # We create the submesh
        omega_h_cells = np.unique(np.hstack([cells_tags.find(1), cells_tags.find(2)]))
        submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(
            mesh, mesh.topology.dim, omega_h_cells
        )  # type: ignore

        cells_tags = _transfer_tags(cells_tags, submesh, c_map)
        mesh_edges_tags = _transfer_tags(
            mesh_edges_tags, submesh, c_map, source_mesh=mesh
        )
        d_boundary_outside = None
        d_boundary_inside = None
        submesh_maps = [c_map, v_map, n_map]

    return (
        cells_tags,
        mesh_edges_tags,
        submesh,
        d_boundary_outside,
        d_boundary_inside,
        submesh_maps,
    )
