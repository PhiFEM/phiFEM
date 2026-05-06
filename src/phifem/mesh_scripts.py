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

    imap = dg_0_space.dofmap.index_map
    detection_vector = dfx.la.vector(imap)
    # detection_denom_vec is not supposed to be zero, this would mean that the levelset is zero at all dofs in a cell.
    # However, in practice it can happen that for a very small cut triangle, detection_denom_vec is of the order of the machine precision.
    # In this case, we set the value of detection_vector to 0.5, meaning we consider the cell as cut.
    if np.any(np.isclose(detection_denom_vec.array, 0.0)):
        warnings.warn(
            "The levelset function is zero everywhere on a cell. We mark it as 'cut' but this can be incorrect and should be carefully checked.",
            RuntimeWarning,
        )
    mask = np.where(detection_denom_vec.array > 0.0)
    detection_vector.array[:] = 0.5
    detection_vector.array[mask] = (
        detection_num_vec.array[mask] / detection_denom_vec.array[mask]
    )
    detection_vector.scatter_forward()
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


def _transfer_markers(
    source_markers: dfx.la.Vector,
    cmap: npt.NDArray[Any],
    edim: int,
) -> dfx.la.Vector:
    """Given entities tags (cells or facets) from a source mesh, a destination mesh and the source mesh-destination mesh cells mapping, transfers the entities tags to the destination mesh.

    Args:
        source_markers: the markers on the source mesh.
        dest_mesh: the destination mesh.
        cmap: the source mesh-destination mesh cells mapping.

    Returns:
        Cells tags on the destination mesh.
    """
    sub_index_map = cmap.sub_topology.index_map(edim)
    local_entities = np.arange(sub_index_map.size_local + sub_index_map.num_ghosts)
    emap = cmap.sub_topology_to_topology(
        local_entities,
        False,
    )
    dest_markers = dfx.la.vector(sub_index_map)
    dest_markers.array[:] = source_markers.array[emap]

    return dest_markers


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
    discrete_levelset: Function,
    detection_degree: int,
    single_layer_cut: bool = False,
) -> dfx.la.Vector:
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
        The cells markers as a dfx.la.Vector object.
    """
    if single_layer_cut:
        cdim = mesh.topology.dim
        vdim = 0
        # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
        mesh.topology.create_connectivity(cdim, vdim)
        c2v_connect = mesh.topology.connectivity(cdim, vdim)
        num_vertices_per_cell = len(c2v_connect.links(0))
        c2v_map = np.reshape(c2v_connect.array, (-1, num_vertices_per_cell))

        mesh.topology.create_connectivity(vdim, cdim)
        v2c_connect = mesh.topology.connectivity(vdim, cdim)
        v2c_map, max_offset = _reshape_map(v2c_connect)

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

    detection_vector = _compute_detection_vector(
        mesh, discrete_levelset, detection_measure
    )
    cut_indices = np.where(
        np.logical_and(
            detection_vector.array[:] > -1.0, detection_vector.array[:] < 1.0
        )
    )[0]
    exterior_indices = np.where(np.isclose(detection_vector.array[:], 1.0))[0]
    interior_indices = np.where(np.isclose(detection_vector.array[:], -1.0))[0]

    if single_layer_cut:
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
    cells_markers = dfx.la.vector(detection_vector.index_map)
    cells_markers.array[interior_indices] = 1
    cells_markers.array[cut_indices] = 2
    cells_markers.array[exterior_indices] = 3
    cells_markers.scatter_forward()

    return cells_markers


def _tag_facets(
    mesh: Mesh,
    cells_markers: dfx.la.Vector,
    discrete_levelset: Function,
    detection_degree: int,
) -> dfx.la.Vector:
    """Tag the mesh facets.
    Strictly interior facets  => tag 1
    Cut facets                => tag 2
    Interior boundary facets  => tag 3
    Boundary facets (Gamma_h) => tag 4
    Strictly exterior facets  => tag 5
    Direct interface facets   => tag 6

    Args:
        mesh: the background mesh.
        cells_markers: the vector storing the cells markers.
        discrete_levelset: the discretization of the levelset.
        detection_degree: the degree of the custom quadrature rule used to detect cut entities.

    Returns:
        The facets markers as a dfx.la.Vector object.
        The wireframe mesh containing all facets.
    """
    cdim = mesh.topology.dim
    fdim = cdim - 1
    # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
    mesh.topology.create_connectivity(cdim, fdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))

    # Get tagged cells
    interior_cells = np.where(cells_markers.array[:] == 1)
    cut_cells = np.where(cells_markers.array[:] == 2)
    exterior_cells = np.where(cells_markers.array[:] == 3)

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

    detection_vector = _compute_detection_vector(
        mesh, discrete_levelset, detection_measure
    )
    mask_cut_indices_cells = np.logical_and(
        detection_vector.array[:] > -1.0, detection_vector.array[:] < 1.0
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
    if debug_mode:
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

    # Allocate distributed vectors supporting facets markers.
    all_facets = dfx.mesh.locate_entities(
        mesh, 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    wireframe = dfx.mesh.create_submesh(mesh, 1, all_facets)[0]
    cell_name_wf = wireframe.topology.cell_name()
    dg0_element_wf = element("DG", cell_name_wf, 0)
    dg0_space_wf = dfx.fem.functionspace(wireframe, dg0_element_wf)
    imap = dg0_space_wf.dofmap.index_map
    facets_markers = dfx.la.vector(imap)

    facets_markers.array[interior_facets] = 1
    facets_markers.array[cut_facets] = 2
    facets_markers.array[interior_boundary_facets] = 3
    facets_markers.array[boundary_facets] = 4
    facets_markers.array[exterior_facets] = 5
    facets_markers.array[direct_interface_facets] = 6
    facets_markers.scatter_forward()

    return facets_markers


def compute_tags_measures(
    mesh: Mesh,
    discrete_levelset: Function,
    detection_degree: int,
    box_mode: bool = False,
    single_layer_cut: bool = False,
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
    cells_markers = _tag_cells(
        mesh, discrete_levelset, detection_degree, single_layer_cut=single_layer_cut
    )
    facets_markers = _tag_facets(
        mesh, cells_markers, discrete_levelset, detection_degree
    )

    if box_mode:
        submesh = None
        integration_cells = np.union1d(
            np.where(np.isclose(cells_markers.array[:], 2)),
            np.where(np.isclose(cells_markers.array[:], 1)),
        )
        integration_entities_outside = _compute_integration_entities(
            mesh,
            integration_cells,
            np.where(np.isclose(facets_markers.array[:], 4)),
            100,
        )
        integration_cells = np.union1d(
            np.where(np.isclose(cells_markers.array[:], 2)),
            np.where(np.isclose(cells_markers.array[:], 3)),
        )
        integration_entities_inside = _compute_integration_entities(
            mesh,
            integration_cells,
            np.where(np.isclose(facets_markers.array[:], 3)),
            101,
        )
        combined_integration_entities = (
            integration_entities_outside + integration_entities_inside
        )

        boundaries_measure = ufl.Measure(
            "ds", domain=mesh, subdomain_data=combined_integration_entities
        )
        submesh_maps = None
        indices = np.arange(
            cells_markers.index_map.size_local + cells_markers.index_map.num_ghosts
        )
        cells_tags = dfx.mesh.meshtags(
            mesh, mesh.topology.dim, indices, cells_markers.array[:]
        )
        indices = np.arange(
            facets_markers.index_map.size_local + facets_markers.index_map.num_ghosts
        )
        facets_tags = dfx.mesh.meshtags(
            mesh, mesh.topology.dim - 1, indices, facets_markers.array[:]
        )
    else:
        # We create the submesh
        omega_h_cells = np.unique(
            np.hstack(
                [
                    np.where(np.isclose(cells_markers.array[:], 1)),
                    np.where(np.isclose(cells_markers.array[:], 2)),
                ]
            )
        )
        submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(
            mesh, mesh.topology.dim, omega_h_cells
        )
        submesh_maps = [c_map, v_map, n_map]

        cells_markers_submesh = _transfer_markers(
            cells_markers, c_map, submesh.topology.dim
        )
        submesh.topology.create_entities(submesh.topology.dim - 1)
        submesh_imap = submesh.topology.index_map(submesh.topology.dim - 1)
        submesh_facets = np.arange(submesh_imap.size_local + submesh_imap.num_ghosts)
        wf_submesh, wf_map = dfx.mesh.create_submesh(
            submesh, submesh.topology.dim - 1, submesh_facets
        )[:2]
        wf_submesh.topology.create_entities(wf_submesh.topology.dim)
        facets_markers_subwf = _transfer_markers(
            facets_markers, wf_map, submesh.topology.dim - 1
        )
        boundaries_measure = ufl.Measure("ds", domain=submesh)

        indices = np.arange(
            cells_markers_submesh.index_map.size_local
            + cells_markers_submesh.index_map.num_ghosts
        )
        cells_tags = dfx.mesh.meshtags(
            submesh, submesh.topology.dim, indices, cells_markers_submesh.array[:]
        )
        facets_tags = dfx.mesh.meshtags(
            submesh,
            submesh.topology.dim - 1,
            submesh_facets,
            facets_markers_subwf.array[submesh_facets],
        )

    return (
        cells_tags,
        facets_tags,
        submesh,
        boundaries_measure,
        submesh_maps,
    )
