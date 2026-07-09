import os
import warnings
from typing import Any, Tuple

import dolfinx as dfx
import numpy as np
import numpy.typing as npt
import ufl
from basix.ufl import element
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import Mesh, MeshTags
from ufl import inner

from phifem import measures
from phifem.utils import reshape_map

debug_mode = False
if "MODE" in os.environ:
    if os.environ["MODE"] == "debug":
        debug_mode = True


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


def overwrite(mesh, tags_to_overwrite, new_tags):
    stack_indices = np.hstack([new_tags.indices, tags_to_overwrite.indices])
    stack_values = np.hstack([new_tags.values, tags_to_overwrite.values])
    overwritten_indices, ind = np.unique(stack_indices, return_index=True)
    overwritten_values = stack_values[ind]

    overwritten_tags = dfx.mesh.meshtags(
        mesh, tags_to_overwrite.dim, overwritten_indices, overwritten_values
    )
    return overwritten_tags


def _single_layer_filter(
    mesh: Mesh, interior_indices, cut_indices, exterior_indices
) -> Tuple[npt.NDArray[np.int32]]:
    """
    Modify the cut and exterior indices to force a single layer of cut cells.

    Args:
        mesh: the mesh.
        interior_indices: the indices of cells strictly inside the domain.
        cut_indices: the initial indices of cells cut by the boundary.
        exterior_indices: the initial indices of cells strictly outside the domain.

    Returns:
        The modified cut and exterior indices.
    """
    cdim = mesh.topology.dim
    vdim = 0
    # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
    mesh.topology.create_connectivity(cdim, vdim)
    c2v_connect = mesh.topology.connectivity(cdim, vdim)
    num_vertices_per_cell = len(c2v_connect.links(0))
    c2v_map = np.reshape(c2v_connect.array, (-1, num_vertices_per_cell))

    mesh.topology.create_connectivity(vdim, cdim)
    v2c_connect = mesh.topology.connectivity(vdim, cdim)
    v2c_map, max_offset = reshape_map(v2c_connect)
    neighbor_cells = np.reshape(
        v2c_map[c2v_map[cut_indices]], (-1, num_vertices_per_cell * max_offset)
    )
    mask_connected_cut_cells = np.any(np.isin(neighbor_cells, interior_indices), axis=1)
    isolated_cut_cells = cut_indices[~mask_connected_cut_cells]
    cut_indices = np.setdiff1d(cut_indices, isolated_cut_cells)
    exterior_indices = np.union1d(exterior_indices, isolated_cut_cells)
    return cut_indices, exterior_indices


def cells(
    mesh: Mesh,
    discrete_levelset: Function,
    detection_degree: int,
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
    cell_type = mesh.topology.cell_type.name
    detection_measure = measures.detection(mesh, detection_degree, cell_type, "dx")

    detection_vector = _compute_detection_vector(
        mesh, discrete_levelset, detection_measure
    )
    print("cells detection = ", detection_vector)
    cut_indices = np.where(
        np.logical_and(detection_vector > -1.0, detection_vector < 1.0)
    )[0]
    print(cut_indices)
    exterior_indices = np.where(detection_vector == 1.0)[0]
    interior_indices = np.where(detection_vector == -1.0)[0]

    if single_layer_cut:
        cut_indices, exterior_indices = _single_layer_filter(
            mesh, interior_indices, cut_indices, exterior_indices
        )

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


def facets(
    mesh: Mesh,
    cells_tags: MeshTags,
    discrete_levelset: Function,
    detection_degree: int,
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

    detection_measure = measures.detection(mesh, detection_degree, "segment", "ds")

    detection_vector = _compute_detection_vector(
        mesh, discrete_levelset, detection_measure
    )
    print("facets detection = ", detection_vector)
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


def transfer(
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
        unique_indices, sorted_indices = np.unique(dest_c2f_map, return_index=True)
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
