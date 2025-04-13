from   basix.ufl         import element
from   collections.abc   import Callable
import dolfinx           as dfx
from   dolfinx.cpp.graph import AdjacencyList_int32 # type: ignore
from   dolfinx.fem       import Function
from   dolfinx.fem.petsc import assemble_vector
from   dolfinx.mesh      import Mesh, MeshTags
import numpy             as np
import numpy.typing      as npt
from   os                import PathLike
from   typing            import Any, Tuple
import ufl # type: ignore
from   ufl               import inner, grad

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

def _reshape_facets_map(f2c_connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
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

def _transfer_cells_tags(source_mesh_cells_tags: MeshTags,
                         dest_mesh: Mesh,
                         cmap: npt.NDArray[Any]) -> MeshTags:
    """ Given a cells tags from a source mesh, a destination mesh and the source mesh-destination mesh cells mapping, transfers the cells tags to the destination mesh.

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
    list_markers    = []
    for value in tag_values:
        source_cells = source_mesh_cells_tags.find(value)
        mask = np.in1d(cmap, source_cells)
        dest_mesh_masked = np.where(mask)[0]
        list_dest_cells.append(dest_mesh_masked)
        list_markers.append(np.full_like(dest_mesh_masked, value))
    
    dest_cells_indices = np.hstack(list_dest_cells).astype(np.int32)
    dest_cells_markers = np.hstack(list_markers).astype(np.int32)
    sorted_indices = np.argsort(dest_cells_indices)

    dest_cells_tags = dfx.mesh.meshtags(dest_mesh,
                                        cdim,
                                        dest_cells_indices[sorted_indices],
                                        dest_cells_markers[sorted_indices])
    return dest_cells_tags

def _tag_cells(mesh: Mesh,
              levelset: Levelset,
              detection_degree: int) -> MeshTags:
    """Tag the mesh cells by computing detection = Σ f(dof)/Σ|f(dof)| for each cell.
         detection == 1  => the cell is stricly OUTSIDE {phi_h < 0} => we tag it as 3
         detection == -1 => the cell is stricly INSIDE  {phi_h < 0} => we tag it as 1
         otherwise       => the cell is CUT by Gamma_h              => we tag is as 2

    Args:
        mesh: the background mesh.
        discrete_levelset: the discretization of the levelset.
    
    Returns:
        The cells tags as a MeshTags object.
    """

    cdim = mesh.topology.dim
    detection_measure_subdomain = np.arange(mesh.topology.index_map(cdim).size_global)
    if detection_degree > 1:
        fdim = cdim - 1
        mesh.topology.create_connectivity(cdim, fdim)
        mesh.topology.create_connectivity(fdim, cdim)
        c2f_connect = mesh.topology.connectivity(cdim, fdim)
        num_facets_per_cell = len(c2f_connect.links(0))
        c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        f2c_connect = mesh.topology.connectivity(fdim, cdim)
        f2c_map = _reshape_facets_map(f2c_connect)
        detection_degrees = [1, detection_degree]
    else:
        detection_degrees = [detection_degree]
    
    for degree in detection_degrees:
        # Create the custom quadrature rule.
        # The evaluation points are the dofs of the reference cell.
        # The weights are 1.
        quadrature_points: npt.NDArray[np.float64]
        if mesh.topology.cell_name() == "triangle":
            xs = np.linspace(0., 1., degree + 1)
            xx, yy = np.meshgrid(xs, xs)
            x_coords = xx.reshape((1, xx.shape[0] * xx.shape[1]))
            y_coords = yy.reshape((1, yy.shape[0] * yy.shape[1]))
            points = np.vstack([x_coords, y_coords])
            quadrature_points = points[:,points[1,:] <= np.ones_like(points[0,:])-points[0,:]]
        elif mesh.topology.cell_name() == "tetrahedron":
            xs = np.linspace(0., 1., degree + 1)
            xx, yy, zz = np.meshgrid(xs, xs, xs)
            x_coords = xx.reshape((1, xx.shape[0] * xx.shape[1]))
            y_coords = yy.reshape((1, yy.shape[0] * yy.shape[1]))
            z_coords = zz.reshape((1, zz.shape[0] * zz.shape[1]))
            points = np.vstack([x_coords, y_coords, z_coords])
            quadrature_points = points[:,points[2,:] <= np.ones_like(points[0,:])-points[0,:]-points[1,:]]
        else:
            raise NotImplementedError("Mesh cell type not supported. Only supported types are: triangle, tetrahedron.")

        quadrature_weights = np.ones_like(quadrature_points[0,:])
        custom_rule = {"quadrature_rule":    "custom",
                       "quadrature_points":  quadrature_points.T,
                       "quadrature_weights": quadrature_weights}
        
        cells_detection_dx = ufl.Measure("dx",
                                        domain=mesh,
                                        subdomain_data=detection_measure_subdomain,
                                        metadata=custom_rule)
        
        detection_element = element("Lagrange", mesh.topology.cell_name(), degree)
        detection_space = dfx.fem.functionspace(mesh, detection_element)
        discrete_levelset = dfx.fem.Function(detection_space)
        detection_expression = levelset.get_detection_expression()
        discrete_levelset.interpolate(detection_expression)
        # We localize at each cell via a DG0 test function.
        DG0Element = element("DG", mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(mesh, DG0Element)
        v0 = ufl.TestFunction(V0)

        # Assemble the numerator of detection
        cells_detection_num = inner(discrete_levelset, v0) * cells_detection_dx
        cells_detection_num_form = dfx.fem.form(cells_detection_num)
        cells_detection_num_vec = assemble_vector(cells_detection_num_form)
        # Assemble the denominator of detection
        cells_detection_denom = inner(ufl.algebra.Abs(discrete_levelset), v0) * cells_detection_dx
        cells_detection_denom_form = dfx.fem.form(cells_detection_denom)
        cells_detection_denom_vec = assemble_vector(cells_detection_denom_form)

        # cells_detection_denom_vec is not supposed to be zero, this would mean that the levelset is zero at all dofs in a cell.
        # However, in practice it can happen that for a very small cut triangle, cells_detection_denom_vec is of the order of the machine precision.
        # In this case, we set the value of cells_detection_vec to 0.5, meaning we consider the cell as cut.
        mask = np.where(cells_detection_denom_vec.array > 0.)
        cells_detection_vec = np.full_like(cells_detection_num_vec.array, 0.5)
        cells_detection_vec[mask] = cells_detection_num_vec.array[mask]/cells_detection_denom_vec.array[mask]

        detection = dfx.fem.Function(V0)
        detection.x.array[:] = cells_detection_vec

        cut_indices = np.where(np.logical_and(cells_detection_vec > -1.,
                                              cells_detection_vec < 1.))[0]
        
        if detection_degree > 1:
            neighbor_cells = f2c_map[c2f_map[cut_indices]]
            detection_measure_subdomain = neighbor_cells

    
    exterior_indices = np.where(cells_detection_vec == 1.)[0]
    interior_indices = np.where(cells_detection_vec == -1.)[0]
    
    if len(interior_indices) == 0:
        raise ValueError("No interior cells (1)!")
    if len(cut_indices) == 0:
        raise ValueError("No cut cells (2)!")

    # Create the meshtags from the indices.
    indices = np.hstack([exterior_indices,
                         interior_indices,
                         cut_indices]).astype(np.int32)
    exterior_marker = np.full_like(exterior_indices, 3).astype(np.int32)
    interior_marker = np.full_like(interior_indices, 1).astype(np.int32)
    cut_marker      = np.full_like(cut_indices,      2).astype(np.int32)
    markers = np.hstack([exterior_marker,
                         interior_marker,
                         cut_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    cells_tags = dfx.mesh.meshtags(mesh,
                                   mesh.topology.dim,
                                   indices[sorted_indices],
                                   markers[sorted_indices])

    return cells_tags

def _tag_facets(mesh: Mesh,
               cells_tags: MeshTags,
               plot: bool = False) -> MeshTags:
    """Tag the mesh facets.
    Strictly interior facets  => tag it 1
    Cut facets                => tag it 2
    Strictly exterior facets  => tag it 3
    Boundary facets (Gamma_h) => tag it 4

    Args:
        mesh: the background mesh.
        cells_tags: the MeshTags object containing cells tags.
        plot: if True plots the mesh with tags (can drastically slow the computation!).
    
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
    cut_cells      = cells_tags.find(2)
    exterior_cells = cells_tags.find(3)
    
    # Facets shared by an interior cell and a cut cell
    interior_boundary_facets = np.intersect1d(c2f_map[interior_cells],
                                              c2f_map[cut_cells])
    # Facets shared by an exterior cell and a cut cell
    exterior_boundary_facets = np.intersect1d(c2f_map[exterior_cells],
                                              c2f_map[cut_cells])
    # Boundary facets ∂Ω_h
    real_boundary_facets = np.intersect1d(c2f_map[cut_cells], 
                                          dfx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.ones_like(x[1]).astype(bool)))
    boundary_facets = np.union1d(exterior_boundary_facets, real_boundary_facets)

    # Cut facets F_h^Γ
    cut_facets = np.setdiff1d(c2f_map[cut_cells],
                              np.union1d(exterior_boundary_facets, boundary_facets))

    # Interior facets 
    interior_facets = np.setdiff1d(c2f_map[interior_cells],
                                   interior_boundary_facets)
    # Exterior facets 
    exterior_facets = np.setdiff1d(c2f_map[exterior_cells],
                                   exterior_boundary_facets)
    
    if len(interior_facets) == 0:
        raise ValueError("No interior facets (1)!")
    if len(cut_facets) == 0:
        raise ValueError("No cut facets (2)!")
    if len(boundary_facets) == 0:
        raise ValueError("No boundary facets (4)!")
    
    # Create the meshtags from the indices.
    indices = np.hstack([exterior_facets,
                         interior_facets,
                         cut_facets,
                         boundary_facets]).astype(np.int32)
    interior_marker = np.full_like(interior_facets, 1).astype(np.int32)
    cut_marker      = np.full_like(cut_facets,      2).astype(np.int32)
    exterior_marker = np.full_like(exterior_facets, 3).astype(np.int32)
    boundary_marker = np.full_like(boundary_facets, 4).astype(np.int32)
    markers = np.hstack([exterior_marker,
                         interior_marker,
                         cut_marker,
                         boundary_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    facets_tags = dfx.mesh.meshtags(mesh,
                                    fdim,
                                    indices[sorted_indices],
                                    markers[sorted_indices])

    return facets_tags

def compute_tags(mesh: Mesh,
                 levelset: Levelset,
                 detection_degree: int,
                 keep_mesh: bool = True) -> Tuple[MeshTags, Mesh | None]:
    """ Compute the mesh tags.

    Args:
        mesh: the mesh on which we compute the tags.
        levelset: the levelset function used to discriminate the cells.
        detection_degree: the degree of the piecewise-polynomial approximation to the levelset.
        keep_mesh: if True (default), returns cells tags on the input mesh, if False, create a submesh and return the cells tags on the submesh.
    
    Returns
        The mesh/submesh cells tags.
        The mesh/submesh facets tags.
        The submesh (None is keep_mesh is True).
    """
    cells_tags = _tag_cells(mesh, levelset, detection_degree)

    if keep_mesh:
        submesh = None
        facets_tags = _tag_facets(mesh, cells_tags)
    else:
        # We create the submesh
        omega_h_cells = np.unique(np.hstack([cells_tags.find(1),
                                             cells_tags.find(2)]))
        submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(mesh,
                                                               mesh.topology.dim,
                                                               omega_h_cells) # type: ignore

        cells_tags = _transfer_cells_tags(cells_tags, submesh, c_map)
        facets_tags = _tag_facets(submesh, cells_tags)

    return cells_tags, facets_tags, submesh