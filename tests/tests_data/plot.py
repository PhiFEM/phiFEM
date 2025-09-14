from   collections.abc import Callable

from   dolfinx.mesh import Mesh, MeshTags
from   mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore
from   matplotlib import cm
import matplotlib.collections as mpl_collections
from   matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import numpy as np
import numpy.typing as npt
from   os import PathLike

from   typing import cast, Any, Collection


PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

# Snippet taken from https://github.com/multiphenics/multiphenicsx/blob/main/tutorials/07_understanding_restrictions/tutorial_understanding_restrictions.ipynb...
# ...and butchered so that we can pass a mesh_tags with more than 2 different tags and display the cells and/or facets indices.
# TODO: add more line styles for the moment it's not very colorblind friendly.
def plot_mesh_tags(
    mesh: Mesh,
    mesh_tags: MeshTags,
    ax: plt.Axes | None = None,
    display_indices: bool = False,
    expression_levelset: Callable[..., np.ndarray] | None = None,
    linewidth: float = 0.5,
    colormap: str = "Set1",
    leg_dict: dict | None = None
) -> plt.Axes:
    """Plot a mesh tags object on the provided (or, if None, the current) axes object.
    
    Args:
        mesh: the corresponding mesh.
        mesh_tags: the mesh tags.
        ax: (optional default None) the matplotlib axes.
        display_indices: (optional default False) boolean, if True displays the indices of the cells/facets.
        expression_levelset: (optional default None), if not None, display the contour line of the levelset.
        linewidth: (optional default 0.5), lines width.
        colormap: (optional default 'Set1') the matplotlib colormap.
        leg_dict: (optional default None) the dictionnary mapping the tags to legend entries.
    
    Returns:
        A matplotlib axis with the corresponding plot.
    """
    num_vertices_cell = mesh.geometry.dofmap.shape[1]

    if num_vertices_cell == 4:
        quad_cells = mesh.geometry.dofmap
    elif num_vertices_cell != 3:
        raise ValueError("plot_mesh_tags only handles triangular or quadrilateral meshes.")

    if ax is None:
        ax = plt.gca()  # type: ignore
    ax.set_aspect("equal")
    points = mesh.geometry.x

    # Get unique tags and create a custom colormap
    num_values = len(np.unique(mesh_tags.values))
    cmap = cm.get_cmap(colormap, num_values)
    boundaries = np.arange(1, num_values + 2) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts

    mappable: mpl_collections.Collection
    if mesh_tags.dim == mesh.topology.dim:
        cells = mesh.geometry.dofmap
        if num_vertices_cell == 4:
            tri_cells = []
            for cell in quad_cells:
                v0, v1, v2, v3 = cell
                tri_cells.append([v0, v1, v2])
                tri_cells.append([v1, v2, v3])
            cells = np.array(tri_cells)
        
        if display_indices:
            tdim = mesh.topology.dim
            connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
            vertex_map = {
                topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
                    connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
            }
        
        if num_vertices_cell == 3:
            cell_colors = np.zeros((cells.shape[0], ))
            for c in range(num_cells):
                if c in mesh_tags.indices:
                    cell_colors[c] = mesh_tags.values[np.where(mesh_tags.indices == c)][0]
                    if display_indices:
                        vertices = [vertex_map[v] for v in connectivity_cells_to_vertices.links(c)]
                        midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
                        ax.text(midpoint[0], midpoint[1], f"{c}", horizontalalignment="center", verticalalignment="center", fontsize=6)
                else:
                    cell_colors[c] = -1  # Handle cells without tags (optional)
        elif num_vertices_cell == 4:
            cell_colors = []
            for c in range(num_cells):
                if c in mesh_tags.indices:
                    tag_value = mesh_tags.values[np.where(mesh_tags.indices == c)][0]
                else:
                    tag_value = -1
                cell_colors.extend([tag_value, tag_value])
            cell_colors = np.array(cell_colors)
        
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        mappable = ax.tripcolor(tria,
                                cell_colors,
                                edgecolor="none",
                                cmap=cmap,
                                norm=norm)

        polygons = []
        for cell in mesh.geometry.dofmap:
            points_arr = points[cell][:, :2]
            points_arr[[0,1]] = points_arr[[1,0]]
            polygons.append(points_arr)
        edges_mappable = PolyCollection(polygons,
                                        facecolor="none",
                                        edgecolors="k",
                                        linewidth=linewidth)
        ax.add_collection(edges_mappable)
    elif mesh_tags.dim == mesh.topology.dim - 1:
        tdim = mesh.topology.dim
        connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
        connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
        connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
        vertex_map = {
            topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
                connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
        }
        lines               = list()
        lines_colors_as_int = list()
        lines_colors_as_str = list()
        lines_linestyles    = list()
        for c in range(num_cells):
            facets = connectivity_cells_to_facets.links(c)
            for f in facets:
                if f in mesh_tags.indices:
                    value_f = mesh_tags.values[np.where(mesh_tags.indices == f)][0]
                else:
                    value_f = -1  # Handle facets without tags (optional)
                vertices = [vertex_map[v] for v in connectivity_facets_to_vertices.links(f)]
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(cmap(value_f - 1) if value_f != -1 else "gray")
                lines.append(points[vertices][:, :2])
                lines_linestyles.append("solid")
                if display_indices:
                    midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
                    ax.text(midpoint[0], midpoint[1], f"{f}", horizontalalignment="center", verticalalignment="center", fontsize=6)
        mappable = mpl_collections.LineCollection(lines,
                                                  cmap=cmap,
                                                  norm=norm,
                                                  colors=lines_colors_as_str,
                                                  linestyles=lines_linestyles,
                                                  linewidth=linewidth)
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(cast(Collection[Any], mappable))
        ax.autoscale()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(mappable, cax=cax, boundaries=boundaries, ticks=np.arange(1, num_values+1))
    
    # Set colorbar labels
    if leg_dict is not None:
        colorbar.set_ticklabels([leg_dict[j+1] + " (" + str(j+1) + ")" for j in range(num_values)])
    else:
        colorbar.set_ticklabels([str(j+1) for j in range(num_values)])

    if expression_levelset is not None:
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        nx = 1000
        ny = 1000
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)

        xx, yy = np.meshgrid(xs, ys)
        xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
        yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
        points = np.vstack([xx_rs, yy_rs])
        zz_rs = expression_levelset(points)
        zz = zz_rs.reshape(xx.shape)

        ax.contour(xx, yy, zz, [0.], linewidths=linewidth)
    return ax