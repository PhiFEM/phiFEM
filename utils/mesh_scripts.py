from   collections.abc import Callable

from   dolfinx.mesh import Mesh, MeshTags
from   dolfinx.fem import Function
from   mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore
from   matplotlib import cm
import matplotlib.collections as mpl_collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import numpy as np
import numpy.typing as npt
from   os import PathLike

from   typing import cast, Any, Collection


PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

# Snippet stolen from https://github.com/multiphenics/multiphenicsx/blob/main/tutorials/07_understanding_restrictions/tutorial_understanding_restrictions.ipynb...
# ...and butchered so that we can pass a mesh_tags with more than 2 different tags and display the cells and/or facets indices.
# TODO: add more line styles for the moment it's not very colorblind friendly.
def plot_mesh_tags(
    mesh: Mesh,
    mesh_tags: MeshTags,
    ax: plt.Axes | None = None,
    display_indices: bool = False,
    expression_levelset: Callable[..., np.ndarray] | None = None
) -> plt.Axes:
    """Plot a mesh tags object on the provided (or, if None, the current) axes object.
    
    Args:
        mesh: the corresponding mesh.
        mesh_tags: the mesh tags.
        ax: (optional) the matplotlib axes.
        display_indices: (optional) boolean, if True displays the indices of the cells/facets.
        expression_levelset: (optional), if not None, display the contour line of the levelset.
    
    Returns:
        A matplotlib axis with the corresponding plot.
    """
    if ax is None:
        ax = plt.gca()  # type: ignore
    ax.set_aspect("equal")
    points = mesh.geometry.x

    # Get unique tags and create a custom colormap
    tab10 = cm.get_cmap("tab10")
    colors = [tab10(i / 10.) for i in range(10)]
    colors = colors[:5]
    cmap = mcolors.ListedColormap(colors) # type: ignore
    norm = mcolors.BoundaryNorm(np.arange(6) - 0.5, 5)

    assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts

    mappable: mpl_collections.Collection
    if mesh_tags.dim == mesh.topology.dim:
        cells = mesh.geometry.dofmap
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        cell_colors = np.zeros((cells.shape[0], ))
        if display_indices:
            tdim = mesh.topology.dim
            connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
            vertex_map = {
                topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
                    connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
            }
        for c in range(num_cells):
            if c in mesh_tags.indices:
                cell_colors[c] = mesh_tags.values[np.where(mesh_tags.indices == c)][0]
                if display_indices:
                    vertices = [vertex_map[v] for v in connectivity_cells_to_vertices.links(c)]
                    midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
                    ax.text(midpoint[0], midpoint[1], f"{c}", horizontalalignment="center", verticalalignment="center", fontsize=6)
            else:
                cell_colors[c] = -1  # Handle cells without tags (optional)
        mappable = ax.tripcolor(tria,
                                cell_colors,
                                edgecolor="k",
                                cmap=cmap,
                                norm=norm)
        tag_dict = {0: "No tag",
                    1: "Interior cells",
                    2: "Cut cells",
                    3: "Exterior cells",
                    4: "Padding cells"}
    elif mesh_tags.dim == mesh.topology.dim - 1:
        tdim = mesh.topology.dim
        connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
        connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
        connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
        vertex_map = {
            topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
                connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
        }
        lines = list()
        lines_colors_as_int = list()
        lines_colors_as_str = list()
        lines_linestyles = list()
        for c in range(num_cells):
            facets = connectivity_cells_to_facets.links(c)
            for f in facets:
                if f in mesh_tags.indices:
                    value_f = mesh_tags.values[np.where(mesh_tags.indices == f)][0]
                else:
                    value_f = -1  # Handle facets without tags (optional)
                vertices = [vertex_map[v] for v in connectivity_facets_to_vertices.links(f)]
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(cmap(value_f) if value_f != -1 else "gray")
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
                                                  linewidth=0.5)
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(cast(Collection[Any], mappable))
        ax.autoscale()
        tag_dict = {0: "No tag",
                    1: "Interior facets",
                    2: "Cut facets",
                    3: "Exterior facets",
                    4: "Gamma_h"}
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(mappable, cax=cax, boundaries=np.arange(5) - 0.5, ticks=np.arange(5))
    
    # Set colorbar labels
    colorbar.set_ticklabels([f"{tag_dict[key]} ({key})" for key in tag_dict.keys()])

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

        ax.contour(xx, yy, zz, [0.], linewidths=0.5)
    return ax

def plot_mesh(
    mesh: Mesh,
    bbox: npt.NDArray,
    ax: plt.Axes | None = None,
    expression_levelset: Callable[..., np.ndarray] | None = None
) -> plt.Axes:
    """Plot a mesh.
    
    Args:
        mesh: the corresponding mesh.
        bbox: the domain bounding box.
        ax: (optional) the matplotlib axes.
        expression_levelset: (optional), if not None, display the contour line of the levelset.
    
    Returns:
        A matplotlib axis with the corresponding plot.
    """
    if ax is None:
        ax = plt.gca()  # type: ignore
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(bbox[0,0], bbox[0,1])
    ax.set_ylim(bbox[1,0], bbox[1,1])
    points = mesh.geometry.x

    mappable: mpl_collections.Collection

    cells = mesh.geometry.dofmap
    num_cells = cells.shape[0]
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    c = np.ones(num_cells)
    # create a colormap with a single color
    cmap = mcolors.ListedColormap("white")
    mappable = ax.tripcolor(tria,
                            facecolors=c,
                            edgecolor="k",
                            linewidth=0.75,
                            cmap=cmap)

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

        ax.contour(xx, yy, zz, [0.], linewidths=0.5)
    
    return ax


def plot_dg0_function(
    mesh: Mesh,
    function: Function,
    ax: plt.Axes | None = None,
    expression_levelset: Callable[..., np.ndarray] | None = None,
    vbounds: tuple = (-1., 1.),
    label: str | None = None,
    cmap_name: str = "RdYlBu",
    display_legend: bool = True,
    display_axes: bool = True) -> plt.Axes:
    """Plot a mesh tags object on the provided (or, if None, the current) axes object.
    
    Args:
        mesh: the corresponding mesh.
        mesh_tags: the mesh tags.
        ax: (optional) the matplotlib axes.
        display_indices: (optional) boolean, if True displays the indices of the cells/facets.
        expression_levelset: (optional), if not None, display the contour line of the levelset.
    
    Returns:
        A matplotlib axis with the corresponding plot.
    """
    if ax is None:
        ax = plt.gca()  # type: ignore
    
    if not display_axes:
        ax.set_axis_off()
    ax.set_aspect("equal")
    points = mesh.geometry.x

    # Get unique tags and create a custom colormap
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vbounds[0], vmax=vbounds[1])

    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_global

    mappable: mpl_collections.Collection
    cells = mesh.geometry.dofmap
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    cell_colors = np.zeros((cells.shape[0], ))

    for c in range(num_cells):
        cell_colors[c] = function.x.array[c]
    mappable = ax.tripcolor(tria,
                            cell_colors,
                            edgecolor="k",
                            cmap=cmap,
                            norm=norm)
    divider = make_axes_locatable(ax)
    if display_legend:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = plt.colorbar(mappable, cax=cax, norm=norm)
        legend_label = ""
        if label is not None:
            legend_label = label
        colorbar.set_label(legend_label)


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

        ax.contour(xx, yy, zz, [0.], linewidths=1., colors='w')
    return ax