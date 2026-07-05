import dolfinx as dfx
import numpy as np
from dolfinx.fem import Function
from dolfinx.mesh import Mesh, MeshTags

from phifem import measures, tags


class PhiFEM:
    def __init__(self, mesh: Mesh, discrete_levelset: Function):
        self.mesh = mesh
        self.levelset = discrete_levelset
        self.box_mode = True

    def tag_cells(self, detection_degree: int, single_layer_cut: bool = False):
        self.detection_degree = detection_degree
        self.cells_tags = tags.cells(
            self.mesh,
            self.levelset,
            detection_degree,
            single_layer_cut=single_layer_cut,
        )

    def tag_facets(self):
        try:
            assert self.cells_tags is not None
        except AssertionError:
            print("You must first tag the cells before tagging the facets.")

        self.facets_tags = tags.facets(
            self.mesh, self.cells_tags, self.levelset, self.detection_degree
        )

    def overwrite_cells_tags(self, new_cells_tags: MeshTags):
        if new_cells_tags.dim != self.mesh.topology.dim:
            raise ValueError(
                "The MeshTags object you want to use to overwrite is of different dimension to the mesh cells dimension."
            )

        if np.any(np.isin([1, 2, 3], new_cells_tags.values)):
            raise ValueError("Cannot overwrite cells tags with values 1, 2 or 3.")
        self.cells_tags = tags.overwrite(self.mesh, self.cells_tags, new_cells_tags)

    def overwrite_facets_tags(self, new_facets_tags: MeshTags):
        if new_facets_tags.dim != self.mesh.topology.dim - 1:
            raise ValueError(
                "The MeshTags object you want to use to overwrite is of different dimension to the mesh cells dimension."
            )

        if np.any(np.isin([1, 2, 3, 4, 5, 6, 100, 101], new_facets_tags.values)):
            raise ValueError(
                "Cannot overwrite facets tags with values 1, 2, 3, 4, 5, 6, 100 or 101."
            )
        self.facets_tags = tags.overwrite(self.mesh, self.facets_tags, new_facets_tags)

    def restrict_mesh(self):

        if self.cells_tags is None:
            raise ValueError("You must tag the cells before restricting the mesh.")

        # We create the submesh
        omega_h_cells = np.unique(
            np.hstack([self.cells_tags.find(1), self.cells_tags.find(2)])
        )
        submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(
            self.mesh, self.mesh.topology.dim, omega_h_cells
        )  # type: ignore

        self.cells_tags = tags.transfer(self.cells_tags, submesh, c_map)
        if self.facets_tags is not None:
            self.facets_tags = tags.transfer(
                self.facets_tags, submesh, c_map, source_mesh=self.mesh
            )
        self.submesh_maps = [c_map, v_map, n_map]
        self.mesh = submesh
        self.box_mode = False

    def compute_ds(self):
        self.ds = measures.one_sided_boundary(
            self.mesh, self.cells_tags, self.facets_tags, self.box_mode
        )
