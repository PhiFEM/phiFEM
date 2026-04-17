import dolfinx as dfx
from basix.ufl import element
from dolfinx.io import XDMFFile


def save_tags(mesh, file_path, mesh_tags):
    """Save a MeshTags object to xdmf as a DG0 function.

    Args:
        mesh: the mesh on which the MeshTags is defined.
        file_path: the file path of the xdmf file.
        mesh_tags: the MeshTags object to save.
    """

    try:
        mesh.topology.dim == mesh_tags.dim
    except ValueError:
        print("The Mesh and MeshTags objects must have the same topological dimension.")

    cell_name = mesh.topology.cell_name()
    dg0_element = element("DG", cell_name, 0)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)
    dg0_mesh_tags = dfx.fem.Function(dg0_space)

    dg0_mesh_tags.x.array[:] = mesh_tags.values

    with XDMFFile(
        mesh.comm,
        file_path,
        "w",
    ) as of:
        of.write_mesh(mesh)
        of.write_function(dg0_mesh_tags)


def save_levelset(mesh: dfx.mesh.Mesh, file_path: str, np_levelset: callable):
    """Save a 'numpy' callable levelset to xdmf.

    Args:
        mesh: the mesh on which the levelset is discretized.
        file_path: the file path of the xdmf file.
        np_levelset: the callable taking and returning numpy arrays defining the levelset.
    """

    cell_name = mesh.topology.cell_name()
    cg1_element = element("Lagrange", cell_name, 1)
    cg1_space = dfx.fem.functionspace(mesh, cg1_element)
    cg1_levelset = dfx.fem.Function(cg1_space)
    cg1_levelset.interpolate(np_levelset)

    with XDMFFile(
        mesh.comm,
        file_path,
        "w",
    ) as of:
        of.write_mesh(mesh)
        of.write_function(cg1_levelset)
