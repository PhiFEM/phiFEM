import adios4dolfinx
import argparse
from   basix.ufl import element
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   dolfinx.io.gmshio import model_to_mesh
import gmsh
from   mpi4py import MPI
import os
import yaml

def generate_mesh(lc, output_dir):
    gmsh.initialize()
    outer_boundary  = gmsh.model.occ.addCircle(0., 0., 0., 0.5)
    gmsh.model.occ.addCurveLoop([outer_boundary],  1)
    inside = gmsh.model.occ.addPlaneSurface([1])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.addPhysicalGroup(2, [inside], name="inside")
    gmsh.model.mesh.generate(2)

    mesh, _, _ = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()

    with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "mesh.xdmf"), 'w') as of:
        of.write_mesh(mesh)
    return mesh

def generate_levelset(levelset_space, output_dir):
    if not os.path.isdir(output_dir):
        print(f"{output_dir} directory not found, we create it.")
        os.mkdir(os.path.join(parent_dir, output_dir))

    def levelset(x):
        return x[0,:]**2+x[0,:]**2-0.25**2

    phih = dfx.fem.Function(levelset_space)
    phih.interpolate(levelset)

    adios4dolfinx.write_function(os.path.join(output_dir, "functions", f"levelset_{str(0).zfill(2)}.bp"), phih, name="levelset")
    return phih

if __name__=="__main__":
    parent_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(prog="Run the demo.",
                                    description="Run iterations of FEM or phiFEM with uniform or adaptive refinement on the given test case.")

    parser.add_argument("parameters", type=str, help="Name of parameters file (without yaml extension).")

    args = parser.parse_args()
    parameters = args.parameters

    parameters_path = os.path.join(parent_dir, parameters + ".yaml")
    output_dir = os.path.join(parent_dir, parameters)

    with open(parameters_path, "rb") as f:
        parameters = yaml.safe_load(f)

    lc              = parameters["initial_mesh_size"]
    levelset_degree = parameters["levelset_degree"]

    mesh = generate_mesh(lc, output_dir)

    cell_name = mesh.topology.cell_name()
    levelset_element = element("Lagrange", cell_name, levelset_degree)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)
    _  = generate_levelset(levelset_space, output_dir)