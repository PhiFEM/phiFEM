from   basix.ufl    import element
import dolfinx      as dfx
from   dolfinx.fem  import Function
from   dolfinx.io   import XDMFFile
from   dolfinx.mesh import Mesh
import os
from   os import PathLike
import pandas as pd

PathStr = PathLike[str] | str

class ResultsSaver:
    """ Class used to save results."""

    def __init__(self, output_path: PathStr) -> None:
        """ Initialize a ResultsSaver object.

        Args:
            output_path: Path object or str, the directory path where the results are saved.
        """
        self.output_path: PathStr = output_path
        self.results: dict[str, list[float]] | None  = None

        if not os.path.isdir(output_path):
	        print(f"{output_path} directory not found, we create it.")
	        os.mkdir(os.path.join(".", output_path))

        file_path = os.path.join(output_path, "results.csv")
        if os.path.isfile(file_path):
            print(f"{file_path} found, we clear it.")
            os.remove(file_path)

        output_functions_path = os.path.join(output_path, "functions/")
        if not os.path.isdir(output_functions_path):
	        print(f"{output_functions_path} directory not found, we create it.")
	        os.mkdir(output_functions_path)
        
    def add_new_value(self, key: str, value: float) -> None:
        """ Add a new value to the results.

        Args:
            key: str, the key where the value must be added.
            value: float, the value to be added.
        """
        if self.results is None:
            self.results = {}
        if key in self.results.keys():
            self.results[key].append(value)
        else:
            self.results[key] = [value]
    
    def save_function(self, function: Function, file_name: str) -> None:
        """ Save a function to the disk.

        Args:
            function: dolfinx.fem.Function, the finite element function to save.
            file_name: str, the name of the XDMF file storing the function.
        """
        if function is None:
            raise ValueError("function is None.")
        element_family = function.function_space.element.basix_element.family.name
        mesh = function.function_space.mesh
        degree = function.function_space.element.basix_element.degree
        if degree > 1:
            mesh_element = element(element_family, mesh.topology.cell_name(), 1)
            mesh_space = dfx.fem.functionspace(mesh, mesh_element)
            interp = dfx.fem.Function(mesh_space)
            interp.interpolate(function)
        else:
            interp = function

        with XDMFFile(mesh.comm, os.path.join(self.output_path, "functions",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(interp)

    def save_mesh(self, mesh: Mesh, file_name: str) -> None:
        """ Save a mesh to the disk.

        Args:
            mesh: dolfinx.mesh.Mesh, the mesh to save.
            file_name: str, the name of the XDMF file storing the mesh.
        """
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "meshes",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
    
    def save_values(self, file_name: str) -> None:
        """ Convert the values to a pandas DataFrame and save it to the disk.

        Args:
            file_name: str, name of the csv file storing the dataframe.
        """
        df = pd.DataFrame(self.results)
        cols = sorted(list(df.columns.values))
        cols.remove("dofs")
        cols.insert(0, "dofs")
        df = df[cols]
        df.to_csv(os.path.join(self.output_path, file_name))
        print(df)