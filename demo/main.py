import argparse
import importlib
import inspect
import os
import yaml

from phiFEM.phifem.poisson_dirichlet import PhiFEMRefinementLoop, FEMRefinementLoop
from phiFEM.phifem.continuous_functions import Levelset

parent_dir = os.path.dirname(__file__)

demos_list = [demo for demo in next(os.walk("."))[1] if "__" not in demo]

parser = argparse.ArgumentParser(prog="Run the demo.",
                                 description="Run iterations of FEM or phiFEM with uniform or adaptive refinement on the given test case.")

parser.add_argument("test_case", type=str, choices=demos_list)

parser.add_argument("solver", 
                    type=str,
                    choices=["FEM", "phiFEM"],
                    help="Finite element solver.")

args   = parser.parse_args()
solver = args.solver
demo   = args.test_case

source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + solver)
if not os.path.isdir(output_dir):
    print(f"{output_dir} directory not found, we create it.")
    os.mkdir(output_dir)

module_name = demo + ".data"
module      = importlib.import_module(module_name)
expressions = {name: func for name, func in inspect.getmembers(module, inspect.isfunction) if name.startswith("expression_")}

levelset = Levelset(expressions["expression_levelset"])
if "expression_detection_levelset" in expressions.keys():
    levelset.set_detection_expression(expressions["expression_detection_levelset" ])
if "expression_u_exact" not in expressions.keys():
    expressions["expression_u_exact"] = None
if "expression_rhs" not in expressions.keys():
    expressions["expression_rhs"] = None

with open(os.path.join(source_dir, "parameters.yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

initial_mesh_size       = parameters["initial_mesh_size"]
num_it                  = parameters["iterations_number"]
ref_method              = parameters["refinement_method"]
stabilization_parameter = parameters["stabilization_parameter"]

if solver=="phiFEM":
    phifem_loop = PhiFEMRefinementLoop(initial_mesh_size,
                                       num_it,
                                       ref_method,
                                       levelset,
                                       stabilization_parameter,
                                       source_dir)
    phifem_loop.set_parameters(parameters, expressions)
    phifem_loop.create_initial_bg_mesh()
    phifem_loop.run()

if solver=="FEM":
    fem_loop = FEMRefinementLoop(initial_mesh_size,
                                 num_it,
                                 ref_method,
                                 levelset,
                                 source_dir,
                                 geometry_vertices=module.geom_vertices)

    fem_loop.set_parameters(parameters, expressions)
    fem_loop.mesh2d_from_levelset()
    fem_loop.run()
