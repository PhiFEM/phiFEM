import adios4dolfinx
import argparse
from   basix.ufl import element, mixed_element
import numpy as np
import dolfinx as dfx
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.io import XDMFFile
import os
import petsc4py.PETSc as PETSc
import sys
import ufl
import yaml

# Import phiFEM modules
from phiFEM.phifem.mesh_scripts import compute_tags, compute_outward_normal
from initial_mesh_levelset import generate_mesh

def save_function(fct, file_name):
    mesh = fct.function_space.mesh
    fct_element = fct.function_space.element.basix_element
    deg = fct_element.degree
    if deg > 1:
        element_family = fct_element.family.name
        mesh = fct.function_space.mesh
        cg1_element = element(element_family, mesh.topology.cell_name(), 1, shape=fct.function_space.value_shape)
        cg1_space = dfx.fem.functionspace(mesh, cg1_element)
        cg1_fct = dfx.fem.Function(cg1_space)
        cg1_fct.interpolate(fct)
    else:
        cg1_fct = fct

    with XDMFFile(mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w") as of:
        of.write_mesh(mesh)
        of.write_function(cg1_fct)

parent_dir = os.path.dirname(__file__)

E = 1.0
nu = 0.3
# Lamé coefficients
lmbda = E * nu/(1.0+nu)/(1.+2.*nu)
mu = E/2.0/(1.0+nu)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)

def solve_primal_problem(i, mesh, spaces, measures, levelset, data_fcts):
    mixed_space, dg0_space = spaces
    f, g = data_fcts

    dx = measures["dx"]
    dS = measures["dS"]
    dBoundary = measures["dBoundary"]

    h_T = ufl.CellDiameter(mesh)
    h_E = ufl.FacetArea(mesh)
    n   = ufl.FacetNormal(mesh)

    u, y, p = ufl.TrialFunctions(mixed_space)
    v, z, q = ufl.TestFunctions(mixed_space)

    Omega_h_n = compute_outward_normal(mesh, levelset)
    Omega_h_indicator = dfx.fem.Function(dg0_space)
    Omega_h_indicator.x.petsc_vec.set(0.)
    interior_cells = cells_tags.find(1)
    cut_cells      = cells_tags.find(2)
    Omega_h_cells = np.union1d(interior_cells, cut_cells)
    Omega_h_indicator.x.array[Omega_h_cells] = 1.

    boundary = ufl.inner(2. * ufl.avg(ufl.dot(y, Omega_h_n) * Omega_h_indicator), 2. * ufl.avg(v * Omega_h_indicator))

    stiffness = ufl.inner(sigma(u), epsilon(v))

    penalization = penalization_coefficient * \
                ( ufl.inner(y + sigma(u), z + sigma(v)) \
                + ufl.inner(ufl.div(y) + u, ufl.div(z) + v) \
    + h_T**(-2) * ufl.inner(ufl.dot(y, ufl.grad(levelset)) + h_T**(-1) * p * levelset, ufl.dot(z, ufl.grad(levelset)) + h_T**(-1) * q * levelset))

    stabilization_facets = stabilization_coefficient * ufl.avg(h_E) * \
                          ufl.inner(ufl.jump(sigma(u), n), ufl.jump(sigma(v), n))
    
    a = stiffness            * (dx(1) + dx(2)) \
      + boundary             * dBoundary \
      + penalization         * dx(2) \
      + stabilization_facets * dS(2)
    
    rhs = ufl.inner(f, v)
    penalization_rhs = penalization_coefficient * \
                    (-h_T**(-2) * ufl.inner(g, ufl.sqrt(ufl.inner(ufl.grad(levelset), ufl.grad(levelset))) * (ufl.dot(z, ufl.grad(levelset)) + h_T**(-1) * q * levelset)) \
                    + ufl.inner(f, ufl.div(z) + v))

    L = rhs                * (dx(1) + dx(2)) \
        + penalization_rhs * dx(2)
    
    bilinear_form = dfx.fem.form(a)
    linear_form   = dfx.fem.form(L)

    A = assemble_matrix(bilinear_form)
    b = assemble_vector(linear_form)
    A.assemble()

    """
    Set up the PETSc solver
    """
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Configure MUMPS to handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    
    """
    Solve
    """
    solution_wh = dfx.fem.Function(mixed_space)

    # Monitor PETSc solve time
    viewer = PETSc.Viewer().createASCII(os.path.join(output_dir, "petsc_log.txt"))
    PETSc.Log.begin()
    ksp.solve(b, solution_wh.x.petsc_vec)
    PETSc.Log.view(viewer)
    ksp.destroy()

    # Get solve time from PETSc logs
    with open(os.path.join(output_dir, "petsc_log.txt")) as fi:
        for line in fi.readlines():
            if "Time (sec):" in line:
                print("Solve time (sec):", float(line[22:31]))

    os.remove(os.path.join(output_dir, "petsc_log.txt"))

    solution_uh, _, _ = solution_wh.split()
    solution_uh.collapse()

    adios4dolfinx.write_function(os.path.join(output_dir, "functions", "uh.bp"), solution_uh, time=i/imax, name="Displacement")

    save_function(solution_uh,       f"uh_{str(i).zfill(2)}")
    save_function(levelset,          f"levelset_{str(i).zfill(2)}")
    save_function(Omega_h_indicator, f"indicator_{str(i).zfill(2)}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Run the demo.",
                                     description="Solve the shape optimization primal problem.")

    parser.add_argument("parameters", type=str, help="Name of parameters file (without yaml extension).")

    args = parser.parse_args()
    parameters = args.parameters

    parameters_path = os.path.join(parent_dir, parameters + ".yaml")
    output_dir = os.path.join(parent_dir, parameters)

    if not os.path.isdir(output_dir):
        print(f"{output_dir} directory not found, we create it.")
        os.mkdir(os.path.join(parent_dir, output_dir))

    # Import data functions (levelset, source term...)
    test_case = parameters.split(sep="/")[0]
    test_case_path = os.path.join(parent_dir, test_case)
    if test_case_path not in sys.path:
        sys.path.insert(0, test_case_path)

    from data import levelset, source_term, neumann

    with open(parameters_path, "rb") as f:
        parameters = yaml.safe_load(f) 

    # Extract parameters
    primal_degree             = parameters["primal_degree"]
    auxiliary_degree          = parameters["auxiliary_degree"]
    flux_degree               = parameters["flux_degree"]
    levelset_degree           = parameters["levelset_degree"]
    initial_mesh_size         = parameters["initial_mesh_size"]
    boundary_detection_degree = parameters["boundary_detection_degree"]
    stabilization_coefficient = parameters["stabilization_coefficient"]
    penalization_coefficient  = parameters["penalization_coefficient"]
    imax = 1

    mesh = generate_mesh(initial_mesh_size, output_dir)

    cell_name = mesh.topology.cell_name()
    gdim = mesh.geometry.dim
    primal_element = element("Lagrange", 
                             cell_name,
                             primal_degree,
                             shape=(gdim,))
    flux_element = element("Lagrange", 
                           cell_name,
                           flux_degree,
                           shape=(gdim,gdim))
    
    if auxiliary_degree == 0:
        auxiliary_element_family = "DG"
    else:
        auxiliary_element_family = "Lagrange"
    auxiliary_element = element(auxiliary_element_family,
                                cell_name,
                                auxiliary_degree,
                                shape=(gdim,))

    mixd_element = mixed_element([primal_element,
                                  flux_element,
                                  auxiliary_element])

    primal_space    = dfx.fem.functionspace(mesh, primal_element)
    auxiliary_space = dfx.fem.functionspace(mesh, auxiliary_element)
    vector_space    = dfx.fem.functionspace(mesh, flux_element)
    mixed_space     = dfx.fem.functionspace(mesh, mixd_element)

    levelset_element = element("Lagrange", cell_name, levelset_degree)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    dg0_element = element("DG", cell_name, 0)
    dg0_space   = dfx.fem.functionspace(mesh, dg0_element)

    spaces = (mixed_space, dg0_space)

    phi_h = dfx.fem.Function(levelset_space)
    phi_h.interpolate(levelset)
    f = dfx.fem.Function(primal_space)
    f.interpolate(source_term)
    g = dfx.fem.Function(primal_space)
    g.interpolate(neumann)

    data_fcts = (f, g)
    import matplotlib.pyplot as plt
    from utils.mesh_scripts import plot_mesh_tags

    for i in range(imax):
        # Compute mesh tags
        cells_tags, facets_tags, _ = compute_tags(mesh, levelset, boundary_detection_degree, box_mode=True)

        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(mesh, cells_tags, ax, expression_levelset=levelset)
        plt.savefig("cells_tags.png", bbox_inches="tight")

        dx = ufl.Measure("dx",
                        domain=mesh,
                        subdomain_data=cells_tags)
        dS = ufl.Measure("dS",
                        domain=mesh,
                        subdomain_data=facets_tags)
        dBoundary = dS(4)
        measures = {"dx": dx, "dS": dS, "dBoundary": dBoundary}

        solve_primal_problem(i, mesh, spaces, measures, phi_h, data_fcts)

    # Removes the test-case path from system path
    if test_case_path in sys.path:
        sys.path.remove(test_case_path)