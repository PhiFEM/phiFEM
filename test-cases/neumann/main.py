import argparse
from   basix.ufl import element, mixed_element
import numpy as np
import dolfinx as dfx
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import os
import polars as pl
import petsc4py.PETSc as PETSc
import sys
import ufl
import yaml

# Import phiFEM modules
from phiFEM.phifem.mesh_scripts import compute_tags_measures, compute_outward_normal, marking

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog="Run the demo.",
                                 description="Run iterations of FEM or phiFEM with uniform or adaptive refinement on the given test case.")

parser.add_argument("parameters", type=str, help="Name of parameters file (without yaml extension).")

args = parser.parse_args()
parameters = args.parameters

parameters_path = os.path.join(parent_dir, parameters + ".yaml")
output_dir = os.path.join(parent_dir, parameters)

if not os.path.isdir(output_dir):
	print(f"{output_dir} directory not found, we create it.")
	os.mkdir(os.path.join(parent_dir, output_dir))

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
        with XDMFFile(mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(cg1_fct)
    else:
        with XDMFFile(mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(fct)

# Import data functions (levelset, source term...)
test_case = parameters.split(sep="/")[0]
test_case_path = os.path.join(parent_dir, test_case)
if test_case_path not in sys.path:
    sys.path.insert(0, test_case_path)

from data import levelset, source_term

try:
    from data import detection_levelset
except ImportError:
    print("detection_levelset not found in data, use levelset instead.")
    detection_levelset = levelset

with open(parameters_path, "rb") as f:
    parameters = yaml.safe_load(f)

# Extract parameters
tolerance                 = parameters["tolerance"]
primal_degree             = parameters["primal_degree"]
auxiliary_degree          = parameters["auxiliary_degree"]
vector_degree             = parameters["vector_degree"]
levelset_degree           = parameters["levelset_degree"]
bbox                      = parameters["bbox"]
mesh_size                 = parameters["initial_mesh_size"]
boundary_detection_degree = parameters["boundary_detection_degree"]
stabilization_coefficient = parameters["stabilization_coefficient"]
penalization_coefficient  = parameters["penalization_coefficient"]
box_mode                  = parameters["box_mode"]
refinement_method         = parameters["refinement_method"]
bc_estimator              = parameters["bc_estimator"]
reference_error           = parameters["reference_error"]

# Create the initial background mesh
nx = int(np.abs(bbox[0][1] - bbox[0][0]) * np.sqrt(2.) / mesh_size)
ny = int(np.abs(bbox[1][1] - bbox[1][0]) * np.sqrt(2.) / mesh_size)
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, np.asarray(bbox).T, [nx, ny])

# Results storage
results = {"dofs":              [],
           "H1 estimator":      [],
           "H1 estimator rate": [0.0],
           "T estimator":       [],
           "E estimator":       []}

if bc_estimator:
    results["bc estimator"] = []

if reference_error:
    results["H1 error"]      = []
    results["H1 error rate"] = [0.0]

results["Solve time"] = []

estimator = np.inf
for i in range(20):
    # Compute mesh tags
    cells_tags, facets_tags, mesh, d_boundary_from_inside, _ = compute_tags_measures(mesh, detection_levelset, boundary_detection_degree, box_mode=box_mode)

    # Defines finite elements
    cell_name = mesh.topology.cell_name()

    primal_element    = element("Lagrange", cell_name, primal_degree)
    if auxiliary_degree == 0:
        auxiliary_element_family = "DG"
    else:
        auxiliary_element_family = "Lagrange"
    auxiliary_element = element(auxiliary_element_family, cell_name, auxiliary_degree)
    vector_element           = element("Lagrange", cell_name, vector_degree, shape=(mesh.geometry.dim,))
    mixd_element             = mixed_element([primal_element, vector_element, auxiliary_element])

    levelset_element         = element("Lagrange", cell_name, levelset_degree)
    dg0_element              = element("DG",       cell_name, 0)

    dg0_element       = element("DG", mesh.topology.cell_name(), 0)
    levelset_element  = element("Lagrange", mesh.topology.cell_name(), levelset_degree)

    primal_space    = dfx.fem.functionspace(mesh, primal_element)
    auxiliary_space = dfx.fem.functionspace(mesh, auxiliary_element)
    vector_space    = dfx.fem.functionspace(mesh, vector_element)
    mixed_space     = dfx.fem.functionspace(mesh, mixd_element)

    interior_cells = cells_tags.find(1)
    cut_cells      = cells_tags.find(2)
    omega_h_cells  = np.union1d(interior_cells, cut_cells)
    cdim = mesh.topology.dim
    mesh.topology.create_connectivity(cdim, cdim)
    num_active_dofs = len(dfx.fem.locate_dofs_topological(primal_space, cdim, omega_h_cells))
    num_active_dofs += len(dfx.fem.locate_dofs_topological(auxiliary_space, cdim, cut_cells))
    num_active_dofs += len(dfx.fem.locate_dofs_topological(vector_space, cdim, cut_cells))

    dg0_space      = dfx.fem.functionspace(mesh, dg0_element)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    phi_h = dfx.fem.Function(levelset_space)
    f_h   = dfx.fem.Function(primal_space)
    phi_h.interpolate(levelset)
    f_h.interpolate(source_term)

    # Neumann data
    u_N = dfx.fem.Function(primal_space)
    try:
        from data import neumann
        u_N.interpolate(neumann)
    except ImportError:
        from data import exact_solution
        norm_grad_phi_h = ufl.sqrt(ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h)))
        grad_phi_h_normalized = ufl.grad(phi_h)/norm_grad_phi_h

        exact_solution_h = dfx.fem.Function(primal_space)
        exact_solution_h.interpolate(exact_solution)

        u_N = ufl.inner(ufl.grad(exact_solution_h), grad_phi_h_normalized)

    u, y, p = ufl.TrialFunctions(mixed_space)
    v, z, q = ufl.TestFunctions(mixed_space)
    results["dofs"].append(num_active_dofs)

    phi_h = dfx.fem.Function(levelset_space)
    phi_h.interpolate(levelset)
    f_h   = dfx.fem.Function(primal_space)
    f_h.interpolate(source_term)

    dx = ufl.Measure("dx",
                     domain=mesh,
                     subdomain_data=cells_tags)
    dS = ufl.Measure("dS",
                     domain=mesh,
                     subdomain_data=facets_tags)
    dBoundary = ufl.Measure("ds",
                            domain=mesh)

    h_T = ufl.CellDiameter(mesh)
    h_E = ufl.FacetArea(mesh)
    n   = ufl.FacetNormal(mesh)

    """
    φ-FEM formulation
    """
    omega_h_indicator = 1.
    boundary = ufl.inner(ufl.inner(y, n), v)
    if box_mode:
        d_boundary = d_boundary_from_inside
    else:
        d_boundary = ufl.ds

    stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)

    penalization = penalization_coefficient * \
                 (ufl.inner(y + ufl.grad(u), z + ufl.grad(v)) \
                 + ufl.inner(ufl.div(y) + u, ufl.div(z) + v) \
                 + h_T**(-2) * ufl.inner(ufl.inner(y, ufl.grad(phi_h)) + h_T**(-1) * ufl.inner(p, phi_h), ufl.inner(z, ufl.grad(phi_h)) + h_T**(-1) * ufl.inner(q, phi_h)))

    stabilization_facets = stabilization_coefficient * ufl.avg(h_E) * \
                            ufl.inner(ufl.jump(ufl.grad(u), n),
                                      ufl.jump(ufl.grad(v), n))

    # The φ-FEM bilinear form
    a = stiffness              * (dx(1) + dx(2)) \
        + boundary             * d_boundary \
        + penalization         * dx(2) \
        + stabilization_facets * dS(3)

    rhs = ufl.inner(f_h, v)
    penalization_rhs = penalization_coefficient * (- h_T**(-2) * ufl.inner(u_N, ufl.sqrt(ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h))) * (ufl.inner(z, ufl.grad(phi_h)) + h_T**(-1) * ufl.inner(q, phi_h))) \
                       + ufl.inner(f_h, ufl.div(z) + v))

    L = rhs                 * (dx(1) + dx(2)) \
        + penalization_rhs  * dx(2)

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

    solution_uh, _, solution_ph = solution_wh.split()
    solution_uh.collapse()

    save_function(solution_uh, f"uh_{str(i).zfill(2)}")

    """
    A posteriori error estimation
    """
    r = f_h + ufl.div(ufl.grad(solution_uh)) - solution_uh
    J_h = ufl.jump(ufl.grad(solution_uh), n)

    w0 = ufl.TestFunction(dg0_space)
    # Interior residual
    eta_T = h_T**2 * ufl.inner(ufl.inner(r, r), w0) * (dx(1) + dx(2))

    eta_T_form = dfx.fem.form(eta_T)
    eta_T_vec = assemble_vector(eta_T_form)
    eta_T_h = dfx.fem.Function(dg0_space)
    eta_T_h.x.petsc_vec.setArray(eta_T_vec.array[:])
    T_est = np.sqrt(eta_T_vec.array.sum())

    save_function(eta_T_h, f"eta_T_{str(i).zfill(2)}")

    # Facets residual (must use a restriction in box mode to avoid averaging both sides of the boundary of Omega_h)
    if box_mode:
        eta_E = ufl.avg(h_E) * ufl.inner(ufl.inner(J_h, J_h), ufl.avg(w0)) * ufl.avg(omega_h_indicator) * (dS(1) + dS(2))
    else:
        eta_E = ufl.avg(h_E) * ufl.inner(ufl.inner(J_h, J_h), ufl.avg(w0)) * (dS(1) + dS(2))
    eta_E_form = dfx.fem.form(eta_E)
    eta_E_vec = assemble_vector(eta_E_form)
    eta_E_h = dfx.fem.Function(dg0_space)
    eta_E_h.x.petsc_vec.setArray(eta_E_vec.array[:])
    E_est = np.sqrt(eta_E_vec.array.sum())

    save_function(eta_E_h, f"eta_E_{str(i).zfill(2)}")
    
    # Weak DBC residual
    if bc_estimator:
        eta_dbc_L2 = h_T**(-1) * ufl.inner(ufl.inner(solution_ph * phi_h, solution_ph * phi_h), w0) * dx(2)
        eta_dbc_H1 = h_T * ufl.inner(ufl.inner(ufl.grad(solution_ph * phi_h), ufl.grad(solution_ph * phi_h)), w0) * dx(2)

        eta_dbc = eta_dbc_H1
        eta_dbc_form = dfx.fem.form(eta_dbc)
        eta_dbc_vec = assemble_vector(eta_dbc_form)
        eta_dbc_h = dfx.fem.Function(dg0_space)
        eta_dbc_h.x.petsc_vec.setArray(eta_dbc_vec.array[:])
        dbc_est = np.sqrt(eta_dbc_vec.array.sum())
        results["bc estimator"].append(dbc_est)
        save_function(eta_dbc_h, f"eta_dbc_{str(i).zfill(2)}")
    else:
        dbc_est = 0.
        eta_dbc = 0.

    eta = eta_T + eta_E + eta_dbc
    eta_form = dfx.fem.form(eta)
    eta_vec = assemble_vector(eta_form)
    eta_h = dfx.fem.Function(dg0_space)
    eta_h.x.petsc_vec.setArray(eta_vec.array[:])
    save_function(eta_h, f"residual_est_{str(i).zfill(2)}")

    residual_est = np.sqrt(eta_vec.array.sum())
    results["H1 estimator"].append(residual_est)
    results["T estimator"].append(T_est)
    results["E estimator"].append(E_est)

    if i>0:
        results["H1 estimator rate"].append((np.log(residual_est) - np.log(results["H1 estimator"][i-1]))/(np.log(results["dofs"][i]) - np.log(results["dofs"][i-1])))

    if reference_error:
        try:
            from data import exact_solution
        except ImportError:
            print(f"exact_solution not found in {test_case}/data.py, we skip the computation of the reference error.")
            pass
        
        # Total number of cells and facets across all processes
        local_num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        mesh_num_cells = MPI.COMM_WORLD.allreduce(local_num_cells, op=MPI.SUM)
        
        reference_mesh = dfx.mesh.create_submesh(mesh, mesh.topology.dim, np.arange(mesh_num_cells))[0]

        for _ in range(2):
            reference_mesh.topology.create_entities(reference_mesh.topology.dim - 1)
            reference_mesh, _, _ = dfx.mesh.refine(reference_mesh)
        
        local_num_cells = reference_mesh.topology.index_map(reference_mesh.topology.dim).size_local
        reference_num_cells = MPI.COMM_WORLD.allreduce(local_num_cells, op=MPI.SUM)
        
        reference_degree = max(primal_degree, levelset_degree) + 2
        reference_element = element("Lagrange", reference_mesh.topology.cell_name(), max(primal_degree, levelset_degree) + 2)
        reference_space = dfx.fem.functionspace(reference_mesh, reference_element)

        def Omega_indicator(x):
            return (detection_levelset(x) - np.abs(detection_levelset(x)))/(2. * detection_levelset(x))

        indicator_h = dfx.fem.Function(reference_space)
        indicator_h.interpolate(Omega_indicator)

        save_function(indicator_h, f"indicator_{str(i).zfill(2)}")

        u_exact_ref = dfx.fem.Function(reference_space)
        uh_ref = dfx.fem.Function(reference_space)
        u_exact_ref.interpolate(exact_solution)
        nmm = dfx.fem.create_interpolation_data(reference_space,
                                                primal_space,
                                                np.arange(reference_num_cells),
                                                padding=1.e-14)
        uh_ref.interpolate_nonmatching(solution_uh,
                                       np.arange(reference_num_cells),
                                       interpolation_data=nmm)
        diff = u_exact_ref - uh_ref
        reference_dg0_element = element("DG", reference_mesh.topology.cell_name(), 0)
        reference_dg0_space = dfx.fem.functionspace(reference_mesh, reference_dg0_element)
        v0 = ufl.TestFunction(reference_dg0_space)

        h10_error_int = ufl.inner(ufl.inner(ufl.grad(diff), ufl.grad(diff)), v0) * indicator_h * ufl.dx
        h10_error_form = dfx.fem.form(h10_error_int)
        h10_error_vec = assemble_vector(h10_error_form)
        h10_error_h = dfx.fem.Function(reference_dg0_space)
        h10_error_h.x.petsc_vec.setArray(h10_error_vec.array[:])
        save_function(h10_error_h, f"error_{str(i).zfill(2)}")

        error = np.sqrt(h10_error_vec.array.sum())

        results["H1 error"].append(error)
        if i>0:
            results["H1 error rate"].append((np.log(error) - np.log(results["H1 error"][i-1]))/(np.log(results["dofs"][i]) - np.log(results["dofs"][i-1])))

    # Get solve time from PETSc logs
    with open(os.path.join(output_dir, "petsc_log.txt")) as fi:
        for line in fi.readlines():
            if "Time (sec):" in line:
                results["Solve time"].append(float(line[22:31]))
    os.remove(os.path.join(output_dir, "petsc_log.txt"))
    
    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    if residual_est < tolerance:
        break

    try:
        from data import exact_solution
        u_exact = dfx.fem.Function(primal_space)
        u_exact.interpolate(exact_solution)
        save_function(u_exact, f"u_exact_{str(i).zfill(2)}")
    except ImportError:
        pass

    if refinement_method=="uniform":
        mesh, _, _ = dfx.mesh.refine(mesh)
    elif refinement_method=="adaptive":
        marked_facets = marking(eta_h)
        mesh, _, _ = dfx.mesh.refine(mesh, marked_facets)

# Removes the test-case path from system path
if test_case_path in sys.path:
    sys.path.remove(test_case_path)