import argparse
from   basix.ufl import element
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
from phiFEM.phifem.mesh_scripts import compute_tags, compute_outward_normal, compute_levelset_boundary_error, marking

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
        cg1_element = element(element_family, mesh.topology.cell_name(), 1)
        cg1_space = dfx.fem.functionspace(mesh, cg1_element)
        cg1_fct = dfx.fem.Function(cg1_space)
        cg1_fct.interpolate(fct)
    else:
        cg1_fct = fct

    with XDMFFile(mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w") as of:
        of.write_mesh(mesh)
        of.write_function(cg1_fct)

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
fe_degree                 = parameters["finite_element_degree"]
levelset_degree           = parameters["levelset_degree"]
solution_degree           = parameters["solution_degree"]
bbox                      = parameters["bbox"]
mesh_size                 = parameters["initial_mesh_size"]
boundary_detection_degree = parameters["boundary_detection_degree"]
stabilization_coefficient = parameters["stabilization_coefficient"]
box_mode                  = parameters["box_mode"]
boundary_correction       = parameters["boundary_correction"]
refinement_method         = parameters["refinement_method"]
reference_error           = parameters["reference_error"]

# Create the initial background mesh
nx = int(np.abs(bbox[0][1] - bbox[0][0]) * np.sqrt(2.) / mesh_size)
ny = int(np.abs(bbox[1][1] - bbox[1][0]) * np.sqrt(2.) / mesh_size)
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, np.asarray(bbox).T, [nx, ny])

# Results storage
results = {"dofs":               [],
           "H10 estimator":      [],
           "H10 estimator rate": [0.0],
           "Solve time":         []}

if reference_error:
    results["H10 error"] = []
    results["H10 error rate"] = [0.0]

estimator = np.inf
for i in range(50):
    # Compute mesh tags
    cells_tags, facets_tags, mesh = compute_tags(mesh, detection_levelset, boundary_detection_degree, box_mode=box_mode)

    # Defines finite elements
    primal_element   = element("Lagrange", mesh.topology.cell_name(), fe_degree)
    levelset_element = element("Lagrange", mesh.topology.cell_name(), levelset_degree)
    solution_element = element("Lagrange", mesh.topology.cell_name(), solution_degree)
    dg0_element      = element("DG", mesh.topology.cell_name(), 0)

    primal_space   = dfx.fem.functionspace(mesh, primal_element)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)
    solution_space = dfx.fem.functionspace(mesh, solution_element)
    dg0_space      = dfx.fem.functionspace(mesh, dg0_element)

    interior_cells = cells_tags.find(1)
    cut_cells      = cells_tags.find(2)
    Omega_h_cells  = np.union1d(interior_cells, cut_cells)
    cdim = mesh.topology.dim
    mesh.topology.create_connectivity(cdim, cdim)
    active_dofs = dfx.fem.locate_dofs_topological(primal_space, cdim, Omega_h_cells)
    results["dofs"].append(len(active_dofs))

    phi_h = dfx.fem.Function(levelset_space)
    phi_h.interpolate(levelset)
    f_h   = dfx.fem.Function(primal_space)
    f_h.interpolate(source_term)

    w = ufl.TrialFunction(primal_space)
    v = ufl.TestFunction(primal_space)

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
    # Bilinear form
    phiw = phi_h * w
    phiv = phi_h * v
    
    Omega_h_indicator = 1.
    # If box mode is used, the unit outward pointing normal to Omega_h has to be computed on internal edges
    if box_mode:
        Omega_h_n = compute_outward_normal(mesh, levelset)
        Omega_h_indicator = dfx.fem.Function(dg0_space)
        Omega_h_indicator.x.petsc_vec.set(0.)
        interior_cells = cells_tags.find(1)
        cut_cells      = cells_tags.find(2)
        Omega_h_cells = np.union1d(interior_cells, cut_cells)
        Omega_h_indicator.x.array[Omega_h_cells] = 1.

        boundary = ufl.inner(2. * ufl.avg(ufl.inner(ufl.grad(phiw), Omega_h_n) * Omega_h_indicator), 2. * ufl.avg(ufl.inner(phiv, Omega_h_indicator)))
        dBoundary = dS(4)
    else:
        boundary = ufl.inner(ufl.inner(ufl.grad(phiw), n), phiv)

    stiffness = ufl.inner(ufl.grad(phiw), ufl.grad(phiv))
    facets_stabilization = stabilization_coefficient * ufl.avg(h_E) \
                           * ufl.inner(ufl.jump(ufl.grad(phiw), n),
                                       ufl.jump(ufl.grad(phiv), n))
    cells_stabilization = stabilization_coefficient * h_T**2 \
                          * ufl.inner(ufl.div(ufl.grad(phiw)),
                                      ufl.div(ufl.grad(phiv)))

    a = stiffness              * (dx(1) + dx(2)) \
        - boundary             * dBoundary \
        + facets_stabilization * dS(2) \
        + cells_stabilization  * dx(2)
    
    bilinear_form = dfx.fem.form(a)

    # Linear form
    rhs = ufl.inner(f_h, phiv)
    rhs_stabilization = stabilization_coefficient * h_T**2 \
                        * ufl.inner(f_h, ufl.div(ufl.grad(phiv)))
    
    L = rhs                 * (dx(1) + dx(2)) \
        - rhs_stabilization * dx(2)

    linear_form = dfx.fem.form(L)

    A = assemble_matrix(bilinear_form)
    b = assemble_vector(linear_form)
    A.assemble()

    """
    Set up the PETSc solver
    """
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("lu")
    if box_mode:
        # Configure MUMPS to handle nullspace.
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    
    """
    Solve
    """
    solution_wh = dfx.fem.Function(primal_space)
    # Monitor PETSc solve time
    viewer = PETSc.Viewer().createASCII(os.path.join(output_dir, "petsc_log.txt"))
    PETSc.Log.begin()
    ksp.solve(b, solution_wh.x.petsc_vec)
    PETSc.Log.view(viewer)
    ksp.destroy()

    save_function(solution_wh, f"wh_{str(i).zfill(2)}")

    solution_uh = dfx.fem.Function(solution_space)
    solution_wh_s_space = dfx.fem.Function(solution_space)
    solution_wh_s_space.interpolate(solution_wh)
    phi_h_s_space = dfx.fem.Function(solution_space)
    phi_h_s_space.interpolate(phi_h)

    solution_uh.x.array[:] = solution_wh_s_space.x.array[:] \
                             * phi_h_s_space.x.array[:]

    save_function(solution_uh, f"uh_{str(i).zfill(2)}")

    """
    A posteriori error estimation
    """
    geometry_correction = 0.
    if boundary_correction is not None:
        if boundary_correction=="h":
            entities_tags = facets_tags
        elif boundary_correction=="p":
            entities_tags = cells_tags
        
        levelset_boundary_correction = compute_levelset_boundary_error(mesh, levelset, levelset_space, entities_tags, boundary_correction)
        levelset_boundary_correction
        correction_space = levelset_boundary_correction.function_space
        correction_wh = dfx.fem.Function(correction_space)
        correction_wh.interpolate(solution_wh)

        correction_function = dfx.fem.Function(correction_space)
        correction_function.x.array[:] = levelset_boundary_correction.x.array[:] * correction_wh.x.array[:]

        geometry_correction = ufl.inner(ufl.grad(correction_function),
                                        ufl.grad(correction_function))

    r = f_h + ufl.div(ufl.grad(solution_uh))
    J_h = ufl.jump(ufl.grad(solution_uh), n)

    w0 = ufl.TestFunction(dg0_space)
    # Interior residual
    eta_T = h_T**2 * ufl.inner(ufl.inner(r, r), w0) * (dx(1) + dx(2))

    eta_T_form = dfx.fem.form(eta_T)
    eta_T_vec = assemble_vector(eta_T_form)
    eta_T_h = dfx.fem.Function(dg0_space)
    eta_T_h.x.petsc_vec.setArray(eta_T_vec.array[:])

    save_function(eta_T_h, f"eta_T_{str(i).zfill(2)}")

    # Facets residual (must use a restriction in box mode to avoid averaging both sides of the boundary of Omega_h)
    if box_mode:
        eta_E = ufl.avg(h_E) * ufl.inner(ufl.inner(J_h, J_h), ufl.avg(w0)) * ufl.avg(Omega_h_indicator) * (dS(1) + dS(2))
    else:
        eta_E = ufl.avg(h_E) * ufl.inner(ufl.inner(J_h, J_h), ufl.avg(w0)) * (dS(1) + dS(2))
    eta_E_form = dfx.fem.form(eta_E)
    eta_E_vec = assemble_vector(eta_E_form)
    eta_E_h = dfx.fem.Function(dg0_space)
    eta_E_h.x.petsc_vec.setArray(eta_E_vec.array[:])

    save_function(eta_E_h, f"eta_E_{str(i).zfill(2)}")
    
    # Geometry residual (0 if boundary_correction is None)
    eta_geometry = ufl.inner(geometry_correction, w0) * (dx(1) + dx(2))

    if boundary_correction is not None:
        eta_G_form = dfx.fem.form(eta_geometry)
        eta_G_vec = assemble_vector(eta_G_form)
        eta_G_h = dfx.fem.Function(dg0_space)
        eta_G_h.x.petsc_vec.setArray(eta_G_vec.array[:])
        save_function(eta_G_h, f"eta_G_{str(i).zfill(2)}")

    eta = eta_T + eta_E + eta_geometry
    eta_form = dfx.fem.form(eta)
    eta_vec = assemble_vector(eta_form)
    eta_h = dfx.fem.Function(dg0_space)
    eta_h.x.petsc_vec.setArray(eta_vec.array[:])
    save_function(eta_h, f"residual_est_{str(i).zfill(2)}")

    residual_est = np.sqrt(eta_vec.array.sum())
    results["H10 estimator"].append(residual_est)
    if i>0:
        results["H10 estimator rate"].append((np.log(residual_est) - np.log(results["H10 estimator"][i-1]))/(np.log(results["dofs"][i]) - np.log(results["dofs"][i-1])))

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
        
        reference_degree = max(fe_degree, levelset_degree) + 2
        reference_element = element("Lagrange", reference_mesh.topology.cell_name(), max(fe_degree, levelset_degree) + 2)
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
                                                solution_space,
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

        results["H10 error"].append(error)
        if i>0:
            results["H10 error rate"].append((np.log(error) - np.log(results["H10 error"][i-1]))/(np.log(results["dofs"][i]) - np.log(results["dofs"][i-1])))


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