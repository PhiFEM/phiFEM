import argparse
from basix.ufl import element, mixed_element
import numpy as np
import dolfinx as dfx
from dolfinx.fem import assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
import os
import petsc4py.PETSc as PETSc
import polars as pl
import ufl
import yaml

from data import E_in, E_out, epsilon, sigma_in, sigma_out, levelset, exact_solution, cos_vec
from phifem.mesh_scripts import compute_tags_measures

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run phiFEM on a multimaterial elasticity test case.",
)

parser.add_argument(
    "parameters", type=str, help="Name of parameters file (without yaml extension)."
)

args = parser.parse_args()
parameters = args.parameters

parameters_path = os.path.join(parent_dir, parameters + ".yaml")
output_dir = os.path.join(parent_dir, parameters + "_output")

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
        cg1_element = element(
            element_family,
            mesh.topology.cell_name(),
            1,
            shape=fct.function_space.value_shape,
        )
        cg1_space = dfx.fem.functionspace(mesh, cg1_element)
        cg1_fct = dfx.fem.Function(cg1_space)
        cg1_fct.interpolate(fct)
        with XDMFFile(
            mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w"
        ) as of:
            of.write_mesh(mesh)
            of.write_function(cg1_fct)
    else:
        with XDMFFile(
            mesh.comm, os.path.join(output_dir, "functions", file_name + ".xdmf"), "w"
        ) as of:
            of.write_mesh(mesh)
            of.write_function(fct)


with open(parameters_path, "rb") as f:
    parameters = yaml.safe_load(f)

# Extract parameters
initial_mesh_size = parameters["initial_mesh_size"]
num_iterations = parameters["num_iterations"]
primal_degree = parameters["primal_degree"]
flux_degree = parameters["flux_degree"]
auxiliary_degree = parameters["auxiliary_degree"]
levelset_degree = parameters["levelset_degree"]
detection_degree = parameters["boundary_detection_degree"]
bbox = parameters["bbox"]
penalization_coefficient = parameters["penalization_coefficient"]
stabilization_coefficient = parameters["stabilization_coefficient"]
cell_type = parameters["cell_type"]

# Create the background mesh
nx = int(np.abs(bbox[0][1] - bbox[0][0]) / initial_mesh_size)
ny = int(np.abs(bbox[1][1] - bbox[1][0]) / initial_mesh_size)
# Quads cells
if cell_type == "triangle":
    cell_type = dfx.cpp.mesh.CellType.triangle
elif cell_type == "quadrilateral":
    cell_type = dfx.cpp.mesh.CellType.quadrilateral
else:
    raise ValueError(
        "Parameter error cell_type can only be 'triangle' or 'quadrilateral'."
    )
mesh = dfx.mesh.create_rectangle(
    MPI.COMM_WORLD, np.asarray(bbox).T, [nx, ny], cell_type
)

results = {"dof": [],
           "H10 relative error": [],
           "L2 relative error": []}
for i in range(num_iterations):
    cells_tags, facets_tags, _, ds_from_inside, ds_from_outside, _ = compute_tags_measures(
        mesh, levelset, detection_degree, box_mode=True
    )

    gdim = mesh.geometry.dim
    cell_name = mesh.topology.cell_name()
    primal_element = element("Lagrange", cell_name, primal_degree, shape=(gdim,))
    flux_element = element("Lagrange", cell_name, flux_degree, shape=(gdim, gdim))
    auxiliary_element = element("Lagrange", cell_name, auxiliary_degree, shape=(gdim,))

    mixd_element = mixed_element(
        [primal_element, primal_element, flux_element, flux_element, auxiliary_element]
    )

    dg0_element = element("DG", cell_name, 0)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)

    levelset_element = element("Lagrange", cell_name, levelset_degree)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    primal_space = dfx.fem.functionspace(mesh, primal_element)
    results["dof"].append(primal_space.dofmap.index_map.size_global)
    mixed_space = dfx.fem.functionspace(mesh, mixd_element)

    exact_solution_h = dfx.fem.Function(primal_space)
    exact_solution_h.interpolate(exact_solution)

    if i == num_iterations - 1:
        save_function(exact_solution_h, "exact_solution")

    x_mesh = ufl.SpatialCoordinate(mesh)
    cos_vec_ufl = cos_vec(ufl)
    f = -ufl.div(sigma_in(cos_vec_ufl(x_mesh))) / E_in

    phi_h = dfx.fem.Function(levelset_space)
    phi_h.interpolate(levelset)

    u_in, u_out, y_in, y_out, p = ufl.TrialFunctions(mixed_space)
    v_in, v_out, z_in, z_out, q = ufl.TestFunctions(mixed_space)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)


    def boundary_dbc(x):
        left = np.isclose(x[0], -1.5).astype(bool)
        bottom = np.isclose(x[1], -1.5).astype(bool)
        right = np.isclose(x[0], 1.5).astype(bool)
        top = np.isclose(x[1], 1.5).astype(bool)
        return left + bottom + right + top


    boundary_dbc_facets = dfx.mesh.locate_entities_boundary(mesh, gdim - 1, boundary_dbc)

    # Create a FE function from outer space
    u_dbc_in = dfx.fem.Function(primal_space)
    u_dbc_in.interpolate(exact_solution)

    bc_in_dofs = dfx.fem.locate_dofs_topological(
        (mixed_space.sub(0), primal_space), gdim - 1, boundary_dbc_facets
    )
    dbc_in = dfx.fem.dirichletbc(u_dbc_in, bc_in_dofs, mixed_space.sub(0))
    bcs = [dbc_in]

    n = ufl.FacetNormal(mesh)
    h_T = ufl.CellDiameter(mesh)

    boundary_in = ufl.inner(ufl.dot(y_in, n), v_in)
    boundary_out = ufl.inner(ufl.dot(y_out, n), v_out)

    stiffness_in = ufl.inner(sigma_in(u_in), epsilon(v_in))
    stiffness_out = ufl.inner(sigma_out(u_out), epsilon(v_out))

    coef_in = (E_in / (E_in + E_out))**2
    coef_out = (E_out / (E_in + E_out))**2 
    penalization = penalization_coefficient * (
        ufl.inner(y_in + sigma_in(u_in), z_in + sigma_in(v_in)) * coef_out
        + ufl.inner(y_out + sigma_out(u_out), z_out + sigma_out(v_out)) * coef_in
        + h_T ** (-2)
        * ufl.inner(
            ufl.dot(y_in, ufl.grad(phi_h)) - ufl.dot(y_out, ufl.grad(phi_h)),
            ufl.dot(z_in, ufl.grad(phi_h)) - ufl.dot(z_out, ufl.grad(phi_h)),
        )
        + h_T ** (-2)
        * ufl.inner(
            u_in - u_out + h_T ** (-1) * p * phi_h, v_in - v_out + h_T ** (-1) * q * phi_h
        )
    )

    stabilization_facets_in = (
        stabilization_coefficient
        * ufl.avg(h_T)
        * ufl.inner(ufl.jump(sigma_in(u_in), n), ufl.jump(sigma_in(v_in), n))
    )

    stabilization_cells_in = stabilization_coefficient * ufl.inner(
        ufl.div(y_in), ufl.div(z_in)
    )

    stabilization_cells_out = stabilization_coefficient * ufl.inner(
        ufl.div(y_out), ufl.div(z_out)
    )

    stabilization_facets_out = (
        stabilization_coefficient
        * ufl.avg(h_T)
        * ufl.inner(ufl.jump(sigma_out(u_out), n), ufl.jump(sigma_out(v_out), n))
    )

    a = (
        stiffness_in * dx((1, 2))
        + stiffness_out * dx((2, 3))
        + penalization * dx(2)
        + stabilization_facets_in * dS(3)
        + stabilization_facets_out * dS(4)
        + stabilization_cells_in * dx(2)
        + stabilization_cells_out * dx(2)
    )

    boundary_in_int = boundary_in * ds_from_inside
    boundary_out_int = boundary_out * ds_from_outside

    bilinear_form = dfx.fem.form(a)
    A = assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()

    bdy_in_form = dfx.fem.form(boundary_in_int)
    A_bdy_in = assemble_matrix(bdy_in_form, bcs=bcs)
    A_bdy_in.assemble()

    bdy_out_form = dfx.fem.form(boundary_out_int)
    A_bdy_out = assemble_matrix(bdy_out_form, bcs=bcs)
    A_bdy_out.assemble()

    A.axpy(1.0, A_bdy_in, False)
    A.axpy(1.0, A_bdy_out, False)

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setType("preonly")
    solver = ksp.create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A)

    # Configure MUMPS to handle nullspace
    pc = solver.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    stabilization_rhs_in = stabilization_coefficient * (ufl.inner(f, ufl.div(z_in)))
    stabilization_rhs_out = stabilization_coefficient * (ufl.inner(f, ufl.div(z_out)))
    rhs_in = ufl.inner(f, v_in)
    rhs_out = ufl.inner(f, v_out)

    L = (
        rhs_in * dx((1, 2))
        + rhs_out * dx((2, 3))
        + stabilization_rhs_in * dx(2)
        + stabilization_rhs_out * dx(2)
    )

    linear_form = dfx.fem.form(L)
    b = assemble_vector(linear_form)

    # Apply the dirichlet bc to the RHS vector
    dfx.fem.petsc.apply_lifting(b, [bilinear_form], bcs=[bcs])
    for bc in bcs:
        bc.set(b.array_w)

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

    solution_uh_in, solution_uh_out, _, _, _ = solution_wh.split()
    save_function(solution_uh_in.collapse(), f"solution_in_{str(i).zfill(2)}")
    save_function(solution_uh_out.collapse(), f"solution_out_{str(i).zfill(2)}")

    # Combine the in and out solutions
    solution_h = dfx.fem.Function(mixed_space)
    solution_uh, _, _, _, _ = solution_h.split()
    solution_uh = solution_uh.collapse()

    mesh.topology.create_connectivity(gdim, gdim)
    dofs_to_remove_in = dfx.fem.locate_dofs_topological(
        mixed_space.sub(0), gdim, cells_tags.find(3)
    )
    dofs_cut_in = dfx.fem.locate_dofs_topological(
        mixed_space.sub(0), gdim, cells_tags.find(2)
    )
    dofs_to_remove_in = np.setdiff1d(dofs_to_remove_in, dofs_cut_in)

    dofs_to_remove_out = dfx.fem.locate_dofs_topological(
        mixed_space.sub(1), gdim, cells_tags.find(1)
    )
    dofs_cut_out = dfx.fem.locate_dofs_topological(
        mixed_space.sub(1), gdim, cells_tags.find(2)
    )
    dofs_to_remove_out = np.setdiff1d(dofs_to_remove_out, dofs_cut_out)

    solution_uh_out.x.array[dofs_cut_out] = solution_uh_out.x.array[dofs_cut_out] / 2.0
    solution_uh_in.x.array[dofs_cut_in] = solution_uh_in.x.array[dofs_cut_in] / 2.0
    solution_uh_out.x.array[dofs_to_remove_out] = 0.0
    solution_uh_in.x.array[dofs_to_remove_in] = 0.0
    solution_uh_out = solution_uh_out.collapse()
    solution_uh_in = solution_uh_in.collapse()
    solution_uh.x.array[:] = solution_uh_in.x.array[:] + solution_uh_out.x.array[:]

    save_function(solution_uh, f"solution_{str(i).zfill(2)}")
    save_function(phi_h, f"levelset_{str(i).zfill(2)}")

    # Discretization error computation

    reference_element = element("Lagrange", cell_name, primal_degree + 2, shape=(gdim,))
    reference_space = dfx.fem.functionspace(mesh, reference_element)

    reference_exact_solution = dfx.fem.Function(reference_space)
    reference_exact_solution.interpolate(exact_solution)
    reference_solution_uh = dfx.fem.Function(reference_space)
    reference_solution_uh.interpolate(solution_uh)

    reference_error = reference_exact_solution - reference_solution_uh

    # H10 error
    h10_norm_exact_solution = (
        ufl.inner(ufl.grad(reference_exact_solution), ufl.grad(reference_exact_solution))
        * dx
    )
    h10_norm_exact_solution = assemble_scalar(dfx.fem.form(h10_norm_exact_solution))

    h10_norm = ufl.inner(ufl.grad(reference_error), ufl.grad(reference_error))

    v0 = ufl.TrialFunction(dg0_space)
    h10_local_fct = dfx.fem.Function(dg0_space)

    h10_local = ufl.inner(h10_norm, v0) * dx
    h10_local_form = dfx.fem.form(h10_local)
    h10_local_vec = assemble_vector(h10_local_form)
    h10_local_fct.x.array[:] = h10_local_vec.array[:]

    save_function(h10_local_fct, f"h10_local_error_{str(i).zfill(2)}")

    h10_global_err = np.sqrt(np.sum(h10_local_vec.array[:]) / h10_norm_exact_solution)
    results["H10 relative error"].append(h10_global_err)

    # L2 error
    l2_norm_exact_solution = (
        ufl.inner(reference_exact_solution, reference_exact_solution) * dx
    )
    l2_norm_exact_solution = assemble_scalar(dfx.fem.form(l2_norm_exact_solution))

    l2_norm = ufl.inner(reference_error, reference_error)

    v0 = ufl.TrialFunction(dg0_space)
    l2_local_fct = dfx.fem.Function(dg0_space)

    l2_local = ufl.inner(l2_norm, v0) * dx
    l2_local_form = dfx.fem.form(l2_local)
    l2_local_vec = assemble_vector(l2_local_form)
    l2_local_fct.x.array[:] = l2_local_vec.array[:]

    save_function(l2_local_fct, f"l2_local_error_{str(i).zfill(2)}")

    l2_global_err = np.sqrt(np.sum(l2_local_vec.array[:]) / l2_norm_exact_solution)
    results["L2 relative error"].append(l2_global_err)

    df = pl.DataFrame(results)
    df.write_csv(os.path.join(output_dir, "results.csv"))
    print(df)

    if i<num_iterations-1:
        mesh = dfx.mesh.refine(mesh)[0]

h10_slope, _ = np.polyfit(np.log(results["dof"][:]), np.log(results["H10 relative error"][:]), 1)
l2_slope, _ = np.polyfit(np.log(results["dof"][:]), np.log(results["L2 relative error"][:]), 1)

print("H10 relative error slope:", h10_slope)
print("L2 relative error slope:", l2_slope)