import os
import time

import dolfinx as dfx
import matplotlib.pyplot as plt
import numpy as np
import petsc4py.PETSc as PETSc
import polars as pl
import ufl
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from tests_data.data_test_time_strong_dirichlet import (
    generate_levelset,
    source_term,
)

from phifem.mesh_scripts import compute_tags_measures

stab_coef = 1.0
bbox = [[-3.0, -3.0], [3.0, 3.0]]


def _compute_tags_measures(
    mesh,
    mesh_type,
    detection_levelset,
    detection_degree,
):
    if mesh_type == "bg":
        cells_tags, facets_tags, _, ds, _, _ = compute_tags_measures(
            mesh, detection_levelset, detection_degree, box_mode=True
        )
    elif mesh_type == "sub":
        cells_tags, facets_tags, mesh, _, _, _ = compute_tags_measures(
            mesh, detection_levelset, detection_degree, box_mode=False
        )
        ds = ufl.Measure("ds", domain=mesh)

    return cells_tags, facets_tags, ds, mesh


def _assemble_phifem_system(primal_space, phi_h, f_h, cells_tags, facets_tags, ds):
    mesh = primal_space.mesh
    w = ufl.TrialFunction(primal_space)
    phiw = phi_h * w
    v = ufl.TestFunction(primal_space)
    phiv = phi_h * v

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    a = (
        ufl.inner(ufl.grad(phiw), ufl.grad(phiv)) * dx((1, 2))
        - ufl.inner(ufl.inner(ufl.grad(phiw), n), phiv) * ds
        + (
            stab_coef
            * h_T**2
            * ufl.inner(ufl.div(ufl.grad(phiw)), ufl.div(ufl.grad(phiv)))
            * dx(2)
        )
        + (
            stab_coef
            * ufl.avg(h_T)
            * ufl.inner(ufl.jump(ufl.grad(phiw), n), ufl.jump(ufl.grad(phiv), n))
            * dS((2, 3))
        )
    )

    bilinear_form = dfx.fem.form(a)
    A = assemble_matrix(bilinear_form)
    A.assemble()

    # Linear form
    L = ufl.inner(f_h, phiv) * dx((1, 2)) - stab_coef * h_T**2 * ufl.inner(
        f_h, ufl.div(ufl.grad(phiv))
    ) * dx(2)

    linear_form = dfx.fem.form(L)
    b = assemble_vector(linear_form)

    return A, b


def _set_up_petsc_solver(mesh, A, mesh_type):
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")

    # When solving on the background mesh, we need mumps to handle the null space of the matrix
    if mesh_type == "bg":
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    return ksp


def _solve(ksp, b, solution_vector):
    ksp.solve(b, solution_vector)
    ksp.destroy()


def _multiply_solution_levelset(solution_space, solution_wh, phi_h):
    solution_uh = dfx.fem.Function(solution_space)
    solution_wh_s_space = dfx.fem.Function(solution_space)
    solution_wh_s_space.interpolate(solution_wh)
    phi_h_s_space = dfx.fem.Function(solution_space)
    phi_h_s_space.interpolate(phi_h)

    solution_uh.x.array[:] = solution_wh_s_space.x.array[:] * phi_h_s_space.x.array[:]


def _run_strong_dirichlet(
    mesh,
    primal_degree,
    levelset_degree,
    solution_degree,
    detection_degree,
    mesh_type,
    discretize_levelset,
    times,
):
    cell_name = mesh.topology.cell_name()

    levelset_element = element("Lagrange", cell_name, levelset_degree)
    bg_levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    start = time.time()
    if discretize_levelset:
        detection_levelset = generate_levelset(np)
        detection_levelset_h = dfx.fem.Function(bg_levelset_space)
        detection_levelset_h.interpolate(detection_levelset)
    else:
        x_ufl = ufl.SpatialCoordinate(mesh)
        detection_levelset = generate_levelset(ufl)
        detection_levelset_h = detection_levelset(x_ufl)
    end = time.time()
    times["interpolation_levelset_detection"].append(end - start)

    start = time.time()
    cells_tags, facets_tags, ds, mesh = _compute_tags_measures(
        mesh,
        mesh_type,
        detection_levelset_h,
        detection_degree,
    )
    end = time.time()
    times["compute_tags_measures"].append(end - start)

    primal_element = element("Lagrange", cell_name, primal_degree)
    solution_element = element("Lagrange", cell_name, solution_degree)

    primal_space = dfx.fem.functionspace(mesh, primal_element)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)
    solution_space = dfx.fem.functionspace(mesh, solution_element)

    # Interpolation of the levelset
    start = time.time()
    if not discretize_levelset:
        levelset = generate_levelset(np)
    else:
        levelset = detection_levelset
    phi_h = dfx.fem.Function(levelset_space)
    phi_h.interpolate(levelset)
    end = time.time()
    times["interpolation_levelset"].append(end - start)

    # Interpolation of the source term f
    start = time.time()
    f_h = dfx.fem.Function(primal_space)
    f_h.interpolate(source_term)
    end = time.time()
    times["interpolation_source"].append(end - start)

    start = time.time()
    A, b = _assemble_phifem_system(
        primal_space, phi_h, f_h, cells_tags, facets_tags, ds
    )
    end = time.time()
    times["assemble_phifem_system"].append(end - start)

    start = time.time()
    ksp = _set_up_petsc_solver(mesh, A, mesh_type)
    end = time.time()
    times["set_up_petsc_solver"].append(end - start)

    start = time.time()
    solution_wh = dfx.fem.Function(primal_space)
    _solve(ksp, b, solution_wh.x.petsc_vec)
    end = time.time()
    times["solve"].append(end - start)

    start = time.time()
    _multiply_solution_levelset(solution_space, solution_wh, phi_h)
    end = time.time()
    times["multiply_solution_levelset"].append(end - start)

    return times


def test_strong_dirichlet(
    primal_degree,
    levelset_degree,
    solution_degree,
    detection_degree,
    mesh_type,
    discretize_levelset,
):
    name_data = "strong_dirichlet_" + "_".join(
        [
            str(primal_degree),
            str(levelset_degree),
            str(solution_degree),
            str(detection_degree),
            mesh_type,
        ]
    )
    if discretize_levelset:
        name_data += "_discretize"

    mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox, [200, 200])
    times = {
        "interpolation_levelset_detection": [],
        "compute_tags_measures": [],
        "interpolation_levelset": [],
        "interpolation_source": [],
        "assemble_phifem_system": [],
        "set_up_petsc_solver": [],
        "solve": [],
        "multiply_solution_levelset": [],
        "total": [],
    }

    for i in range(100):
        print(i)
        times = _run_strong_dirichlet(
            mesh,
            primal_degree,
            levelset_degree,
            solution_degree,
            detection_degree,
            mesh_type,
            discretize_levelset,
            times,
        )

    avg_times = {"measure": ["Wallclock time (s)", "Percentage"]}

    avg_times["total"] = [0.0]
    avg_times_vals = []
    avg_times_labels = []
    for key, val in times.items():
        if key not in ["measure", "total"]:
            avg_times["total"][0] += np.mean(val)
            avg_times[key] = [np.mean(val)]
            avg_times_vals.append(np.mean(val))
            avg_times_labels.append(key)

    for key, val in avg_times.items():
        if key != "measure":
            avg_times[key].append(100.0 * val[0] / avg_times["total"][0])

    df = pl.DataFrame(avg_times)
    print(df)
    df.write_csv(os.path.join("tests_data", name_data + ".csv"))

    fig = plt.figure()
    plt.title(name_data)
    ax = fig.subplots()
    ax.pie(avg_times_vals, labels=avg_times_labels)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.axis("off")
    plt.savefig(
        os.path.join("tests_data", name_data + ".png"), dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    test_strong_dirichlet(1, 5, 6, 5, "sub", False)
