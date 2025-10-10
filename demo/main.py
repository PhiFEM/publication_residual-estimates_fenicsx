import argparse
import os
import shutil
import sys

import adios4dolfinx
import dolfinx as dfx
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ufl
import yaml
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import assemble_vector
from meshtagsplot import plot_mesh_tags
from mpi4py import MPI
from phifem.mesh_scripts import compute_tags_measures
from utils import (
    compute_boundary_local_estimators,
    marking,
    phifem_direct_solve,
    phifem_dual_solve,
    residual_estimation,
    save_function,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run phiFEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

parser.add_argument("--tags", default=False, action="store_true")

args = parser.parse_args()
demo, parameters = args.parameters.split(sep="/")
plot_tags = args.tags


source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_phifem_" + parameters)
checkpoint_dir = os.path.join(output_dir, "checkpoints")

if os.path.isdir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

for dir_path in [output_dir, checkpoint_dir]:
    if not os.path.isdir(dir_path):
        print(f"{dir_path} directory not found, we create it.")
        os.mkdir(dir_path)

sys.path.append(source_dir)

from data import generate_levelset, source_term

try:
    from data import generate_detection_levelset
except ImportError:
    print(
        "Didn't find detection_levelset for the specified demo, using levelset instead."
    )
    generate_detection_levelset = generate_levelset

exact_sol_available = True
try:
    from data import exact_solution
except ImportError:
    exact_sol_available = False

with open(os.path.join(source_dir, parameters + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

initial_mesh_size = parameters["initial_mesh_size"]
iterations_num = parameters["iterations_number"]
fe_degree = parameters["finite_element_degree"]
levelset_degree = parameters["levelset_degree"]
solution_degree = parameters["solution_degree"]
detection_degree = parameters["boundary_detection_degree"]
pen_coef = parameters["penalization_coefficient"]
stab_coef = parameters["stabilization_coefficient"]
coefs = {"penalization": pen_coef, "stabilization": stab_coef}
dorfler_param = parameters["marking_parameter"]
bbox = parameters["bbox"]
box_mode = parameters["mesh_type"] == "bg"
refinement = parameters["refinement"]
boundary_correction = parameters["boundary_correction"]
auxiliary_degree = parameters["auxiliary_degree"]
discretize_levelset = parameters["discretize_levelset"]

dual = auxiliary_degree > -1

# Create background mesh
nx = int(np.abs(bbox[0][1] - bbox[0][0]) / initial_mesh_size / np.sqrt(2.0))
ny = int(np.abs(bbox[1][1] - bbox[1][0]) / initial_mesh_size / np.sqrt(2.0))

cell_type = dfx.cpp.mesh.CellType.triangle
mesh = dfx.mesh.create_rectangle(
    MPI.COMM_WORLD, np.asarray(bbox).T, [nx, ny], cell_type
)

results = {
    "iteration": [],
    "dof": [],
    "H10_estimator": [],
    "eta_T": [],
    "eta_E": [],
    "eta_p": [],
    "eta_eps": [],
}


for i in range(iterations_num):
    cell_name = mesh.topology.cell_name()
    levelset_element = element("Lagrange", cell_name, levelset_degree)
    levelset_bg_space = dfx.fem.functionspace(mesh, levelset_element)
    if discretize_levelset:
        detection_levelset = dfx.fem.Function(levelset_bg_space)
        detection_levelset.interpolate(generate_detection_levelset(np))
    else:
        x_ufl = ufl.SpatialCoordinate(mesh)
        detection_levelset = generate_detection_levelset(ufl)(x_ufl)
    if box_mode:
        cells_tags, facets_tags, _, ds, _, _ = compute_tags_measures(
            mesh, detection_levelset, detection_degree, box_mode=box_mode
        )
    else:
        cells_tags, facets_tags, mesh, _, _, _ = compute_tags_measures(
            mesh, detection_levelset, detection_degree, box_mode=box_mode
        )
        ds = ufl.Measure("ds", domain=mesh)

    if plot_tags:
        fig = plt.figure()
        ax = fig.subplots()
        leg_dict = {1: "inside", 2: "cut"}
        plot_mesh_tags(
            mesh,
            cells_tags,
            ax,
            expression_levelset=generate_detection_levelset(np),
            leg_dict=leg_dict,
        )
        plt.savefig(
            os.path.join(output_dir, f"cells_tags_{str(i).zfill(2)}.png"),
            dpi=500,
            bbox_inches="tight",
        )
        fig = plt.figure()
        ax = fig.subplots()
        leg_dict = {1: "inside", 2: "cut", 3: "inside boundary", 4: "outside boundary"}
        plot_mesh_tags(
            mesh,
            facets_tags,
            ax,
            expression_levelset=generate_detection_levelset(np),
            leg_dict=leg_dict,
        )
        plt.savefig(
            os.path.join(output_dir, f"facets_tags_{str(i).zfill(2)}.png"),
            dpi=500,
            bbox_inches="tight",
        )
    results["iteration"].append(i)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

    measures = {"dx": dx, "dS": dS, "ds": ds}

    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    fe_element = element("Lagrange", cell_name, fe_degree)
    if dual:
        aux_element = element("Lagrange", cell_name, auxiliary_degree)
        mxd_element = mixed_element([fe_element, aux_element])
        fe_space = dfx.fem.functionspace(mesh, mxd_element)
    else:
        fe_space = dfx.fem.functionspace(mesh, fe_element)
        solution_element = element("Lagrange", cell_name, solution_degree)
        solution_space = dfx.fem.functionspace(mesh, solution_element)

    dg0_element = element("DG", cell_name, 0)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)

    levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    omega_h_cells = np.union1d(cells_tags.find(1), cells_tags.find(2))
    cdim = mesh.topology.dim
    mesh.topology.create_connectivity(cdim, cdim)
    active_dofs = dfx.fem.locate_dofs_topological(fe_space, cdim, omega_h_cells)

    results["dof"].append(len(active_dofs))

    levelset = generate_levelset(np)
    phih = dfx.fem.Function(levelset_space)

    if dual:
        phih.interpolate(levelset, cells_tags.find(2))
    else:
        phih.interpolate(levelset)

    save_function(phih, os.path.join(output_dir, f"levelset_{str(i).zfill(2)}.xdmf"))

    if exact_sol_available:
        x_ufl = ufl.SpatialCoordinate(mesh)
        fh = source_term(x_ufl)
    else:
        u_space = dfx.fem.functionspace(mesh, fe_element)
        fh = dfx.fem.Function(u_space)
        fh.interpolate(source_term)
        save_function(
            fh, os.path.join(output_dir, f"source_term_{str(i).zfill(2)}.xdmf")
        )

    if dual:
        solution_u, solution_p = phifem_dual_solve(fe_space, fh, phih, measures, coefs)
    else:
        spaces = {"primal": fe_space, "solution": solution_space}
        solution_u, solution_w = phifem_direct_solve(spaces, fh, phih, measures, coefs)

    save_function(
        solution_u, os.path.join(output_dir, f"solution_u_{str(i).zfill(2)}.xdmf")
    )

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{str(i).zfill(2)}.bp")
    adios4dolfinx.write_mesh(checkpoint_file, mesh)
    adios4dolfinx.write_function(checkpoint_file, solution_u, name="solution")

    # Residual error estimation
    if dual:
        eta_dict = residual_estimation(
            dg0_space, solution_u, fh, phih, measures, coefs, solution_p=solution_p
        )
    else:
        eta_dict = residual_estimation(dg0_space, solution_u, fh, phih, measures, coefs)
        results["eta_p"].append(np.nan)

    if boundary_correction:
        if dual:
            solution = solution_p
        else:
            solution = solution_w

        eta_dict["eta_eps"] = compute_boundary_local_estimators(
            mesh, solution, levelset, phih, cells_tags, facets_tags, dual=dual
        )
    else:
        results["eta_eps"].append(np.nan)

    est_h = dfx.fem.Function(dg0_space)
    for name, eta in eta_dict.items():
        save_function(
            eta, os.path.join(output_dir, name + "_" + str(i).zfill(2) + ".xdmf")
        )

        results[name].append(np.sqrt(eta.x.array.sum()))
        est_h.x.array[:] += eta.x.array[:]

    results["H10_estimator"].append(np.sqrt(est_h.x.array.sum()))

    save_function(
        est_h, os.path.join(output_dir, f"H10_estimator_{str(i).zfill(2)}.xdmf")
    )

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    if i < iterations_num:
        # Marking and refinement
        if refinement == "adap":
            facets_indices = marking(est_h, dorfler_param)
            mesh = dfx.mesh.refine(mesh, facets_indices)[0]
        elif refinement == "unif":
            mesh = dfx.mesh.refine(mesh)[0]

if not exact_sol_available:
    fdim = mesh.topology.dim - 1
    for k in range(2):
        mesh.topology.create_entities(fdim)
        mesh = dfx.mesh.refine(mesh)[0]
        x_ufl = ufl.SpatialCoordinate(mesh)
        detection_levelset = generate_detection_levelset(ufl)(x_ufl)
        if box_mode:
            cells_tags, facets_tags, _, ds, _, _ = compute_tags_measures(
                mesh, detection_levelset, detection_degree, box_mode=box_mode
            )
        else:
            cells_tags, facets_tags, mesh, _, _, _ = compute_tags_measures(
                mesh, detection_levelset, detection_degree, box_mode=box_mode
            )
            ds = ufl.Measure("ds", domain=mesh)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

    measures = {"dx": dx, "dS": dS, "ds": ds}

results["Reference_error"] = np.empty_like(results["dof"][:]).astype(np.float64)
results["Reference_error"][:] = np.nan

reference_fe_space = dfx.fem.functionspace(mesh, fe_element)
reference_levelset_space = dfx.fem.functionspace(mesh, levelset_element)
reference_dg0_space = dfx.fem.functionspace(mesh, dg0_element)
reference_fh = dfx.fem.Function(reference_fe_space)
reference_fh.interpolate(source_term)
reference_phih = dfx.fem.Function(reference_levelset_space)

if dual:
    reference_phih.interpolate(levelset, cells_tags.find(2))
else:
    reference_phih.interpolate(levelset)

if exact_sol_available:
    reference_u = dfx.fem.Function(reference_fe_space)
    reference_u.interpolate(exact_solution)
else:
    if dual:
        reference_mixed_space = dfx.fem.functionspace(mesh, mxd_element)
        reference_u = phifem_dual_solve(
            reference_mixed_space, reference_fh, reference_phih, measures, coefs
        )[0]
    else:
        reference_solution_space = dfx.fem.functionspace(mesh, solution_element)
        reference_spaces = {
            "primal": reference_fe_space,
            "solution": reference_solution_space,
        }
        reference_u = phifem_direct_solve(
            reference_spaces, reference_fh, reference_phih, measures, coefs
        )[0]

save_function(reference_u, os.path.join(output_dir, "reference_u.xdmf"))

reference_space = reference_u.function_space
for j in range(i):
    coarse_mesh = adios4dolfinx.read_mesh(
        os.path.join(checkpoint_dir, f"checkpoint_{str(j).zfill(2)}.bp"),
        comm=MPI.COMM_WORLD,
    )

    if dual:
        coarse_space = dfx.fem.functionspace(coarse_mesh, fe_element)
    else:
        coarse_space = dfx.fem.functionspace(coarse_mesh, solution_element)

    coarse_solution = dfx.fem.Function(coarse_space)
    adios4dolfinx.read_function(
        os.path.join(checkpoint_dir, f"checkpoint_{str(j).zfill(2)}.bp"),
        coarse_solution,
        name="solution",
    )

    num_reference_cells = mesh.topology.index_map(cdim).size_global
    reference_cells = np.arange(num_reference_cells)
    nmm_coarse_space2ref_space = dfx.fem.create_interpolation_data(
        reference_space, coarse_space, reference_cells, padding=1.0e-14
    )

    reference_coarse_u = dfx.fem.Function(reference_space)
    reference_coarse_u.interpolate_nonmatching(
        coarse_solution, reference_cells, nmm_coarse_space2ref_space
    )

    save_function(
        reference_coarse_u,
        os.path.join(output_dir, f"reference_coarse_u_{str(j).zfill(2)}.xdmf"),
    )

    diff = dfx.fem.Function(reference_space)
    diff.x.array[:] = reference_u.x.array[:] - reference_coarse_u.x.array[:]

    save_function(
        diff, os.path.join(output_dir, f"reference_diff_{str(j).zfill(2)}.xdmf")
    )
    x_ufl = ufl.SpatialCoordinate(mesh)
    detection_levelset = generate_detection_levelset(ufl)(x_ufl)
    cells_tags = compute_tags_measures(
        mesh, detection_levelset, detection_degree, box_mode=box_mode
    )[0]

    reference_dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    fine_v0 = ufl.TestFunction(reference_dg0_space)
    grad_diff = ufl.grad(diff)
    h10_norm_diff = ufl.inner(grad_diff, grad_diff) * fine_v0 * reference_dx((1, 2))
    h10_norm_form = dfx.fem.form(h10_norm_diff)
    h10_norm_vec = assemble_vector(h10_norm_form)

    h10_norm_h = dfx.fem.Function(reference_dg0_space)
    h10_norm_h.x.array[:] = h10_norm_vec.array[:]

    save_function(
        h10_norm_h, os.path.join(output_dir, f"reference_error_{str(j).zfill(2)}.xdmf")
    )

    results["Reference_error"][j] = np.sqrt(h10_norm_vec.array.sum())

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))
