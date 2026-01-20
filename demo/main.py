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
from meshtagsplot import plot_mesh_tags
from mpi4py import MPI
from phifem.mesh_scripts import compute_tags_measures
from utils import (
    compute_boundary_local_estimators,
    marking,
    phifem_dual_solve,
    residual_estimation,
    save_function,
    write_log,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run phiFEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")
parser.add_argument("--plot", action="store_true", help="Plot cells and facets tags.")

args = parser.parse_args()
demo, parameters_name = args.parameters.split(sep="/")
plot = args.plot


source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + parameters_name)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

checkpoint_dir = os.path.join(output_dir, "checkpoints")

if os.path.isdir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

data_dir = os.path.join(output_dir, "data")
tags_dir = os.path.join(output_dir, "tags")
solutions_dir = os.path.join(output_dir, "solutions")
levelsets_dir = os.path.join(output_dir, "levelsets")
etas_dir = os.path.join(output_dir, "etas")
estimators_dir = os.path.join(output_dir, "estimators")
reference_errors_dir = os.path.join(output_dir, "reference_errors")
meshes_dir = os.path.join(output_dir, "meshes")
dirs = [
    output_dir,
    tags_dir,
    data_dir,
    checkpoint_dir,
    solutions_dir,
    levelsets_dir,
    etas_dir,
    estimators_dir,
    reference_errors_dir,
    meshes_dir,
]
for dir_path in dirs:
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)


sys.path.append(source_dir)

from data import generate_levelset

exact_solution_available = False
try:
    from data import generate_exact_solution

    exact_solution_available = True
except ImportError:
    pass

if not exact_solution_available:
    from data import generate_dirichlet_data, generate_source_term

try:
    from data import generate_detection_levelset
except ImportError:
    print(
        "Didn't find detection_levelset for the specified demo, using levelset instead."
    )
    generate_detection_levelset = generate_levelset

with open(os.path.join(source_dir, parameters_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

initial_mesh_size = parameters["initial_mesh_size"]
max_dof = float(parameters["maximum_dof"])
fe_degree = parameters["finite_element_degree"]
levelset_degree = parameters["levelset_degree"]
solution_degree = parameters["solution_degree"]
detection_degree = parameters["boundary_detection_degree"]
pen_coef = parameters["penalization_coefficient"]
stab_coef = parameters["stabilization_coefficient"]
coefs = {"penalization": pen_coef, "stabilization": stab_coef}
dorfler_param = parameters["marking_parameter"]
bbox = parameters["bbox"]
refinement = parameters["refinement"]
boundary_correction = parameters["boundary_correction"]
auxiliary_degree = parameters["auxiliary_degree"]
discretize_levelset = parameters["discretize_levelset"]
dirichlet_estimator = parameters["dirichlet_estimator"]
single_layer = parameters["single_layer"]

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
    "estimator": [],
    "eta_r": [],
    "eta_J": [],
    "eta_BC": [],
    "eta_geo": [],
}

num_dof = 0
i = 0
while num_dof < max_dof:
    prefix = f"PHIFEM | Iteration: {str(i).zfill(2)} | Test case: {demo} | Method: {parameters_name} | "
    cell_name = mesh.topology.cell_name()
    levelset_element = element("Lagrange", cell_name, levelset_degree)
    levelset_bg_space = dfx.fem.functionspace(mesh, levelset_element)
    if discretize_levelset:
        detection_levelset = dfx.fem.Function(levelset_bg_space)
        detection_levelset.interpolate(generate_detection_levelset(np))
    else:
        detection_levelset = generate_detection_levelset(ufl)
    write_log(prefix + "Computation of mesh tags")
    cells_tags, facets_tags, _, ds, _, _ = compute_tags_measures(
        mesh,
        detection_levelset,
        detection_degree,
        box_mode=True,
        single_layer_cut=single_layer,
    )

    results["iteration"].append(i)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

    measures = {"dx": dx, "dS": dS, "ds": ds}

    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    fe_element = element("Lagrange", cell_name, fe_degree)
    aux_element = element("Lagrange", cell_name, auxiliary_degree)
    mxd_element = mixed_element([fe_element, aux_element])
    mixed_space = dfx.fem.functionspace(mesh, mxd_element)
    fe_space = dfx.fem.functionspace(mesh, fe_element)
    aux_space = dfx.fem.functionspace(mesh, aux_element)

    dg0_element = element("DG", cell_name, 0)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)

    cells_tags_dg0 = dfx.fem.Function(dg0_space)
    cells_tags_dg0.x.array[:] = cells_tags.values

    save_function(
        cells_tags_dg0,
        os.path.join(output_dir, tags_dir, f"cells_tags_{str(i).zfill(2)}"),
    )

    mesh_edges = dfx.mesh.locate_entities(
        mesh, 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    wireframe = dfx.mesh.create_submesh(mesh, 1, mesh_edges)[0]
    wf_cell_name = wireframe.topology.cell_name()
    wf_dg0_element = element("DG", wf_cell_name, 0)
    wf_dg0_space = dfx.fem.functionspace(wireframe, wf_dg0_element)
    facets_tags_dg0 = dfx.fem.Function(wf_dg0_space)
    facets_tags_dg0.x.array[:] = facets_tags.values

    save_function(
        facets_tags_dg0,
        os.path.join(output_dir, tags_dir, f"facets_tags_{str(i).zfill(2)}"),
    )

    levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    omega_h_cells = np.union1d(cells_tags.find(1), cells_tags.find(2))
    cdim = mesh.topology.dim
    mesh.topology.create_connectivity(cdim, cdim)
    fe_active_dofs = dfx.fem.locate_dofs_topological(fe_space, cdim, omega_h_cells)
    aux_active_dofs = dfx.fem.locate_dofs_topological(
        aux_space, cdim, cells_tags.find(2)
    )

    num_dof = len(fe_active_dofs) + len(aux_active_dofs)
    results["dof"].append(num_dof)

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{str(i).zfill(2)}.bp")
    adios4dolfinx.write_mesh(checkpoint_file, mesh)

    levelset = generate_levelset(np)
    phih = dfx.fem.Function(levelset_space)

    phih.interpolate(levelset, cells_tags.find(2))
    phih_plot = dfx.fem.Function(levelset_space)
    phih_plot.interpolate(levelset)

    adios4dolfinx.write_function(checkpoint_file, phih_plot, name="levelset")
    save_function(phih_plot, os.path.join(levelsets_dir, f"levelset_{str(i).zfill(2)}"))
    save_function(
        phih_plot,
        os.path.join(levelsets_dir, f"levelset_{str(i).zfill(2)}"),
    )

    u_space = dfx.fem.functionspace(mesh, fe_element)

    if not exact_solution_available:
        source_term = generate_source_term(np)
        dirichlet_data = generate_dirichlet_data(np)
        fh = dfx.fem.Function(u_space)
        fh.interpolate(source_term)
        gh = dfx.fem.Function(u_space)
        gh.interpolate(dirichlet_data)
        save_function(fh, os.path.join(data_dir, f"source_term_{str(i).zfill(2)}"))
        save_function(gh, os.path.join(data_dir, f"dirichlet_data_{str(i).zfill(2)}"))
        save_function(
            fh,
            os.path.join(data_dir, f"source_term_{str(i).zfill(2)}"),
        )
        save_function(
            gh,
            os.path.join(data_dir, f"dirichlet_data_{str(i).zfill(2)}"),
        )
    else:
        x = ufl.SpatialCoordinate(mesh)
        exact_solution = generate_exact_solution(ufl)(x)
        fh = -ufl.div(ufl.grad(exact_solution))
        gh = exact_solution

    write_log(prefix + "phiFEM solve.")
    solution_u, solution_p = phifem_dual_solve(
        mixed_space, fh, gh, phih, measures, coefs
    )

    save_function(
        solution_u, os.path.join(solutions_dir, f"solution_u_{str(i).zfill(2)}")
    )

    save_function(
        solution_p, os.path.join(solutions_dir, f"solution_p_{str(i).zfill(2)}")
    )

    adios4dolfinx.write_meshtags(
        checkpoint_file, mesh, cells_tags, meshtag_name="cells_tags"
    )
    adios4dolfinx.write_meshtags(
        checkpoint_file, mesh, facets_tags, meshtag_name="facets_tags"
    )
    adios4dolfinx.write_function(checkpoint_file, solution_u, name="solution_u")
    adios4dolfinx.write_function(checkpoint_file, solution_p, name="solution_p")

    # Residual error estimation
    write_log(prefix + "Residual estimation.")
    eta_dict = residual_estimation(
        dg0_space,
        solution_u,
        fh,
        gh,
        {"dx": dx((1, 2)), "dS": dS((1, 2))},
        coefs=coefs,
        phih=phih,
        solution_p=solution_p,
        dirichlet_estimator=dirichlet_estimator,
    )
    if not dirichlet_estimator:
        results["eta_BC"].append(np.nan)

    if boundary_correction:
        solution = solution_p

        write_log(prefix + "Computation boundary estimator.")
        eta_dict["eta_geo"], parent_cells_tags, fine_mesh = (
            compute_boundary_local_estimators(
                mesh, solution, levelset, phih, cells_tags, dual=True
            )
        )
        if plot:
            leg_dict = {1: "inside", 2: "cut", 3: "outside"}
            levelset_kwargs = {"colors": "k", "linewidths": 2.0, "linestyles": "--"}
            fig = plt.figure()
            ax = fig.subplots()
            plot_mesh_tags(
                fine_mesh,
                parent_cells_tags,
                ax,
                expression_levelset=plot_levelset,
                leg_dict=leg_dict,
                levelset_kwargs=levelset_kwargs,
                display_scalarbar=False,
            )
            plt.savefig(
                os.path.join(tags_dir, f"parent_cells_tags_{str(i).zfill(2)}.png"),
                bbox_inches="tight",
                dpi=500,
            )
    else:
        results["eta_geo"].append(np.nan)

    est_h = dfx.fem.Function(dg0_space)
    for name, eta in eta_dict.items():
        save_function(eta, os.path.join(etas_dir, name + "_" + str(i).zfill(2)))

        est_h.x.array[:] += eta.x.array[:]
        results[name].append(np.sqrt(eta.x.array.sum()))

    results["estimator"].append(np.sqrt(est_h.x.array.sum()))

    save_function(est_h, os.path.join(estimators_dir, f"estimator_{str(i).zfill(2)}"))

    df = pl.DataFrame(results)
    header = f"======================================================================================================\n{prefix}\n======================================================================================================"
    print(header)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    if num_dof < max_dof:
        # Marking and refinement
        if refinement == "adap":
            write_log(prefix + "Marking.")
            facets_indices = marking(est_h, dorfler_param)[0]
            write_log(prefix + "Refinement.")
            mesh = dfx.mesh.refine(mesh, facets_indices)[0]
        elif refinement == "unif":
            write_log(prefix + "Refinement.")
            mesh = dfx.mesh.refine(mesh)[0]

    i += 1
