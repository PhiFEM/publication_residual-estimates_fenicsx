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
from plots import plot_mesh, plot_scalar, plot_tags
from utils import (
    compute_boundary_local_estimators,
    marking,
    phifem_direct_solve,
    phifem_dual_solve,
    residual_estimation,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run phiFEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")
parser.add_argument("--plot", action="store_true", help="Plot cells and facets tags.")

args = parser.parse_args()
demo, parameters = args.parameters.split(sep="/")
plot = args.plot


source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + parameters)

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
    if not os.path.isdir(dir_path):
        print(f"{dir_path} directory not found, we create it.")
        os.mkdir(dir_path)

sys.path.append(source_dir)

from data import generate_dirichlet_data, generate_levelset, generate_source_term

try:
    from data import generate_detection_levelset
except ImportError:
    print(
        "Didn't find detection_levelset for the specified demo, using levelset instead."
    )
    generate_detection_levelset = generate_levelset

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
dirichlet_estimator = parameters["dirichlet_estimator"]

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
    "estimator": [],
    "eta_T": [],
    "eta_E": [],
    "eta_p": [],
    "eta_1_z": [],
    "eta_0_z": [],
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
    plot_mesh(
        mesh,
        os.path.join(meshes_dir, f"mesh_{str(i).zfill(2)}"),
        wireframe=True,
        linewidth=1.5,
    )
    if plot:
        leg_dict = {1: "inside", 2: "cut", 3: "outside"}
        plot_levelset = generate_detection_levelset(np)
        fig = plt.figure()
        ax = fig.subplots()
        levelset_kwargs = {"colors": "k", "linewidths": 2.0, "linestyles": "--"}
        plot_mesh_tags(
            mesh,
            cells_tags,
            ax,
            expression_levelset=plot_levelset,
            leg_dict=leg_dict,
            levelset_kwargs=levelset_kwargs,
            display_scalarbar=False,
        )
        plt.savefig(
            os.path.join(tags_dir, f"cells_tags_{str(i).zfill(2)}.png"),
            bbox_inches="tight",
            dpi=500,
        )
        plot_tags(
            mesh,
            cells_tags,
            os.path.join(tags_dir, f"cells_tags_{str(i).zfill(2)}"),
            annotations=leg_dict,
            line_width=5.0,
        )
        leg_dict = {1: "inside", 2: "cut", 3: "inside boundary", 4: "outside boundary"}
        fig = plt.figure()
        ax = fig.subplots()
        plot_mesh_tags(
            mesh,
            facets_tags,
            ax,
            expression_levelset=plot_levelset,
            leg_dict=leg_dict,
            levelset_kwargs=levelset_kwargs,
        )
        plt.savefig(
            os.path.join(tags_dir, f"facets_tags_{str(i).zfill(2)}.png"),
            bbox_inches="tight",
            dpi=500,
        )
        plot_tags(
            mesh,
            facets_tags,
            os.path.join(tags_dir, f"facets_tags_{str(i).zfill(2)}"),
            line_width=8.0,
            annotations=leg_dict,
        )
    results["iteration"].append(i)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

    measures = {"dx": dx, "dS": dS, "ds": ds}

    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    fe_element = element("Lagrange", cell_name, fe_degree)
    if dual:
        aux_element = element("DG", cell_name, auxiliary_degree)
        mxd_element = mixed_element([fe_element, aux_element])
        fe_space = dfx.fem.functionspace(mesh, mxd_element)
        aux_space = dfx.fem.functionspace(mesh, aux_element)
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

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{str(i).zfill(2)}.bp")
    adios4dolfinx.write_mesh(checkpoint_file, mesh)

    levelset = generate_levelset(np)
    phih = dfx.fem.Function(levelset_space)

    if dual:
        phih.interpolate(levelset, cells_tags.find(2))
        phih_plot = dfx.fem.Function(levelset_space)
        phih_plot.interpolate(levelset)
    else:
        phih.interpolate(levelset)
        phih_plot = phih

    adios4dolfinx.write_function(checkpoint_file, phih_plot, name="levelset")
    plot_scalar(phih_plot, os.path.join(levelsets_dir, f"levelset_{str(i).zfill(2)}"))
    plot_scalar(
        phih_plot,
        os.path.join(levelsets_dir, f"levelset_{str(i).zfill(2)}"),
        warp_by_scalar=True,
    )

    u_space = dfx.fem.functionspace(mesh, fe_element)
    source_term = generate_source_term(np)
    dirichlet_data = generate_dirichlet_data(np)
    fh = dfx.fem.Function(u_space)
    fh.interpolate(source_term)
    gh = dfx.fem.Function(u_space)
    gh.interpolate(dirichlet_data)
    plot_scalar(fh, os.path.join(data_dir, f"source_term_{str(i).zfill(2)}"))
    plot_scalar(gh, os.path.join(data_dir, f"dirichlet_data_{str(i).zfill(2)}"))
    plot_scalar(
        fh,
        os.path.join(data_dir, f"source_term_{str(i).zfill(2)}"),
        warp_by_scalar=True,
    )
    plot_scalar(
        gh,
        os.path.join(data_dir, f"dirichlet_data_{str(i).zfill(2)}"),
        warp_by_scalar=True,
    )

    if dual:
        solution_u, solution_p = phifem_dual_solve(
            fe_space, fh, gh, phih, measures, coefs
        )
    else:
        spaces = {"primal": fe_space, "solution": solution_space}
        solution_u, solution_w = phifem_direct_solve(
            spaces, fh, gh, phih, measures, coefs
        )

    plot_scalar(
        solution_u, os.path.join(solutions_dir, f"solution_u_{str(i).zfill(2)}")
    )
    plot_scalar(
        solution_u,
        os.path.join(solutions_dir, f"solution_u_{str(i).zfill(2)}"),
        warp_by_scalar=True,
    )

    plot_scalar(
        solution_p, os.path.join(solutions_dir, f"solution_p_{str(i).zfill(2)}")
    )
    plot_scalar(
        solution_p,
        os.path.join(solutions_dir, f"solution_p_{str(i).zfill(2)}"),
        warp_by_scalar=True,
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
    if dual:
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
            results["eta_p"].append(np.nan)
    else:
        eta_dict = residual_estimation(
            dg0_space,
            solution_u,
            fh,
            gh,
            {"dx": dx((1, 2)), "dS": dS((1, 2))},
            coefs=coefs,
            phih=phih,
        )
        results["eta_p"].append(np.nan)

    if boundary_correction:
        if dual:
            solution = solution_p
        else:
            solution = solution_w

        eta_dict["eta_1_z"], eta_dict["eta_0_z"], parent_cells_tags, fine_mesh = (
            compute_boundary_local_estimators(
                mesh, solution, levelset, phih, cells_tags, facets_tags, dual=dual
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
            plot_tags(
                fine_mesh,
                parent_cells_tags,
                os.path.join(tags_dir, f"parent_cells_tags_{str(i).zfill(2)}"),
                line_width=5.0,
                annotations=leg_dict,
            )
    else:
        results["eta_1_z"].append(np.nan)
        results["eta_0_z"].append(np.nan)

    est_h = dfx.fem.Function(dg0_space)
    for name, eta in eta_dict.items():
        plot_scalar(eta, os.path.join(etas_dir, name + "_" + str(i).zfill(2)))

        est_h.x.array[:] += eta.x.array[:]
        results[name].append(np.sqrt(eta.x.array.sum()))

    results["estimator"].append(np.sqrt(est_h.x.array.sum()))

    plot_scalar(est_h, os.path.join(estimators_dir, f"estimator_{str(i).zfill(2)}"))

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    if i < iterations_num:
        # Marking and refinement
        if refinement == "adap":
            facets_indices = marking(est_h, dorfler_param)[0]
            mesh = dfx.mesh.refine(mesh, facets_indices)[0]
        elif refinement == "unif":
            mesh = dfx.mesh.refine(mesh)[0]
