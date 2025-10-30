import argparse
import os
import shutil
import sys

import adios4dolfinx
import dolfinx as dfx
import numpy as np
import polars as pl
import ufl
import yaml
from basix.ufl import element

sys.path.append("../")

from plots import plot_mesh, plot_scalar
from utils import (
    fem_solve,
    marking,
    residual_estimation,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run FEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

args = parser.parse_args()
demo, parameters = args.parameters.split(sep="/")

ref = parameters == "reference"

source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + parameters)

checkpoint_dir = os.path.join(output_dir, "checkpoints")

if os.path.isdir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

solutions_dir = os.path.join(output_dir, "solutions")
etas_dir = os.path.join(output_dir, "etas")
estimators_dir = os.path.join(output_dir, "estimators")
errors_dir = os.path.join(output_dir, "errors")
meshes_dir = os.path.join(output_dir, "meshes")
dirs = [
    output_dir,
    checkpoint_dir,
    solutions_dir,
    etas_dir,
    estimators_dir,
    errors_dir,
    meshes_dir,
]
for dir_path in dirs:
    if not os.path.isdir(dir_path):
        print(f"{dir_path} directory not found, we create it.")
        os.mkdir(dir_path)

sys.path.append(source_dir)

from data import gen_mesh, generate_dirichlet_data, generate_source_term

with open(os.path.join(source_dir, parameters + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

initial_mesh_size = parameters["initial_mesh_size"]
iterations_num = parameters["iterations_number"]
fe_degree = parameters["finite_element_degree"]
dorfler_param = parameters["marking_parameter"]
refinement = parameters["refinement"]
curved = parameters["curved"]


mesh, geoModel = gen_mesh(initial_mesh_size, curved=curved)

tdim = mesh.topology.dim
fdim = tdim - 1
cell_name = mesh.topology.cell_name()
fe_element = element("CG", cell_name, fe_degree)
dg0_element = element("DG", cell_name, 0)


results = {
    "iteration": [],
    "dof": [],
    "estimator": [],
    "eta_T": [],
    "eta_E": [],
}

if ref:
    iterations_num += 2
for i in range(iterations_num):
    plot_mesh(
        mesh,
        os.path.join(meshes_dir, f"mesh_{str(i).zfill(2)}"),
        wireframe=True,
        linewidth=1.5,
    )
    results["iteration"].append(i)
    fe_space = dfx.fem.functionspace(mesh, fe_element)

    all_cells = dfx.mesh.locate_entities(
        mesh, tdim, lambda x: np.ones_like(x[0]).astype(bool)
    )
    mesh.topology.create_connectivity(tdim, tdim)
    active_dofs = dfx.fem.locate_dofs_topological(fe_space, tdim, all_cells)
    results["dof"].append(len(active_dofs))
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)

    """
    source_term = generate_source_term(ufl)
    dirichlet_data = generate_dirichlet_data(ufl)
    x = ufl.SpatialCoordinate(mesh)
    fh = source_term(x) - ufl.div(ufl.grad(dirichlet_data(x)))
    # fh = dfx.fem.Function(u_space)
    # fh.interpolate(source_term)
    gh = dfx.fem.Function(fe_space)
    """
    source_term = generate_source_term(np)
    dirichlet_data = generate_dirichlet_data(np)
    fh = dfx.fem.Function(fe_space)
    fh.interpolate(source_term)
    gh = dfx.fem.Function(fe_space)
    gh.interpolate(dirichlet_data)

    solution = fem_solve(fe_space, fh, gh)

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{str(i).zfill(2)}.bp")
    adios4dolfinx.write_mesh(checkpoint_file, mesh)
    adios4dolfinx.write_function(checkpoint_file, solution, name="solution_u")

    plot_scalar(solution, os.path.join(solutions_dir, f"solution_{str(i).zfill(2)}"))
    plot_scalar(
        solution,
        os.path.join(solutions_dir, f"solution_{str(i).zfill(2)}"),
        warp_by_scalar=True,
    )

    dx = ufl.Measure("dx", domain=mesh)
    dS = ufl.Measure("dS", domain=mesh)
    measures = {"dx": dx, "dS": dS}
    eta_dict = residual_estimation(dg0_space, solution, fh, gh, measures, curved=curved)

    est_h = dfx.fem.Function(dg0_space)
    for name, eta in eta_dict.items():
        plot_scalar(eta, os.path.join(etas_dir, name + f"_{str(i).zfill(2)}"))

        results[name].append(np.sqrt(eta.x.array.sum()))
        est_h.x.array[:] += eta.x.array[:]

    results["estimator"].append(np.sqrt(est_h.x.array.sum()))

    plot_scalar(
        est_h,
        os.path.join(estimators_dir, f"estimator_{str(i).zfill(2)}"),
    )

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    if ref:
        last_iteration = iterations_num - 2
    else:
        last_iteration = iterations_num

    if i < last_iteration:
        # Marking and refinement
        if refinement == "adap":
            facets_indices, cells_indices = marking(est_h, dorfler_param)
            mesh = geoModel.refineMarkedElements(tdim, cells_indices)[0]
            if curved:
                mesh = geoModel.curveField(3)
        elif refinement == "unif":
            mesh = geoModel.refineMarkedElements(tdim, all_cells)[0]
            if curved:
                mesh = geoModel.curveField(3)
    else:
        if ref:
            mesh = geoModel.refineMarkedElements(tdim, all_cells)[0]
            if curved:
                mesh = geoModel.curveField(3)
