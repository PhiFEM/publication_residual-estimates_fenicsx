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

from utils import (
    fem_solve,
    marking,
    residual_estimation,
    save_function,
    write_log,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run FEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

args = parser.parse_args()
demo, parameters_name = args.parameters.split(sep="/")

source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + parameters_name)

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

from data import INITIAL_MESH_SIZE, MAX_EXTRA_STEP, MAXIMUM_DOF, REFERENCE, gen_mesh

exact_solution_available = False
try:
    from data import generate_exact_solution

    exact_solution_available = True
    exact_sol = generate_exact_solution(ufl)
except ImportError:
    pass

if not exact_solution_available:
    from data import generate_dirichlet_data, generate_source_term

    source_term = generate_source_term(np)
    dirichlet_data = generate_dirichlet_data(np)

with open(os.path.join(source_dir, parameters_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

initial_mesh_size = INITIAL_MESH_SIZE
max_dof = float(MAXIMUM_DOF)
fe_degree = parameters["finite_element_degree"]
dorfler_param = parameters["marking_parameter"]
refinement = parameters["refinement"]
curved = parameters["curved"]
reference = parameters_name == REFERENCE

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
    "eta_r": [],
    "eta_J": [],
}

num_dof = 0
i = 0
extra_step = 0
if reference:
    ref_criterion = extra_step < MAX_EXTRA_STEP
else:
    ref_criterion = True

dof_num_criterion = num_dof < MAXIMUM_DOF
stopping_criterion = dof_num_criterion and ref_criterion
while stopping_criterion:
    prefix = f"FEM | Iteration: {str(i).zfill(2)} | Test case: {demo} | Method: {parameters_name} | "
    results["iteration"].append(i)
    fe_space = dfx.fem.functionspace(mesh, fe_element)

    all_cells = dfx.mesh.locate_entities(
        mesh, tdim, lambda x: np.ones_like(x[0]).astype(bool)
    )
    mesh.topology.create_connectivity(tdim, tdim)
    active_dofs = dfx.fem.locate_dofs_topological(fe_space, tdim, all_cells)
    num_dof = len(active_dofs)
    results["dof"].append(num_dof)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)

    if exact_solution_available:
        x = ufl.SpatialCoordinate(mesh)
        exact_solution = exact_sol(x)
        fh = -ufl.div(ufl.grad(exact_solution))
        dirichlet_data = generate_exact_solution(np)
    else:
        fh = dfx.fem.Function(fe_space)
        fh.interpolate(source_term)
        dirichlet_data = generate_dirichlet_data(np)
    gh = dfx.fem.Function(fe_space)
    gh.interpolate(dirichlet_data)

    write_log(prefix + "FEM solve.")
    solution = fem_solve(fe_space, fh, gh)

    if not curved:
        save_function(
            solution, os.path.join(solutions_dir, f"solution_u_{str(i).zfill(2)}")
        )
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{str(i).zfill(2)}.bp")
    adios4dolfinx.write_mesh(checkpoint_file, mesh)
    adios4dolfinx.write_function(checkpoint_file, solution, name="solution_u")

    dx = ufl.Measure("dx", domain=mesh)
    dS = ufl.Measure("dS", domain=mesh)
    measures = {"dx": dx, "dS": dS}
    write_log(prefix + "Residual estimation.")
    eta_dict = residual_estimation(dg0_space, solution, fh, gh, measures, curved=curved)

    est_h = dfx.fem.Function(dg0_space)
    for name, eta in eta_dict.items():
        results[name].append(np.sqrt(eta.x.array.sum()))
        est_h.x.array[:] += eta.x.array[:]

    results["estimator"].append(np.sqrt(est_h.x.array.sum()))

    df = pl.DataFrame(results)
    header = f"======================================================================================================\n{prefix}\n======================================================================================================"
    print(header)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    # Marking and refinement
    if (refinement == "adap") and dof_num_criterion:
        write_log(prefix + "Marking.")
        facets_indices, cells_indices = marking(est_h, dorfler_param)
        write_log(prefix + "Refinement (adaptive).")
        mesh = geoModel.refineMarkedElements(tdim, cells_indices)[0]
        if curved:
            mesh = geoModel.curveField(3)
    else:
        write_log(prefix + "Refinement (uniform).")
        mesh = geoModel.refineMarkedElements(tdim, all_cells)[0]
        if curved:
            mesh = geoModel.curveField(3)
        if reference:
            extra_step += 1

    i += 1
