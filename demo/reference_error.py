import argparse
import os
import shutil
import sys

import adios4dolfinx
import dolfinx as dfx
import numpy as np
import polars as pl
import yaml
from basix.ufl import element
from mpi4py import MPI

sys.path.append("../")

from plots import plot_scalar
from utils import (
    cell_diameter,
    compute_dirichlet_oscillations,
    compute_h10_error,
    compute_l2_error_p,
    compute_source_term_oscillations,
    compute_xi_h10_l2,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="reference_error.py",
    description="Compute a higher-order approximation of the error from a reference solution.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

args = parser.parse_args()
demo, parameters_name = args.parameters.split(sep="/")

source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + parameters_name)

checkpoint_dir = os.path.join(output_dir, "checkpoints")

errors_dir = os.path.join(output_dir, "errors")
dirs = [
    output_dir,
    checkpoint_dir,
    errors_dir,
]
if os.path.isdir(errors_dir):
    shutil.rmtree(errors_dir)
    os.mkdir(errors_dir)
else:
    os.mkdir(errors_dir)

sys.path.append(source_dir)

from data import generate_dirichlet_data, generate_levelset, generate_source_term

with open(os.path.join(source_dir, parameters_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

iterations_num = parameters["iterations_number"]
fe_degree = parameters["finite_element_degree"]
auxiliary_degree = parameters["auxiliary_degree"]
levelset_degree = parameters["levelset_degree"]

with open(os.path.join(source_dir, "reference.yaml"), "rb") as f:
    ref_parameters = yaml.safe_load(f)

ref_max_iteration = ref_parameters["iterations_number"] - 1
ref_degree = ref_parameters["finite_element_degree"]

reference_mesh = adios4dolfinx.read_mesh(
    os.path.join(
        source_dir,
        "output_reference",
        "checkpoints",
        f"checkpoint_{str(ref_max_iteration).zfill(2)}.bp",
    ),
    comm=MPI.COMM_WORLD,
)

cdim = reference_mesh.topology.dim
num_reference_cells = reference_mesh.topology.index_map(cdim).size_global
reference_cells = np.arange(num_reference_cells)
cell_name = reference_mesh.topology.cell_name()

# Define finite elements and reference FE spaces
cg1_element = element("Lagrange", cell_name, fe_degree)
ref_element = element("Lagrange", cell_name, ref_degree)
levelset_element = element("Lagrange", cell_name, levelset_degree)
levelset_element = element("Lagrange", cell_name, levelset_degree)
dg0_element = element("DG", cell_name, 0)
dg1_element = element("DG", cell_name, auxiliary_degree)

reference_space = dfx.fem.functionspace(reference_mesh, ref_element)
reference_levelset_space = dfx.fem.functionspace(reference_mesh, levelset_element)
ref_dg0_space = dfx.fem.functionspace(reference_mesh, dg0_element)
ref_dg1_space = dfx.fem.functionspace(reference_mesh, dg1_element)

# Load reference source terms
source_term = generate_source_term(np)
ref_f = dfx.fem.Function(reference_space)
ref_f.interpolate(source_term)
dirichlet_data = generate_dirichlet_data(np)
ref_g = dfx.fem.Function(reference_space)
ref_g.interpolate(dirichlet_data)

levelset = generate_levelset(np)

# Load reference solution
reference_solution = dfx.fem.Function(reference_space)
adios4dolfinx.read_function(
    os.path.join(
        source_dir,
        "output_reference",
        "checkpoints",
        f"checkpoint_{str(ref_max_iteration).zfill(2)}.bp",
    ),
    reference_solution,
    name="solution_u",
)

# Allocate memory for results
results = pl.read_csv(os.path.join(output_dir, "results.csv")).to_dict()
results["error"] = [np.nan] * iterations_num
results["l2_p_error"] = [np.nan] * iterations_num
results["triple_norm_error"] = [np.nan] * iterations_num
results["source_term_osc"] = [np.nan] * iterations_num
results["dirichlet_data_osc"] = [np.nan] * iterations_num
results["total_error"] = [np.nan] * iterations_num
results["xi_h10"] = [np.nan] * iterations_num
results["xi_l2"] = [np.nan] * iterations_num

for i in range(iterations_num):
    # Load coarse mesh, create FE spaces and load functions
    mesh = adios4dolfinx.read_mesh(
        os.path.join(
            checkpoint_dir,
            f"checkpoint_{str(i).zfill(2)}.bp",
        ),
        comm=MPI.COMM_WORLD,
    )
    fe_space = dfx.fem.functionspace(mesh, cg1_element)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)
    dg1_space = dfx.fem.functionspace(mesh, dg1_element)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)
    solution_u = dfx.fem.Function(fe_space)
    solution_p = dfx.fem.Function(dg1_space)
    fe_levelset = dfx.fem.Function(levelset_space)

    adios4dolfinx.read_function(
        os.path.join(
            checkpoint_dir,
            f"checkpoint_{str(i).zfill(2)}.bp",
        ),
        solution_u,
        name="solution_u",
    )
    if "phifem" in parameters_name:
        adios4dolfinx.read_function(
            os.path.join(
                checkpoint_dir,
                f"checkpoint_{str(i).zfill(2)}.bp",
            ),
            solution_p,
            name="solution_p",
        )
        adios4dolfinx.read_function(
            os.path.join(
                checkpoint_dir,
                f"checkpoint_{str(i).zfill(2)}.bp",
            ),
            fe_levelset,
            name="levelset",
        )
        cells_tags = adios4dolfinx.read_meshtags(
            os.path.join(
                checkpoint_dir,
                f"checkpoint_{str(i).zfill(2)}.bp",
            ),
            mesh,
            meshtag_name="cells_tags",
        )

        cut_indicator = dfx.fem.Function(dg0_space)
        cut_indicator.x.array[cells_tags.find(2)] = 1.0

    # Compute NMM interpolation data between coarse spaces and ref spaces
    nmm_fe2ref = dfx.fem.create_interpolation_data(
        reference_space, fe_space, reference_cells, padding=1.0e-14
    )
    nmm_dg12ref_dg1 = dfx.fem.create_interpolation_data(
        ref_dg1_space, dg1_space, reference_cells, padding=1.0e-14
    )
    nmm_dg02ref_dg0 = dfx.fem.create_interpolation_data(
        ref_dg0_space, dg0_space, reference_cells, padding=1.0e-14
    )
    nmm_dg02ref_dg1 = dfx.fem.create_interpolation_data(
        ref_dg1_space, dg0_space, reference_cells, padding=1.0e-14
    )
    nmm_levelset2ref_levelset = dfx.fem.create_interpolation_data(
        reference_levelset_space, levelset_space, reference_cells, padding=1.0e-14
    )

    # Interpolate coarse functions to ref spaces
    solution_u_2_ref = dfx.fem.Function(reference_space)
    solution_u_2_ref.interpolate_nonmatching(solution_u, reference_cells, nmm_fe2ref)

    dg0_cut_indicator_2_ref = dfx.fem.Function(ref_dg0_space)
    dg0_cut_indicator_2_ref.interpolate_nonmatching(
        cut_indicator, reference_cells, nmm_dg02ref_dg0
    )
    dg0_cut_indicator_2_ref = dfx.fem.Function(ref_dg0_space)
    dg0_cut_indicator_2_ref.interpolate_nonmatching(
        cut_indicator, reference_cells, nmm_dg02ref_dg0
    )

    dg1_solution_p_2_ref_dg1 = dfx.fem.Function(ref_dg1_space)
    dg1_solution_p_2_ref_dg1.interpolate_nonmatching(
        solution_p, reference_cells, nmm_dg12ref_dg1
    )

    coarse_levelset_2_ref = dfx.fem.Function(reference_levelset_space)
    coarse_levelset_2_ref.interpolate_nonmatching(
        fe_levelset, reference_cells, nmm_levelset2ref_levelset
    )

    coarse_mesh_h_T = cell_diameter(dg0_space)
    dg1_coarse_h_T_2_ref = dfx.fem.Function(ref_dg1_space)
    dg1_coarse_h_T_2_ref.interpolate_nonmatching(
        coarse_mesh_h_T, reference_cells, nmm_dg02ref_dg1
    )

    # Compute reference H10 error
    ref_h10_norm, coarse_h10_norm = compute_h10_error(
        solution_u, reference_solution, ref_dg0_space, dg0_space
    )

    if coarse_h10_norm is not None:
        plot_scalar(
            coarse_h10_norm, os.path.join(errors_dir, f"h10_error_{str(i).zfill(2)}")
        )

    h10_err_sqd = ref_h10_norm.x.array.sum()
    results["error"][i] = np.sqrt(h10_err_sqd)

    # Compute L2 error for p
    h_t_reference_p = reference_solution - ref_g

    plot_scalar(
        dg1_solution_p_2_ref_dg1,
        os.path.join(errors_dir, f"ref_solution_p_{str(i).zfill(2)}"),
    )
    plot_scalar(
        coarse_levelset_2_ref,
        os.path.join(errors_dir, f"ref_levelset_{str(i).zfill(2)}"),
    )
    plot_scalar(
        dg0_coarse_h_T_2_ref,
        os.path.join(errors_dir, f"ref_coarse_mesh_h_T_{str(i).zfill(2)}"),
    )

    ref_l2_p_err = compute_l2_error_p(
        dg1_solution_p_2_ref,
        dg1_ref_solution,
        dg1_ref_g,
        dg1_levelset_2_ref,
        dg1_coarse_h_T_2_ref,
        dg1_cut_indicator_2_ref,
        ref_dg0_space,
    )

    plot_scalar(ref_l2_p_err, os.path.join(errors_dir, f"l2_p_err_{str(i).zfill(2)}"))
    l2_p_err_sqd = ref_l2_p_err.x.array.sum()

    results["l2_p_error"][i] = np.sqrt(l2_p_err_sqd)

    # Source term oscillations estimation
    fh = dfx.fem.Function(fe_space)
    fh.interpolate(source_term)

    ref_osc_f, coarse_osc_f = compute_source_term_oscillations(
        fh, ref_f, dg0_coarse_h_T_2_ref, ref_dg0_space, dg0_space
    )

    plot_scalar(
        coarse_osc_f, os.path.join(errors_dir, f"source_term_osc_{str(i).zfill(2)}")
    )
    results["source_term_osc"][i] = np.sqrt(ref_osc_f.x.array.sum())

    # Dirichlet data oscillations estimation
    gh = dfx.fem.Function(fe_space)
    gh.interpolate(dirichlet_data)

    ref_osc_g, coarse_osc_g = compute_dirichlet_oscillations(
        gh, ref_g, ref_dg0_space, dg0_space, dg0_cut_indicator_2_ref
    )

    results["dirichlet_data_osc"][i] = np.sqrt(ref_osc_g.x.array.sum())

    xi_ref_h10, xi_ref_l2 = compute_xi_h10_l2(solution_u_2_ref, ref_g, ref_dg0_space)

    results["xi_h10"][i] = np.sqrt(xi_ref_h10.x.array.sum())
    results["xi_l2"][i] = np.sqrt(xi_ref_l2.x.array.sum())

    results["triple_norm_error"][i] = np.sqrt(h10_err_sqd + l2_p_err_sqd)
    results["total_error"][i] = np.sqrt(
        h10_err_sqd
        + l2_p_err_sqd
        + ref_osc_f.x.array.sum()
        + ref_osc_g.x.array.sum()
        + xi_ref_h10.x.array.sum()
        + xi_ref_l2.x.array.sum()
    )

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))
