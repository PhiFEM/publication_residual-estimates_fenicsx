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
from dolfinx.io import XDMFFile
from mpi4py import MPI

sys.path.append("../")

from utils import (
    cell_diameter,
    compute_h10_error,
    compute_l2_error,
    compute_phi_error,
    compute_phi_p_error,
    write_log,
)

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="reference_error.py",
    description="Compute a higher-order approximation of the error from a reference solution.",
)

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")
parser.add_argument(
    "--compute_xi_ref",
    action="store_true",
    help="Choose if an higher order approximation to xi is computed.",
)

args = parser.parse_args()
demo, parameters_name = args.parameters.split(sep="/")
compute_xi_ref = args.compute_xi_ref

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

from data import generate_levelset

exact_solution_available = False
try:
    from data import generate_exact_solution

    exact_solution_available = True
except ImportError:
    pass

if not exact_solution_available:
    from data import generate_dirichlet_data, generate_source_term

with open(os.path.join(source_dir, parameters_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

nums = []
for f in os.listdir(
    os.path.join(source_dir, "output_" + parameters_name, "checkpoints")
):
    if f.endswith(".bp"):
        num = f[:-3].split(sep="_")[-1]
        nums.append(int(num))
iterations_num = sorted(nums)[-1] + 1

fe_degree = parameters["finite_element_degree"]
phifem = "phifem" in parameters_name
if phifem:
    levelset_degree = parameters["levelset_degree"]

with open(os.path.join(source_dir, "reference.yaml"), "rb") as f:
    ref_parameters = yaml.safe_load(f)

ref_degree = ref_parameters["finite_element_degree"]

nums = []
for f in os.listdir(os.path.join(source_dir, "output_reference", "checkpoints")):
    if f.endswith(".bp"):
        num = f[:-3].split(sep="_")[-1]
        nums.append(int(num))
ref_max_iteration = sorted(nums)[-1]

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
ref_dg1_element = element("DG", cell_name, ref_degree)
if phifem:
    levelset_element = element("Lagrange", cell_name, levelset_degree)
    reference_levelset_space = dfx.fem.functionspace(reference_mesh, levelset_element)
dg0_element = element("DG", cell_name, 0)
dg1_element = element("DG", cell_name, 0)

reference_space = dfx.fem.functionspace(reference_mesh, ref_element)
ref_dg0_space = dfx.fem.functionspace(reference_mesh, dg0_element)
ref_dg1_space = dfx.fem.functionspace(reference_mesh, ref_dg1_element)


# Load reference source terms
if not exact_solution_available:
    write_log("Read reference solution.")
    source_term = generate_source_term(np)
    ref_f = dfx.fem.Function(reference_space)
    ref_f.interpolate(source_term)
    dirichlet_data = generate_dirichlet_data(np)

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
else:
    write_log("Interpolate analytical solution.")
    ref_solution = generate_exact_solution(ufl)
    ref_x = ufl.SpatialCoordinate(reference_mesh)
    reference_solution = ref_solution(ref_x)
    ref_f = -ufl.div(ufl.grad(reference_solution))
    dirichlet_data = generate_exact_solution(np)

ref_g = dfx.fem.Function(reference_space)
ref_g.interpolate(dirichlet_data)

levelset = generate_levelset(np)
reference_levelset = dfx.fem.Function(reference_levelset_space)
reference_levelset.interpolate(levelset)

# Allocate memory for results
results = pl.read_csv(os.path.join(output_dir, "results.csv")).to_dict()
results["h10_error"] = [np.nan] * iterations_num
results["l2_p_error"] = [np.nan] * iterations_num
results["triple_norm_error"] = [np.nan] * iterations_num
results["source_term_osc"] = [np.nan] * iterations_num
results["dirichlet_data_osc"] = [np.nan] * iterations_num
results["total_error"] = [np.nan] * iterations_num
results["xi_h10"] = [np.nan] * iterations_num
results["xi_l2"] = [np.nan] * iterations_num

for i in range(iterations_num):
    prefix = f"REFERENCE ERROR | Iteration: {str(i).zfill(2)} | Test case: {demo} | Method: {parameters_name} | "
    write_log(prefix + "Load mesh.")
    # Load coarse mesh, create FE spaces and load functions
    mesh = adios4dolfinx.read_mesh(
        os.path.join(
            checkpoint_dir,
            f"checkpoint_{str(i).zfill(2)}.bp",
        ),
        comm=MPI.COMM_WORLD,
    )
    fe_space = dfx.fem.functionspace(mesh, cg1_element)
    aux_space = dfx.fem.functionspace(mesh, cg1_element)

    if phifem:
        levelset_space = dfx.fem.functionspace(mesh, levelset_element)
        fe_levelset = dfx.fem.Function(levelset_space)
    dg1_space = dfx.fem.functionspace(mesh, dg1_element)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)
    solution_u = dfx.fem.Function(fe_space)
    solution_p = dfx.fem.Function(aux_space)

    write_log(prefix + "Load solution_u.")
    adios4dolfinx.read_function(
        os.path.join(
            checkpoint_dir,
            f"checkpoint_{str(i).zfill(2)}.bp",
        ),
        solution_u,
        name="solution_u",
    )
    nmm_fe2ref = dfx.fem.create_interpolation_data(
        reference_space, fe_space, reference_cells, padding=1.0e-14
    )
    if phifem:
        write_log(prefix + "Load solution_p.")
        adios4dolfinx.read_function(
            os.path.join(
                checkpoint_dir,
                f"checkpoint_{str(i).zfill(2)}.bp",
            ),
            solution_p,
            name="solution_p",
        )
        write_log(prefix + "Load levelset.")
        adios4dolfinx.read_function(
            os.path.join(
                checkpoint_dir,
                f"checkpoint_{str(i).zfill(2)}.bp",
            ),
            fe_levelset,
            name="levelset",
        )
        write_log(prefix + "Load cells_tags.")
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

        nmm_levelset2ref_levelset = dfx.fem.create_interpolation_data(
            reference_levelset_space, levelset_space, reference_cells, padding=1.0e-14
        )
        # Compute NMM interpolation data between coarse spaces and ref spaces
        nmm_dg12ref_dg1 = dfx.fem.create_interpolation_data(
            ref_dg1_space, dg1_space, reference_cells, padding=1.0e-14
        )
        nmm_dg02ref_dg0 = dfx.fem.create_interpolation_data(
            ref_dg0_space, dg0_space, reference_cells, padding=1.0e-14
        )
        nmm_dg02ref_dg1 = dfx.fem.create_interpolation_data(
            ref_dg1_space, dg0_space, reference_cells, padding=1.0e-14
        )
        dg0_cut_indicator_2_ref = dfx.fem.Function(ref_dg0_space)
        dg0_cut_indicator_2_ref.interpolate_nonmatching(
            cut_indicator, reference_cells, nmm_dg02ref_dg0
        )
        dg0_cut_indicator_2_ref = dfx.fem.Function(ref_dg0_space)
        dg0_cut_indicator_2_ref.interpolate_nonmatching(
            cut_indicator, reference_cells, nmm_dg02ref_dg0
        )

        solution_p_2_ref = dfx.fem.Function(reference_space)
        solution_p_2_ref.interpolate_nonmatching(
            solution_p, reference_cells, nmm_dg12ref_dg1
        )

        coarse_levelset_2_ref = dfx.fem.Function(reference_levelset_space)
        coarse_levelset_2_ref.interpolate_nonmatching(
            fe_levelset, reference_cells, nmm_levelset2ref_levelset
        )

        coarse_mesh_h_T = cell_diameter(dg0_space)
        dg0_coarse_h_T_2_ref = dfx.fem.Function(ref_dg0_space)
        dg0_coarse_h_T_2_ref.interpolate_nonmatching(
            coarse_mesh_h_T, reference_cells, nmm_dg02ref_dg0
        )

    # Interpolate coarse functions to ref spaces
    solution_u_2_ref = dfx.fem.Function(reference_space)
    solution_u_2_ref.interpolate_nonmatching(solution_u, reference_cells, nmm_fe2ref)

    triple_norm_err_sqd = 0.0

    # Compute reference H10 error
    write_log(prefix + "Compute H10 error.")
    ref_h10_norm, coarse_h10_norm = compute_h10_error(
        solution_u_2_ref, reference_solution, ref_dg0_space, dg0_space
    )

    with XDMFFile(
        reference_mesh.comm, os.path.join(errors_dir, "ref_h10_error.xdmf"), "w"
    ) as of:
        of.write_mesh(reference_mesh)
        of.write_function(ref_h10_norm)

    h10_err_sqd = ref_h10_norm.x.array.sum()
    results["h10_error"][i] = np.sqrt(h10_err_sqd)

    triple_norm_err_sqd += h10_err_sqd

    if phifem:
        write_log(prefix + "Compute L2 error.")
        ref_l2_norm, coarse_l2_norm = compute_l2_error(
            solution_u_2_ref,
            reference_solution,
            ref_dg0_space,
            dg0_space,
            dg0_coarse_h_T_2_ref,
            dg0_cut_indicator_2_ref,
        )

        with XDMFFile(
            reference_mesh.comm, os.path.join(errors_dir, "ref_l2_error.xdmf"), "w"
        ) as of:
            of.write_mesh(reference_mesh)
            of.write_function(ref_l2_norm)

        l2_err_sqd = ref_l2_norm.x.array.sum()
        triple_norm_err_sqd += l2_err_sqd

        write_log(prefix + "Compute L2 phi p error.")
        ref_phi_p_norm, coarse_phi_p_norm = compute_phi_p_error(
            solution_p_2_ref,
            reference_solution,
            ref_g,
            ref_dg0_space,
            dg0_space,
            coarse_levelset_2_ref,
            reference_levelset,
            dg0_coarse_h_T_2_ref,
            dg0_cut_indicator_2_ref,
        )

        with XDMFFile(
            reference_mesh.comm, os.path.join(errors_dir, "ref_l2_p_error.xdmf"), "w"
        ) as of:
            of.write_mesh(reference_mesh)
            of.write_function(ref_phi_p_norm)

        assert not np.any(np.isnan(ref_phi_p_norm.x.array)), (
            "ref_l2_p_err.x.array contains NaNs."
        )
        phi_p_err_sqd = ref_phi_p_norm.x.array.sum()

        triple_norm_err_sqd += phi_p_err_sqd

        write_log(prefix + "Compute phi error.")
        ref_phi_norm, coarse_phi_norm = compute_phi_error(
            solution_p_2_ref,
            reference_levelset,
            coarse_levelset_2_ref,
            dg0_coarse_h_T_2_ref,
            dg0_cut_indicator_2_ref,
            ref_dg0_space,
            dg0_space,
        )

        with XDMFFile(
            reference_mesh.comm, os.path.join(errors_dir, "ref_l2_p_error.xdmf"), "w"
        ) as of:
            of.write_mesh(reference_mesh)
            of.write_function(ref_phi_norm)

        assert not np.any(np.isnan(ref_phi_norm.x.array)), (
            "ref_l2_p_err.x.array contains NaNs."
        )
        phi_err_sqd = ref_phi_norm.x.array.sum()

        triple_norm_err_sqd += phi_err_sqd

        results["triple_norm_error"][i] = np.sqrt(triple_norm_err_sqd)

    # Source term oscillations estimation
    # Dirichlet data oscillations estimation
    if exact_solution_available:
        x = ufl.SpatialCoordinate(mesh)
        coarse_ref_solution = ref_solution(x)
        fh = -ufl.div(ufl.grad(coarse_ref_solution))
        gh = coarse_ref_solution
        dirichlet_data = generate_exact_solution(np)
    else:
        fh = dfx.fem.Function(fe_space)
        fh.interpolate(source_term)
        dirichlet_data = generate_dirichlet_data(np)

    gh = dfx.fem.Function(fe_space)
    gh.interpolate(dirichlet_data)

    """
    ref_osc_f, coarse_osc_f = compute_source_term_oscillations(
        fh,
        ref_f,
        dg0_coarse_h_T_2_ref,
        ref_dg0_space,
        dg0_space,
        fe_space,
        reference_space,
    )

    plot_scalar(
        coarse_osc_f, os.path.join(errors_dir, f"source_term_osc_{str(i).zfill(2)}")
    )
    results["source_term_osc"][i] = np.sqrt(ref_osc_f.x.array.sum())

    ref_osc_g, coarse_osc_g = compute_dirichlet_oscillations(
        gh,
        ref_g,
        ref_dg0_space,
        dg0_space,
        dg0_cut_indicator_2_ref,
        fe_space,
        reference_space,
    )

    results["dirichlet_data_osc"][i] = np.sqrt(ref_osc_g.x.array.sum())
    """

    df = pl.DataFrame(results)
    header = f"======================================================================================================\n{prefix}\n======================================================================================================"
    print(header)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))
