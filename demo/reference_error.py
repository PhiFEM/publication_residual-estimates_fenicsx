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
from dolfinx.io import XDMFFile
from mpi4py import MPI
from phifem.mesh_scripts import compute_tags_measures

sys.path.append("../")

from utils import (
    cell_diameter,
    compute_boundary_error,
    compute_h10_error,
    nmm_interpolation,
    phifem_solve,
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

from data import (
    MAX_EXTRA_STEP_UNIF,
    generate_exact_solution,
    generate_levelset,
)

if "phifem" not in parameters_name:
    REFERENCE = parameters_name

exact_solution_available = False
try:
    from data import generate_exact_solution

    exact_solution_available = True
except ImportError:
    pass

if not exact_solution_available:
    from data import generate_dirichlet_data, generate_source_term

    data_fcts = {
        "dirichlet": generate_dirichlet_data,
        "source_term": generate_source_term,
    }

with open(os.path.join(source_dir, parameters_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)
fe_degree = parameters["finite_element_degree"]

phifem = "phifem" in parameters_name
if phifem:
    try:
        from data import generate_detection_levelset
    except ImportError:
        from data import generate_levelset

        generate_detection_levelset = generate_levelset

    levelset_degree = parameters["levelset_degree"]

detection_degree = parameters["boundary_detection_degree"]
auxiliary_degree = parameters["auxiliary_degree"]
degrees = {
    "fe": fe_degree,
    "levelset": levelset_degree,
    "detection": detection_degree,
    "aux": auxiliary_degree,
}
pen_coef = parameters["penalization_coefficient"]
stab_coef = parameters["stabilization_coefficient"]
coefs = {"penalization": pen_coef, "stabilization": stab_coef}
discretize_levelset = parameters["discretize_levelset"]
single_layer = parameters["single_layer"]
detection_parameters = [generate_detection_levelset, discretize_levelset, single_layer]

nums = []
for f in os.listdir(checkpoint_dir):
    if f.endswith(".bp"):
        num = f[:-3].split(sep="_")[-1]
        nums.append(int(num))
max_iteration = sorted(nums)[-1]

reference_mesh = adios4dolfinx.read_mesh(
    os.path.join(
        checkpoint_dir,
        f"checkpoint_{str(max_iteration).zfill(2)}.bp",
    ),
    comm=MPI.COMM_WORLD,
)
cell_name = reference_mesh.topology.cell_name()
fe_element = element("CG", cell_name, fe_degree)

for i in range(MAX_EXTRA_STEP_UNIF):
    reference_mesh.topology.create_entities(1)
    reference_mesh = dfx.mesh.refine(reference_mesh)[0]

if phifem:
    levelset_element = element("Lagrange", cell_name, levelset_degree)
    reference_levelset_space = dfx.fem.functionspace(reference_mesh, levelset_element)

reference_fe_space = dfx.fem.functionspace(reference_mesh, fe_element)

if ("phifem" in parameters_name) or exact_solution_available:
    reference_solution = dfx.fem.Function(reference_fe_space)
    reference_solution.interpolate(generate_exact_solution(np))
    reference_detection_levelset = dfx.fem.Function(reference_levelset_space)
    reference_detection_levelset.interpolate(generate_detection_levelset(np))
    reference_cells_tags, _, _, _, _, _ = compute_tags_measures(
        reference_mesh,
        reference_detection_levelset,
        degrees["detection"],
        box_mode=True,
        single_layer_cut=single_layer,
    )
    ref_g = dfx.fem.Function(reference_fe_space)
    if exact_solution_available:
        ref_g = reference_solution
    else:
        ref_g.interpolate(generate_dirichlet_data(np))

elif "phifem" in parameters_name:
    (
        reference_solution,
        _,
        ref_f,
        ref_g,
        _,
        reference_levelset,
        reference_cells_tags,
        _,
        reference_measures,
    ) = phifem_solve(reference_mesh, degrees, coefs, detection_parameters, data_fcts)[0]
else:
    pass
    # reference_solution = fem_solve(
    #     reference_mesh, degrees, coefs, detection_parameters, data_fcts
    # )[0]


cdim = reference_mesh.topology.dim
num_reference_cells = reference_mesh.topology.index_map(cdim).size_global
reference_cells = np.arange(num_reference_cells)
cell_name = reference_mesh.topology.cell_name()

# Define finite elements and reference FE spaces
cg1_element = element("Lagrange", cell_name, fe_degree)
ref_element = element("Lagrange", cell_name, fe_degree)
dg0_element = element("DG", cell_name, 0)

reference_space = dfx.fem.functionspace(reference_mesh, ref_element)
ref_dg0_space = dfx.fem.functionspace(reference_mesh, dg0_element)

if phifem:
    levelset = generate_levelset(np)
    reference_levelset = dfx.fem.Function(reference_levelset_space)
    reference_levelset.interpolate(levelset)

# Allocate memory for results
results = pl.read_csv(os.path.join(output_dir, "results.csv")).to_dict()
iterations_num = len(results["iteration"])
results["h10_error"] = [np.nan] * iterations_num
results["l2_error"] = [np.nan] * iterations_num
results["boundary_error_phih_gh"] = [np.nan] * iterations_num
results["boundary_error_phi_g"] = [np.nan] * iterations_num
results["phi_p_error"] = [np.nan] * iterations_num
results["triple_norm_error"] = [np.nan] * iterations_num

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

    dg0_space = dfx.fem.functionspace(mesh, dg0_element)
    solution_u = dfx.fem.Function(fe_space)
    solution_p = dfx.fem.Function(aux_space)
    coarse_g_h = dfx.fem.Function(fe_space)
    if exact_solution_available:
        coarse_g_h.interpolate(generate_exact_solution(np))
    else:
        coarse_g_h.interpolate(generate_dirichlet_data(np))

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

        dg0_cut_indicator_2_ref, nmm_dg02ref_dg0 = nmm_interpolation(
            ref_dg0_space, cut_indicator
        )
        coarse_levelset_2_ref, nmm_levelset2ref_levelset = nmm_interpolation(
            reference_levelset_space, fe_levelset
        )
        solution_p_2_ref, nmm_aux2ref = nmm_interpolation(reference_space, solution_p)
        coarse_mesh_h_T = cell_diameter(dg0_space)
        dg0_coarse_h_T_2_ref = nmm_interpolation(
            ref_dg0_space, coarse_mesh_h_T, interpolation_data=nmm_dg02ref_dg0
        )[0]

    solution_u_2_ref, nmm_fe2ref = nmm_interpolation(reference_space, solution_u)
    coarse_g_h_2_ref = nmm_interpolation(reference_space, coarse_g_h)[0]

    # Compute reference H10 error
    write_log(prefix + "Compute H10 error.")
    ref_h10_norm, coarse_h10_norm = compute_h10_error(
        solution_u_2_ref,
        reference_solution,
        ref_dg0_space,
        dg0_space,
        ref_cells_tags=reference_cells_tags,
    )

    with XDMFFile(
        mesh.comm,
        os.path.join(errors_dir, f"h10_error_{str(i).zfill(2)}.xdmf"),
        "w",
    ) as of:
        of.write_mesh(mesh)
        of.write_function(coarse_h10_norm)

    h10_err_sqd = ref_h10_norm.x.array.sum()
    results["h10_error"][i] = np.sqrt(h10_err_sqd)

    if phifem:
        # Compute boundary error u - phih ph - gh
        (
            ref_boundary_error_phih_gh_norm,
            coarse_boundary_error_phih_gh_norm,
        ) = compute_boundary_error(
            reference_solution,
            coarse_levelset_2_ref,
            solution_p_2_ref,
            coarse_g_h_2_ref,
            ref_dg0_space,
            dg0_cut_indicator_2_ref,
            dg0_space,
            ref_cells_tags=reference_cells_tags,
        )

        with XDMFFile(
            reference_mesh.comm,
            os.path.join(errors_dir, f"boundary_error_phih_gh_{str(i).zfill(2)}.xdmf"),
            "w",
        ) as of:
            of.write_mesh(mesh)
            of.write_function(coarse_boundary_error_phih_gh_norm)

        assert not np.any(np.isnan(ref_boundary_error_phih_gh_norm.x.array)), (
            "ref_boundary_error_phih_gh_norm.x.array contains NaNs."
        )
        boundary_err_phih_gh_sqd = ref_boundary_error_phih_gh_norm.x.array.sum()

        results["boundary_error_phih_gh"][i] = np.sqrt(boundary_err_phih_gh_sqd)

        # Compute boundary error u - phi ph - g
        (
            ref_boundary_error_phi_g_norm,
            coarse_boundary_error_phi_g_norm,
        ) = compute_boundary_error(
            reference_solution,
            reference_levelset,
            solution_p_2_ref,
            ref_g,
            ref_dg0_space,
            dg0_cut_indicator_2_ref,
            dg0_space,
            ref_cells_tags=reference_cells_tags,
        )

        with XDMFFile(
            reference_mesh.comm,
            os.path.join(errors_dir, f"boundary_error_phi_g_{str(i).zfill(2)}.xdmf"),
            "w",
        ) as of:
            of.write_mesh(mesh)
            of.write_function(coarse_boundary_error_phi_g_norm)

        assert not np.any(np.isnan(ref_boundary_error_phi_g_norm.x.array)), (
            "ref_boundary_error_phi_g_norm.x.array contains NaNs."
        )
        boundary_err_phi_g_sqd = ref_boundary_error_phi_g_norm.x.array.sum()

        results["boundary_error_phi_g"][i] = np.sqrt(boundary_err_phi_g_sqd)

        triple_norm_err_sqd = h10_err_sqd
        triple_norm_err_sqd += boundary_err_phih_gh_sqd
        triple_norm_err_sqd += boundary_err_phi_g_sqd

        results["triple_norm_error"][i] = np.sqrt(triple_norm_err_sqd)
    else:
        results["triple_norm_error"][i] = np.nan

    df = pl.DataFrame(results)
    header = f"======================================================================================================\n{prefix}\n======================================================================================================"
    print(header)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))
