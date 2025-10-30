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
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI

sys.path.append("../")

from plots import plot_scalar
from utils import cell_diameter, compute_xi_ref, fem_solve

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
for dir_path in dirs:
    if not os.path.isdir(dir_path):
        print(f"{dir_path} directory not found, we create it.")
        os.mkdir(dir_path)

sys.path.append(source_dir)

from data import generate_dirichlet_data, generate_levelset, generate_source_term

with open(os.path.join(source_dir, parameters_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

iterations_num = parameters["iterations_number"]
fe_degree = parameters["finite_element_degree"]
levelset_degree = parameters["levelset_degree"]

with open(os.path.join(source_dir, "reference.yaml"), "rb") as f:
    ref_parameters = yaml.safe_load(f)

ref_iterations_num = ref_parameters["iterations_number"] - 1
reference_mesh = adios4dolfinx.read_mesh(
    os.path.join(
        source_dir,
        "output_reference",
        "checkpoints",
        f"checkpoint_{str(ref_iterations_num).zfill(2)}.bp",
    ),
    comm=MPI.COMM_WORLD,
)

cdim = reference_mesh.topology.dim
num_reference_cells = reference_mesh.topology.index_map(cdim).size_global
reference_cells = np.arange(num_reference_cells)

cell_name = reference_mesh.topology.cell_name()
cg1_element = element("CG", cell_name, fe_degree)
cg2_element = element("CG", cell_name, fe_degree + 1)
levelset_element = element("Lagrange", cell_name, levelset_degree)
dg0_element = element("DG", cell_name, 0)
dg1_element = element("DG", cell_name, 1)
dg2_element = element("DG", cell_name, 2)

reference_space = dfx.fem.functionspace(reference_mesh, cg2_element)
dg0_ref_space = dfx.fem.functionspace(reference_mesh, dg0_element)

source_term = generate_source_term(np)
f_ref = dfx.fem.Function(reference_space)
f_ref.interpolate(source_term)
dirichlet_data = generate_dirichlet_data(np)
g_ref = dfx.fem.Function(reference_space)
g_ref.interpolate(dirichlet_data)

levelset = generate_levelset(np)

reference_solution = fem_solve(reference_space, f_ref, g_ref)

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
    fe_levelset = dfx.fem.Function(levelset_space)
    solution_p = dfx.fem.Function(dg1_space)
    fe_levelset = dfx.fem.Function(fe_space)
    fe_levelset.interpolate(levelset)

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

    plot_scalar(
        cut_indicator, os.path.join(errors_dir, f"cut_indicator_{str(i).zfill(2)}")
    )
    solution_ref = dfx.fem.Function(reference_space)

    num_cells = mesh.topology.index_map(cdim).size_global
    all_cells = np.arange(num_cells)
    nmm_coarse_space2ref_space = dfx.fem.create_interpolation_data(
        reference_space, fe_space, reference_cells, padding=1.0e-14
    )
    solution_ref.interpolate_nonmatching(
        solution_u, reference_cells, nmm_coarse_space2ref_space
    )

    ref_cut_indicator = dfx.fem.Function(dg0_ref_space)
    nmm_coarse_dg0_space2ref_dg0_space = dfx.fem.create_interpolation_data(
        dg0_ref_space, dg0_space, reference_cells, padding=1.0e-14
    )
    ref_cut_indicator.interpolate_nonmatching(
        cut_indicator, reference_cells, nmm_coarse_dg0_space2ref_dg0_space
    )

    plot_scalar(
        ref_cut_indicator,
        os.path.join(errors_dir, f"ref_cut_indicator_{str(i).zfill(2)}"),
    )

    diff = dfx.fem.Function(reference_space)
    diff.x.array[:] = reference_solution.x.array[:] - solution_ref.x.array[:]

    grad_diff = ufl.grad(diff)
    ref_v0 = ufl.TestFunction(dg0_ref_space)
    h10_norm_diff = ufl.inner(grad_diff, grad_diff) * ref_v0 * ufl.dx
    h10_norm_form = dfx.fem.form(h10_norm_diff)
    h10_norm_vec = assemble_vector(h10_norm_form)
    h10_norm_ref = dfx.fem.Function(dg0_ref_space)
    h10_norm_ref.x.array[:] = h10_norm_vec.array[:]

    h10_err = h10_norm_ref.x.array.sum()
    results["error"][i] = np.sqrt(h10_err)

    ref_mesh_indicator = dfx.fem.Function(dg0_ref_space)
    ref_mesh_indicator.x.array[:] = 1.0

    coarse_mesh = solution_u.function_space.mesh
    cf_mesh = dfx.mesh.refine(coarse_mesh)[0]
    cf_mesh.topology.create_entities(cdim - 1)
    cf_mesh = dfx.mesh.refine(cf_mesh)[0]

    cf_cg2_space = dfx.fem.functionspace(cf_mesh, cg2_element)
    cf_dg2_space = dfx.fem.functionspace(cf_mesh, dg2_element)
    cf_dg0_space = dfx.fem.functionspace(cf_mesh, dg0_element)
    cf_num_cells = cf_mesh.topology.index_map(cdim).size_global
    cf_all_cells = np.arange(cf_num_cells)
    nmm_fe_space2cf_space = dfx.fem.create_interpolation_data(
        cf_cg2_space, fe_space, cf_all_cells, padding=1.0e-14
    )
    nmm_levelset_space2cf_space = dfx.fem.create_interpolation_data(
        cf_cg2_space, levelset_space, cf_all_cells, padding=1.0e-14
    )
    nmm_dg1_space2cf_dg2_space = dfx.fem.create_interpolation_data(
        cf_dg2_space, dg1_space, cf_all_cells, padding=1.0e-14
    )
    nmm_ref_space2cf_space = dfx.fem.create_interpolation_data(
        cf_cg2_space, reference_space, cf_all_cells, padding=1.0e-14
    )
    nmm_dg0_space2cf_dg0_space = dfx.fem.create_interpolation_data(
        cf_dg0_space, dg0_space, cf_all_cells, padding=1.0e-14
    )
    nmm_dg0_ref_space2cf_dg0_space = dfx.fem.create_interpolation_data(
        cf_dg0_space, dg0_ref_space, cf_all_cells, padding=1.0e-14
    )
    cf_solution_p = dfx.fem.Function(cf_dg2_space)
    cf_cut_indicator = dfx.fem.Function(cf_dg0_space)
    cf_levelset = dfx.fem.Function(cf_cg2_space)

    cf_levelset.interpolate_nonmatching(
        fe_levelset, cf_all_cells, nmm_levelset_space2cf_space
    )

    cf_solution_p.interpolate_nonmatching(
        solution_p, cf_all_cells, nmm_dg1_space2cf_dg2_space
    )
    cf_cut_indicator.interpolate_nonmatching(
        ref_cut_indicator, cf_all_cells, nmm_dg0_ref_space2cf_dg0_space
    )

    plot_scalar(
        cf_cut_indicator,
        os.path.join(errors_dir, f"cf_cut_indicator_{str(i).zfill(2)}"),
    )

    coarse_mesh_h_T = cell_diameter(dg0_space)
    cf_coarse_mesh_h_T = dfx.fem.Function(cf_dg0_space)
    cf_coarse_mesh_h_T.interpolate_nonmatching(
        coarse_mesh_h_T, cf_all_cells, nmm_dg0_space2cf_dg0_space
    )

    cf_g = dfx.fem.Function(cf_cg2_space)
    cf_g.interpolate(dirichlet_data)

    cf_reference_solution = dfx.fem.Function(cf_cg2_space)
    cf_reference_solution.interpolate_nonmatching(
        reference_solution, cf_all_cells, nmm_ref_space2cf_space
    )

    h_t_reference_p = cf_reference_solution - cf_g

    p_diff = h_t_reference_p - (cf_solution_p * cf_levelset) / cf_coarse_mesh_h_T

    cf_v0 = ufl.TestFunction(cf_dg0_space)
    l2_norm_p_int = (
        cf_coarse_mesh_h_T ** (-2)
        * ufl.inner(p_diff, p_diff)
        * cf_cut_indicator
        * cf_v0
        * ufl.dx
    )
    l2_norm_p_form = dfx.fem.form(l2_norm_p_int)
    l2_p_err_vec = dfx.fem.assemble_vector(l2_norm_p_form)

    cf_l2_p_err = dfx.fem.Function(cf_dg0_space)
    cf_l2_p_err.x.array[:] = l2_p_err_vec.array[:]

    plot_scalar(cf_l2_p_err, os.path.join(errors_dir, f"l2_p_err_{str(i).zfill(2)}"))
    l2_p_err = l2_p_err_vec.array.sum()

    results["l2_p_error"][i] = np.sqrt(l2_p_err)
    triple_norm_err = np.sqrt(h10_err + l2_p_err)
    results["triple_norm_error"][i] = triple_norm_err

    try:
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, dg0_ref_space, all_cells, padding=1.0e-20
        )
        h10_norm_coarse = dfx.fem.Function(dg0_space)
        h10_norm_coarse.interpolate_nonmatching(
            h10_norm_ref, all_cells, nmm_ref_space2coarse_space
        )
        plot_scalar(
            h10_norm_coarse,
            os.path.join(errors_dir, f"error_{str(i).zfill(2)}"),
        )
    except RuntimeError:
        print(
            f"Failed to interpolate h10_norm to coarse space at iteration {str(i).zfill(2)}."
        )
        pass

    # Source term oscillations estimation
    fh = dfx.fem.Function(fe_space)
    fh.interpolate(source_term)

    fh_ref = dfx.fem.Function(reference_space)
    fh_ref.interpolate_nonmatching(fh, reference_cells, nmm_coarse_space2ref_space)
    f_diff = dfx.fem.Function(reference_space)
    f_diff.x.array[:] = fh_ref.x.array[:] - f_ref.x.array[:]

    h_T_ref = cell_diameter(dg0_ref_space)

    osc_f_int = h_T_ref**2 * ufl.inner(f_diff, f_diff) * ref_v0 * ufl.dx
    osc_f_form = dfx.fem.form(osc_f_int)
    osc_f_vec = assemble_vector(osc_f_form)
    osc_f_ref = dfx.fem.Function(dg0_ref_space)
    osc_f_ref.x.array[:] = osc_f_vec.array[:]

    results["source_term_osc"][i] = np.sqrt(osc_f_ref.x.array.sum())

    # Dirichlet data oscillations estimation
    gh = dfx.fem.Function(fe_space)
    gh.interpolate(dirichlet_data)

    gh_ref = dfx.fem.Function(reference_space)
    gh_ref.interpolate_nonmatching(gh, reference_cells, nmm_coarse_space2ref_space)
    g_diff = dfx.fem.Function(reference_space)
    g_diff.x.array[:] = gh_ref.x.array[:] - g_ref.x.array[:]

    osc_g_int = (
        ref_cut_indicator
        * ufl.inner(ufl.grad(g_diff), ufl.grad(g_diff))
        * ref_v0
        * ufl.dx
    )
    osc_g_form = dfx.fem.form(osc_g_int)
    osc_g_vec = assemble_vector(osc_g_form)
    osc_g_ref = dfx.fem.Function(dg0_ref_space)
    osc_g_ref.x.array[:] = osc_g_vec.array[:]

    results["dirichlet_data_osc"][i] = np.sqrt(osc_g_ref.x.array.sum())

    results["total_error"][i] = np.sqrt(
        h10_err + l2_p_err + osc_f_ref.x.array.sum() + osc_g_ref.x.array.sum()
    )

    xi_ref = compute_xi_ref(solution_ref, gh_ref)

    xi_ref_h10_int = ufl.inner(ufl.grad(xi_ref), ufl.grad(xi_ref)) * ref_v0 * ufl.dx
    xi_ref_h10_form = dfx.fem.form(xi_ref_h10_int)
    xi_ref_h10_vec = assemble_vector(xi_ref_h10_form)
    xi_ref_h10_ref = dfx.fem.Function(dg0_ref_space)
    xi_ref_h10_ref.x.array[:] = xi_ref_h10_vec.array[:]

    results["xi_h10"][i] = np.sqrt(xi_ref_h10_ref.x.array.sum())

    xi_ref_l2_int = h_T_ref ** (-1) * ufl.inner(xi_ref, xi_ref) * ref_v0 * ufl.dx
    xi_ref_l2_form = dfx.fem.form(xi_ref_l2_int)
    xi_ref_l2_vec = assemble_vector(xi_ref_l2_form)
    xi_ref_l2_ref = dfx.fem.Function(dg0_ref_space)
    xi_ref_l2_ref.x.array[:] = xi_ref_l2_vec.array[:]

    results["xi_l2"][i] = np.sqrt(xi_ref_l2_ref.x.array.sum())

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))
