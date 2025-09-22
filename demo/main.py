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
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc
from phifem.mesh_scripts import compute_tags_measures
from utils import compute_boundary_correction, read_function, save_function

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Run the demo.",
    description="Run phiFEM with adaptive refinement steered by a residual estimator with boundary correction.",
)

demos_list = [demo for demo in next(os.walk("."))[1] if "__" not in demo]
parser.add_argument(
    "demo", type=str, choices=demos_list, help="Choose the demo to run."
)
parser.add_argument(
    "refinement",
    type=str,
    choices=["unif", "adap"],
    help="Choose the refinement type (choices: 'unif' or 'adap').",
)
parser.add_argument(
    "mesh_type",
    type=str,
    choices=["bg", "sub"],
    help="Choose the mesh type (choices: 'bg' to keep the background mesh, 'sub' to use the submesh at each step).",
)

args = parser.parse_args()
demo = args.demo
refinement = args.refinement
mesh_type = args.mesh_type

box_mode = mesh_type == "bg"

source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_phifem_" + refinement + "_" + mesh_type)
checkpoint_dir = os.path.join(output_dir, "checkpoints")

if os.path.isdir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

for dir_path in [output_dir, checkpoint_dir]:
    if not os.path.isdir(dir_path):
        print(f"{dir_path} directory not found, we create it.")
        os.mkdir(dir_path)

sys.path.append(source_dir)

from data import levelset, source_term

try:
    from data import detection_levelset
except ImportError:
    print(
        "Didn't find detection_levelset for the specified demo, using levelset instead."
    )
    detection_levelset = levelset

exact_sol_available = True
try:
    from data import exact_solution
except ImportError:
    exact_sol_available = False

with open(os.path.join(source_dir, "parameters.yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

initial_mesh_size = parameters["initial_mesh_size"]
iterations_num = parameters["iterations_number"]
fe_degree = parameters["finite_element_degree"]
levelset_degree = parameters["levelset_degree"]
solution_degree = parameters["solution_degree"]
detection_degree = parameters["boundary_detection_degree"]
stab_coef = parameters["stabilization_coefficient"]
dorfler_param = parameters["marking_parameter"]
bbox = parameters["bbox"]

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
    "eta_eps": [],
}

for i in range(iterations_num):
    try:
        if box_mode:
            cells_tags, facets_tags, _, ds, _, _ = compute_tags_measures(
                mesh, levelset, detection_degree, box_mode=box_mode
            )
        else:
            cells_tags, facets_tags, mesh, _, _, _ = compute_tags_measures(
                mesh, levelset, detection_degree, box_mode=box_mode
            )
            ds = ufl.Measure("ds", domain=mesh)
    except ValueError:
        break

    results["iteration"].append(i)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cells_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=facets_tags)

    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    cell_name = mesh.topology.cell_name()
    fe_element = element("Lagrange", cell_name, fe_degree)
    fe_space = dfx.fem.functionspace(mesh, fe_element)

    solution_element = element("Lagrange", cell_name, solution_degree)
    solution_space = dfx.fem.functionspace(mesh, solution_element)

    dg0_element = element("DG", cell_name, 0)
    dg0_space = dfx.fem.functionspace(mesh, dg0_element)

    levelset_element = element("Lagrange", cell_name, levelset_degree)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)

    omega_h_cells = np.union1d(cells_tags.find(1), cells_tags.find(2))
    cdim = mesh.topology.dim
    mesh.topology.create_connectivity(cdim, cdim)
    active_dofs = dfx.fem.locate_dofs_topological(fe_space, cdim, omega_h_cells)
    results["dof"].append(len(active_dofs))

    phih = dfx.fem.Function(levelset_space)
    phih.interpolate(levelset)

    fh = dfx.fem.Function(fe_space)
    fh.interpolate(source_term)

    # Bilinear form
    wh = ufl.TrialFunction(fe_space)
    uh = phih * wh
    zh = ufl.TestFunction(fe_space)
    vh = phih * zh

    def delta(u):
        return ufl.div(ufl.grad(u))

    stiffness = ufl.inner(ufl.grad(uh), ufl.grad(vh))

    boundary = ufl.inner(ufl.inner(ufl.grad(uh), n), vh)

    stabilization_facets = (
        stab_coef
        * ufl.avg(h_T)
        * ufl.inner(ufl.jump(ufl.grad(uh), n), ufl.jump(ufl.grad(vh), n))
    )
    stabilization_cells = stab_coef * h_T**2 * ufl.inner(delta(uh), delta(vh))

    a = (
        stiffness * dx((1, 2))
        - boundary * ds
        + stabilization_cells * dx(2)
        + stabilization_facets * dS(2)
    )

    # Linear form
    rhs = ufl.inner(fh, vh)
    stabilization_rhs = stab_coef * h_T**2 * ufl.inner(fh, delta(vh))

    L = rhs * dx((1, 2)) - stabilization_rhs * dx(2)

    # Assemble linear system
    bilinear_form = dfx.fem.form(a)
    A = assemble_matrix(bilinear_form)
    A.assemble()
    linear_form = dfx.fem.form(L)
    b = assemble_vector(linear_form)
    b.assemble()

    # PETSc solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setType("preonly")
    solver.setOperators(A)
    pc = solver.getPC()
    pc.setType("lu")

    # Let mumps handle the null space in box mode
    if box_mode:
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    # Solve linear system
    solution_w = dfx.fem.Function(fe_space)
    solver.solve(b, solution_w.x.petsc_vec)
    solver.destroy()

    # Compute the product of phih with solution_w to get solution_u
    solution_u = dfx.fem.Function(solution_space)
    solution_w_u = dfx.fem.Function(solution_space)
    levelset_u = dfx.fem.Function(solution_space)
    solution_w_u.interpolate(solution_w)
    levelset_u.interpolate(phih)
    solution_u.x.array[:] = solution_w_u.x.array[:] * levelset_u.x.array[:]

    save_function(
        solution_u, os.path.join(output_dir, f"solution_u_{str(i).zfill(2)}.xdmf")
    )

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{str(i).zfill(2)}.bp")
    adios4dolfinx.write_mesh(checkpoint_file, mesh)
    adios4dolfinx.write_function(checkpoint_file, solution_u, name="solution")

    # Residual error estimation
    k = solution_u.function_space.element.basix_element.degree
    quadrature_degree_cells = max(0, k - 2)
    quadrature_degree_facets = max(0, k - 1)

    dx_est = dx.reconstruct(
        metadata={"quadrature_degree": quadrature_degree_cells},
    )
    dS_est = dS.reconstruct(
        metadata={"quadrature_degree": quadrature_degree_cells},
    )
    ds_est = ds.reconstruct(metadata={"quadrature_degree": quadrature_degree_facets})

    rh = fh + delta(solution_u)
    Jh = ufl.jump(ufl.grad(solution_u), -n)

    w0 = ufl.TestFunction(dg0_space)

    eta_T = h_T**2 * ufl.inner(ufl.inner(rh, rh), w0) * dx_est((1, 2))
    eta_E = ufl.avg(h_T) * ufl.inner(ufl.inner(Jh, Jh), ufl.avg(w0)) * dS_est((1, 2))

    correction_fct = dfx.fem.Function(fe_space)
    eta_eps = correction_fct * w0 * dx_est((1, 2))

    eta = eta_T + eta_E + eta_eps

    eta_dict = {
        "eta_T": eta_T,
        "eta_E": eta_E,
        "eta_eps": eta_eps,
        "H10_estimator": eta,
    }
    for name, e in eta_dict.items():
        e_form = dfx.fem.form(e)
        e_vec = assemble_vector(e_form)
        e_h = dfx.fem.Function(dg0_space)
        e_h.x.array[:] = e_vec.array[:]

        save_function(
            e_h, os.path.join(output_dir, name + "_" + str(i).zfill(2) + ".xdmf")
        )
        e_val = np.sqrt(e_vec.array.sum())
        results[name].append(e_val)

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))

    norm_h10_local = (
        h_T ** (-2)
        * ufl.inner(ufl.grad(solution_u), ufl.grad(solution_u))
        * w0
        * dx((1, 2))
    )

    norm_h10_form = dfx.fem.form(norm_h10_local)
    norm_h10_vec = assemble_vector(norm_h10_form)

    norm_h10_h = dfx.fem.Function(dg0_space)
    norm_h10_h.x.array[:] = norm_h10_vec.array[:]

    save_function(
        norm_h10_h, os.path.join(output_dir, f"norm_H10_{str(i).zfill(2)}.xdmf")
    )

    # Marking and refinement
    if i < iterations_num - 1:
        if refinement == "adap":
            eta_global = e_h.x.array.sum()
            cutoff = dorfler_param * eta_global

            sorted_cells = np.argsort(e_h.x.array)[::-1]
            rolling_sum = 0.0
            breakpt = 0
            for j, e in enumerate(e_h.x.array[sorted_cells]):
                rolling_sum += e
                if rolling_sum > cutoff:
                    breakpt = j
                    break

            refine_cells = sorted_cells[0 : breakpt + 1]
            indices = np.array(np.sort(refine_cells), dtype=np.int32)
            fdim = cdim - 1
            c2f_connect = mesh.topology.connectivity(cdim, fdim)
            num_facets_per_cell = len(c2f_connect.links(0))
            c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
            facets_indices = np.unique(np.sort(c2f_map[indices]))
            mesh = dfx.mesh.refine(mesh, facets_indices)[0]
        elif refinement == "unif":
            mesh = dfx.mesh.refine(mesh)[0]

results["Reference_error"] = [np.nan] * i
if exact_sol_available:
    fine_space = fe_space
    fine_solution = dfx.fem.Function(fine_space)
    fine_solution.interpolate(exact_solution)

gap = 3
for j in range(i - 3):

    def checkpoint_file(k):
        return os.path.join(checkpoint_dir, f"checkpoint_{str(k).zfill(2)}.bp")

    coarse_mesh = adios4dolfinx.read_mesh(checkpoint_file(j), comm=MPI.COMM_WORLD)
    coarse_space = dfx.fem.functionspace(coarse_mesh, fe_element)
    coarse_solution = dfx.fem.Function(coarse_space)
    adios4dolfinx.read_function(checkpoint_file(j), coarse_solution, name="solution")
    if not exact_sol_available:
        fine_mesh = adios4dolfinx.read_mesh(
            checkpoint_file(j + gap), comm=MPI.COMM_WORLD
        )
        fine_space = dfx.fem.functionspace(fine_mesh, fe_element)
        fine_solution = dfx.fem.Function(fine_space)
        adios4dolfinx.read_function(
            checkpoint_file(j + gap), fine_solution, name="solution"
        )
    dg0_fine_space = dfx.fem.functionspace(fine_mesh, dg0_element)
    tdim = fine_mesh.topology.dim
    fine_cells = fine_mesh.topology.index_map(tdim).size_global
    nmm = dfx.fem.create_interpolation_data(
        fine_space, coarse_space, np.arange(fine_cells), padding=1.0e-14
    )
    cf_solution = dfx.fem.Function(fine_space)
    coarse_cells = coarse_mesh.topology.index_map(tdim).size_global
    cf_solution.interpolate_nonmatching(coarse_solution, np.arange(fine_cells), nmm)

    save_function(
        cf_solution, os.path.join(output_dir, f"cf_solution_{str(j).zfill(2)}.xdmf")
    )
    diff = dfx.fem.Function(fine_space)
    diff.x.array[:] = fine_solution.x.array[:] - cf_solution.x.array[:]

    save_function(
        diff, os.path.join(output_dir, f"reference_diff_{str(j).zfill(2)}.xdmf")
    )

    cells_tags = compute_tags_measures(
        fine_mesh, levelset, detection_degree, box_mode=box_mode
    )[0]

    dx_fine = ufl.Measure("dx", domain=fine_mesh, subdomain_data=cells_tags)
    fine_v0 = ufl.TestFunction(dg0_fine_space)
    grad_diff = ufl.grad(diff)
    h10_norm_diff = ufl.inner(grad_diff, grad_diff) * fine_v0 * dx_fine((1, 2))
    h10_norm_form = dfx.fem.form(h10_norm_diff)
    print("Assemble h10 norm")
    h10_norm_vec = assemble_vector(h10_norm_form)
    print("Post assembly")

    h10_norm_h = dfx.fem.Function(dg0_fine_space)
    h10_norm_h.x.array[:] = h10_norm_vec.array[:]

    save_function(
        h10_norm_h, os.path.join(output_dir, f"reference_error_{str(j).zfill(2)}.xdmf")
    )

    results["Reference_error"][j] = np.sqrt(h10_norm_vec.array.sum())

    df = pl.DataFrame(results)
    print(df)
    df.write_csv(os.path.join(output_dir, "results.csv"))
