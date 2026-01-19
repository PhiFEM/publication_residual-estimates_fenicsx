import datetime
import os

import dolfinx as dfx
import numpy as np
import pyvista as pv
import ufl
from basix.ufl import element
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from matplotlib.colors import ListedColormap
from petsc4py import PETSc

parent_dir = os.path.dirname(__file__)


def write_log(text):
    dt = str(datetime.datetime.now()).split(sep=".")[0]
    with open(os.path.join(parent_dir, "run.log"), "a") as log_file:
        log_file.write(dt + "\t" + text + "\n")


def _reshape_facets_map(f2c_connect):
    """Reshape the facets-to-cells indices mapping.

    Args:
        f2c_connect: the facets-to-cells connectivity.

    Returns:
        The facets-to-cells mapping as a ndarray.
    """
    f2c_array = f2c_connect.array
    num_cells_per_facet = np.diff(f2c_connect.offsets)
    max_cells_per_facet = num_cells_per_facet.max()
    f2c_map = -np.ones((len(f2c_connect.offsets) - 1, max_cells_per_facet), dtype=int)

    # Mask to select the boundary facets
    mask = np.where(num_cells_per_facet == 1)
    f2c_map[mask, 0] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    f2c_map[mask, 1] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    # Mask to select the interior facets
    mask = np.where(num_cells_per_facet == 2)
    f2c_map[mask, 0] = f2c_array[num_cells_per_facet.cumsum()[mask] - 2]
    f2c_map[mask, 1] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    return f2c_map


def delta(u):
    return ufl.div(ufl.grad(u))


def compute_parent_cells(coarse_mesh, fine_mesh, initial_parent_cells):
    cdim = coarse_mesh.geometry.dim
    num_cells_dummy = coarse_mesh.topology.index_map(cdim).size_global
    parent_ct_indices = np.arange(num_cells_dummy).astype(np.int32)
    parent_ct_markers = parent_ct_indices
    sorted_indices = np.argsort(parent_ct_indices)
    parent_ct = dfx.mesh.meshtags(
        fine_mesh,
        cdim,
        parent_ct_indices[sorted_indices],
        parent_ct_markers[sorted_indices],
    )
    parent_cells_tags = dfx.mesh.transfer_meshtag(
        parent_ct, fine_mesh, initial_parent_cells
    )
    parent_cells = parent_cells_tags.values
    return parent_cells


def compute_boundary_local_estimators(
    coarse_mesh,
    solution_p,
    levelset,
    phih,
    cells_tags,
    dual=False,
    padding=1.0e-14,
):
    cdim = coarse_mesh.topology.dim
    submesh, cmap = dfx.mesh.create_submesh(coarse_mesh, cdim, cells_tags.find(2))[:2]
    submesh.topology.create_entities(cdim - 1)
    fine_mesh, parent_cells = dfx.mesh.refine(
        submesh, option=RefinementOption.parent_cell
    )[:2]

    parent_cells = compute_parent_cells(submesh, fine_mesh, parent_cells)

    fine_cells_tags = dfx.mesh.transfer_meshtag(cells_tags, fine_mesh, parent_cells)

    phih_space = phih.function_space
    fine_element = phih_space.ufl_element()
    fine_space = dfx.fem.functionspace(fine_mesh, fine_element)

    num_fine_cells = fine_mesh.topology.index_map(cdim).size_global
    fine_cells = np.arange(num_fine_cells)
    nmm_phih_space2fine_space = dfx.fem.create_interpolation_data(
        fine_space, phih_space, fine_cells, padding=padding
    )
    phih_fine = dfx.fem.Function(fine_space)
    phih_fine.interpolate_nonmatching(phih, fine_cells, nmm_phih_space2fine_space)

    phi_fine = dfx.fem.Function(fine_space)
    phi_fine.interpolate(levelset)

    nmm_solution_w_space2fine_space = dfx.fem.create_interpolation_data(
        fine_space, solution_p.function_space, fine_cells, padding=padding
    )
    solution_p_fine = dfx.fem.Function(fine_space)
    solution_p_fine.interpolate_nonmatching(
        solution_p, fine_cells, nmm_solution_w_space2fine_space
    )

    cell_name = fine_mesh.topology.cell_name()
    dg0_element = element("DG", cell_name, 0)
    dg0_coarse_space = dfx.fem.functionspace(coarse_mesh, dg0_element)
    dg0_fine_space = dfx.fem.functionspace(fine_mesh, dg0_element)

    dx = ufl.Measure("dx", domain=fine_mesh)
    v0 = ufl.TestFunction(dg0_fine_space)

    h_T_coarse = cell_diameter(dg0_coarse_space)
    correction_function_fine = dfx.fem.Function(fine_space)
    correction_function_fine.x.array[:] = (
        phih_fine.x.array[:] - phi_fine.x.array[:]
    ) * solution_p_fine.x.array[:]
    correction_function_fine = correction_function_fine
    grad_correction = ufl.grad(correction_function_fine)
    if dual:
        measure_ind = 2
    else:
        measure_ind = (1, 2)

    # eta_{1,z}
    h10_norm_correction = ufl.inner(grad_correction, grad_correction) * v0 * dx
    h10_norm_correction_form = dfx.fem.form(h10_norm_correction)
    h10_norm_correction_vec = assemble_vector(h10_norm_correction_form)

    h10_norm_dg0 = dfx.fem.Function(dg0_coarse_space)
    h10_norm_dg0.x.array[cmap] = np.bincount(
        cmap[parent_cells], weights=h10_norm_correction_vec.array[:]
    )[cmap]

    # eta_{0,z}
    l2_norm_correction = (
        ufl.inner(correction_function_fine, correction_function_fine)
        * v0
        * dx(measure_ind)
    )
    l2_norm_correction_form = dfx.fem.form(l2_norm_correction)
    l2_norm_correction_vec = assemble_vector(l2_norm_correction_form)

    l2_norm_dg0_fine = dfx.fem.Function(dg0_fine_space)
    l2_norm_dg0_fine.x.array[:] = l2_norm_correction_vec.array[:] * h_T_coarse.x.array[
        :
    ] ** (-2)
    l2_norm_dg0 = dfx.fem.Function(dg0_coarse_space)
    l2_norm_dg0.x.array[cmap] = np.bincount(
        cmap[parent_cells], weights=l2_norm_correction_vec.array[:]
    )[cmap]

    fine_submesh, emap = dfx.mesh.create_submesh(
        fine_mesh, cdim, fine_cells_tags.find(2)
    )[:2]
    fine_submesh_indices = fine_cells_tags.indices[emap]
    fine_submesh_markers = fine_cells_tags.values[emap]
    sorted_indices = np.argsort(fine_submesh_indices)
    fine_submesh_tags = dfx.mesh.meshtags(
        fine_submesh,
        cdim,
        fine_submesh_indices[sorted_indices],
        fine_submesh_markers[sorted_indices],
    )

    geo_dg0 = dfx.fem.Function(dg0_coarse_space)
    geo_dg0.x.array[:] = h10_norm_dg0.x.array[:] + l2_norm_dg0.x.array[:]
    return geo_dg0, fine_submesh_tags, fine_submesh


def phifem_direct_solve(spaces, fh, phih, measures, coefs):
    fe_space, solution_space = (spaces["primal"], spaces["solution"])
    dx, dS, ds = (measures["dx"], measures["dS"], measures["ds"])
    stab_coef = coefs["stabilization"]

    mesh = fe_space.mesh
    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    wh = ufl.TrialFunction(fe_space)
    uh = phih * wh
    zh = ufl.TestFunction(fe_space)
    vh = phih * zh

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

    return solution_u, solution_w


def phifem_dual_solve(mixed_space, fh, gh, phih, measures, coefs):
    dx, dS, ds = (measures["dx"], measures["dS"], measures["ds"])
    pen_coef, stab_coef = (coefs["penalization"], coefs["stabilization"])

    mesh = mixed_space.mesh
    h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    uh, ph = ufl.TrialFunctions(mixed_space)
    vh, qh = ufl.TestFunctions(mixed_space)

    stiffness = ufl.inner(ufl.grad(uh), ufl.grad(vh))

    boundary = ufl.inner(ufl.inner(ufl.grad(uh), n), vh)

    stabilization_facets = (
        stab_coef
        * ufl.avg(h_T)
        * ufl.inner(ufl.jump(ufl.grad(uh), n), ufl.jump(ufl.grad(vh), n))
    )
    stabilization_cells = stab_coef * h_T**2 * ufl.inner(delta(uh), delta(vh))

    penalization = pen_coef * h_T ** (-2) * ufl.inner(uh - ph * phih, vh - qh * phih)

    a = (
        stiffness * dx((1, 2))
        - boundary * ds
        + penalization * dx(2)
        + stabilization_cells * dx(2)
        + stabilization_facets * dS(2)
    )

    # Linear form
    rhs = ufl.inner(fh, vh)
    penalization_rhs = pen_coef * h_T ** (-2) * ufl.inner(gh, vh - qh * phih)
    stabilization_rhs = stab_coef * h_T**2 * ufl.inner(fh, delta(vh))

    L = rhs * dx((1, 2)) + penalization_rhs * dx(2) - stabilization_rhs * dx(2)

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
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    # Solve linear system
    solution_w = dfx.fem.Function(mixed_space)
    solver.solve(b, solution_w.x.petsc_vec)
    solver.destroy()

    solution_u, solution_p = solution_w.split()
    solution_u = solution_u.collapse()
    solution_p = solution_p.collapse()

    return solution_u, solution_p


def fem_solve(fe_space, fh, gh):
    mesh = fe_space.mesh
    tdim = mesh.topology.dim
    dx = ufl.Measure("dx", domain=mesh)

    bcs = []
    boundary_facets = dfx.mesh.locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.ones_like(x[0]).astype(bool)
    )
    dofs_D = dfx.fem.locate_dofs_topological(fe_space, tdim - 1, boundary_facets)
    bc = dfx.fem.dirichletbc(gh, dofs_D)
    bcs = [bc]

    u = ufl.TrialFunction(fe_space)
    v = ufl.TestFunction(fe_space)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(fh, v) * dx

    bilinear_form = dfx.fem.form(a)
    A = assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    linear_form = dfx.fem.form(L)
    b = assemble_vector(linear_form)
    if len(bcs) > 0:
        dfx.fem.apply_lifting(b, [bilinear_form], [bcs])
        dfx.fem.set_bc(b, bcs)
    b.assemble()

    # PETSc solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setType("preonly")
    solver.setOperators(A)
    pc = solver.getPC()
    pc.setType("cholesky")

    solution = dfx.fem.Function(fe_space)
    solver.solve(b, solution.x.petsc_vec)
    solver.destroy()
    return solution


def cell_diameter(dg0_space):
    v0 = ufl.TestFunction(dg0_space)

    vol = v0 * ufl.dx
    vol_form = dfx.fem.form(vol)
    vol_vec = assemble_vector(vol_form)
    size = np.sqrt(vol_vec.array)

    h_T = dfx.fem.Function(dg0_space)
    h_T.x.array[:] = size
    return h_T


def residual_estimation(
    dg0_space,
    solution_u,
    fh,
    gh,
    measures,
    coefs=None,
    phih=None,
    solution_p=None,
    curved=False,
    dirichlet_estimator=False,
):
    dual = solution_p is not None
    dx, dS = (measures["dx"], measures["dS"])

    mesh = solution_u.function_space.mesh
    if curved:
        h_T = cell_diameter(dg0_space)
    else:
        h_T = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    k = solution_u.function_space.element.basix_element.degree
    quadrature_degree_cells = max(0, k - 2)
    quadrature_degree_facets = max(0, k - 1)

    dx_est = dx.reconstruct(
        metadata={"quadrature_degree": quadrature_degree_cells},
    )
    dS_est = dS.reconstruct(
        metadata={"quadrature_degree": quadrature_degree_facets},
    )

    rh = fh + delta(solution_u)
    Jh = ufl.jump(ufl.grad(solution_u), -n)

    w0 = ufl.TestFunction(dg0_space)

    eta_r = h_T**2 * ufl.inner(ufl.inner(rh, rh), w0) * dx_est
    eta_J = ufl.avg(h_T) * ufl.inner(ufl.inner(Jh, Jh), ufl.avg(w0)) * dS_est

    eta_dict = {"eta_r": eta_r, "eta_J": eta_J}

    if dual and dirichlet_estimator:
        if phih is None:
            raise ValueError(
                "You must pass a discrete levelset in order to compute eta_BC."
            )
        pen_coef = coefs["penalization"]
        eta_BC = (
            pen_coef
            * (
                h_T ** (-2)
                * ufl.inner(
                    solution_u - solution_p * phih - gh,
                    solution_u - solution_p * phih - gh,
                )
                * w0
            )
            * dx(2)
        )
        eta_dict["eta_BC"] = eta_BC

    for name, e in eta_dict.items():
        e_form = dfx.fem.form(e)
        e_vec = assemble_vector(e_form)
        e_h = dfx.fem.Function(dg0_space)
        e_h.x.array[:] = e_vec.array[:]
        eta_dict[name] = e_h
    return eta_dict


def marking(est_h, dorfler_param):
    mesh = est_h.function_space.mesh
    cdim = mesh.topology.dim

    est_global = est_h.x.array.sum()
    cutoff = dorfler_param * est_global

    sorted_cells = np.argsort(est_h.x.array)[::-1]
    rolling_sum = 0.0
    breakpt = 0
    for j, e in enumerate(est_h.x.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpt = j
            break

    refine_cells = sorted_cells[0 : breakpt + 1]
    cells_indices = np.array(np.sort(refine_cells), dtype=np.int32)
    fdim = cdim - 1
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
    facets_indices = np.unique(np.sort(c2f_map[cells_indices]))
    return facets_indices, cells_indices


def compute_h10_error(solution_u_2_ref, reference_solution, ref_dg0_space, dg0_space):
    grad_diff = ufl.grad(reference_solution - solution_u_2_ref)
    ref_v0 = ufl.TestFunction(ref_dg0_space)
    h10_norm_diff = (
        ufl.inner(grad_diff, grad_diff)
        * ref_v0
        * ufl.dx(metadata={"quadrature_degree": 20})
    )
    h10_norm_form = dfx.fem.form(h10_norm_diff)
    h10_norm_vec = assemble_vector(h10_norm_form)
    ref_h10_norm = dfx.fem.Function(ref_dg0_space)
    ref_h10_norm.x.array[:] = h10_norm_vec.array[:]

    coarse_h10_norm = None
    try:
        coarse_mesh = dg0_space.mesh
        cdim = coarse_mesh.geometry.dim
        num_cells = coarse_mesh.topology.index_map(cdim).size_global
        all_cells = np.arange(num_cells)
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, ref_dg0_space, all_cells, padding=1.0e-20
        )
        coarse_h10_norm = dfx.fem.Function(dg0_space)
        coarse_h10_norm.interpolate_nonmatching(
            ref_h10_norm, all_cells, nmm_ref_space2coarse_space
        )
    except RuntimeError:
        print("Failed to interpolate h10_norm to coarse space.")
        pass
    return ref_h10_norm, coarse_h10_norm


def compute_l2_error(
    solution_u_2_ref,
    reference_solution,
    ref_dg0_space,
    dg0_space,
    coarse_h_T_2_ref,
    coarse_cut_indicator_2_ref,
):
    ref_error = solution_u_2_ref - reference_solution
    ref_v0 = ufl.TestFunction(ref_dg0_space)

    l2_norm_int = (
        coarse_h_T_2_ref ** (-2)
        * coarse_cut_indicator_2_ref
        * ufl.inner(ref_error, ref_error)
        * ref_v0
        * ufl.dx(metadata={"quadrature_degree": 20})
    )
    l2_norm_form = dfx.fem.form(l2_norm_int)
    l2_err_vec = dfx.fem.assemble_vector(l2_norm_form)

    ref_l2_norm = dfx.fem.Function(ref_dg0_space)
    # We replace eventual NaN values with zero.
    ref_l2_norm.x.array[:] = np.nan_to_num(l2_err_vec.array[:], copy=False, nan=0.0)

    coarse_l2_norm = None
    try:
        coarse_mesh = dg0_space.mesh
        cdim = coarse_mesh.geometry.dim
        num_cells = coarse_mesh.topology.index_map(cdim).size_global
        all_cells = np.arange(num_cells)
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, ref_dg0_space, all_cells, padding=1.0e-20
        )
        coarse_l2_norm = dfx.fem.Function(dg0_space)
        coarse_l2_norm.interpolate_nonmatching(
            ref_l2_norm, all_cells, nmm_ref_space2coarse_space
        )
    except RuntimeError:
        print("Failed to interpolate l2_norm to coarse space.")
        pass
    return ref_l2_norm, coarse_l2_norm


def compute_phi_p_error(
    coarse_solution_p_2_ref,
    ref_solution,
    ref_g,
    ref_dg0_space,
    dg0_space,
    coarse_levelset_2_ref,
    ref_levelset,
    coarse_h_T_2_ref,
    coarse_cut_indicator_2_ref,
):
    ref_p = ref_solution - ref_g
    phi_p_error = ref_p * ref_levelset - coarse_solution_p_2_ref * coarse_levelset_2_ref

    ref_v0 = ufl.TestFunction(ref_dg0_space)

    phi_p_norm_int = (
        coarse_h_T_2_ref ** (-2)
        * coarse_cut_indicator_2_ref
        * ufl.inner(phi_p_error, phi_p_error)
        * ref_v0
        * ufl.dx(metadata={"quadrature_degree": 20})
    )
    phi_p_norm_form = dfx.fem.form(phi_p_norm_int)
    phi_p_err_vec = dfx.fem.assemble_vector(phi_p_norm_form)

    ref_phi_p_norm = dfx.fem.Function(ref_dg0_space)
    # We replace eventual NaN values with zero.
    ref_phi_p_norm.x.array[:] = np.nan_to_num(
        phi_p_err_vec.array[:], copy=False, nan=0.0
    )

    coarse_phi_p_norm = None
    try:
        coarse_mesh = dg0_space.mesh
        cdim = coarse_mesh.geometry.dim
        num_cells = coarse_mesh.topology.index_map(cdim).size_global
        all_cells = np.arange(num_cells)
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, ref_dg0_space, all_cells, padding=1.0e-20
        )
        coarse_phi_p_norm = dfx.fem.Function(dg0_space)
        coarse_phi_p_norm.interpolate_nonmatching(
            ref_phi_p_norm, all_cells, nmm_ref_space2coarse_space
        )
    except RuntimeError:
        print("Failed to interpolate l2_norm to coarse space.")
        pass

    return ref_phi_p_norm, coarse_phi_p_norm


def compute_phi_error(
    coarse_solution_p_2_ref,
    ref_levelset,
    coarse_levelset_2_ref,
    coarse_h_T_2_ref,
    coarse_cut_indicator_2_ref,
    ref_dg0_space,
    dg0_space,
):
    phi_diff = (ref_levelset - coarse_levelset_2_ref) * coarse_solution_p_2_ref

    ref_v0 = ufl.TestFunction(ref_dg0_space)
    l2_norm_phi_int = (
        coarse_h_T_2_ref ** (-2)
        * ufl.inner(phi_diff, phi_diff)
        * coarse_cut_indicator_2_ref
        * ref_v0
        * ufl.dx(metadata={"quadrature_degree": 20})
    )
    l2_norm_phi_form = dfx.fem.form(l2_norm_phi_int)
    l2_phi_err_vec = dfx.fem.assemble_vector(l2_norm_phi_form)

    ref_l2_phi_norm = dfx.fem.Function(ref_dg0_space)
    # We replace eventual NaN values with zero.
    ref_l2_phi_norm.x.array[:] = np.nan_to_num(
        l2_phi_err_vec.array[:], copy=False, nan=0.0
    )

    coarse_l2_phi_norm = None
    try:
        coarse_mesh = dg0_space.mesh
        cdim = coarse_mesh.geometry.dim
        num_cells = coarse_mesh.topology.index_map(cdim).size_global
        all_cells = np.arange(num_cells)
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, ref_dg0_space, all_cells, padding=1.0e-20
        )
        coarse_l2_phi_norm = dfx.fem.Function(dg0_space)
        coarse_l2_phi_norm.interpolate_nonmatching(
            ref_l2_phi_norm, all_cells, nmm_ref_space2coarse_space
        )
    except RuntimeError:
        print("Failed to interpolate l2_norm to coarse space.")
        pass
    return ref_l2_phi_norm, coarse_l2_phi_norm


def save_function(fct, path):
    mesh = fct.function_space.mesh
    fct_element = fct.function_space.element.basix_element
    deg = fct_element.degree
    if deg > 1:
        element_family = fct_element.family.name
        mesh = fct.function_space.mesh
        cg1_element = element(
            element_family,
            mesh.topology.cell_name(),
            1,
            shape=fct.function_space.value_shape,
        )
        cg1_space = dfx.fem.functionspace(mesh, cg1_element)
        cg1_fct = dfx.fem.Function(cg1_space)
        cg1_fct.interpolate(fct)
        with XDMFFile(mesh.comm, path + ".xdmf", "w") as of:
            of.write_mesh(mesh)
            of.write_function(cg1_fct)
    else:
        with XDMFFile(mesh.comm, path + ".xdmf", "w") as of:
            of.write_mesh(mesh)
            of.write_function(fct)
