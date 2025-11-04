import dolfinx as dfx
import numpy as np
import ufl
from basix.ufl import element
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc


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


def save_function(fct: dfx.fem.Function, file_path: str, mode=dfx.io.XDMFFile):
    """Save a dolfinx function using XDMFFile and interpolate it to a linear space if needed.

    Args:
        fct: the dolfinx Function to save.
        file_path: the path where the xdmf file is saved.
    """
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
        with mode(mesh.comm, file_path, "w") as of:
            of.write_mesh(mesh)
            of.write_function(cg1_fct)
    else:
        with mode(mesh.comm, file_path, "w") as of:
            of.write_mesh(mesh)
            of.write_function(fct)


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
    facets_tags,
    dual=False,
    padding=1.0e-14,
):
    facets_to_refine = np.union1d(facets_tags.find(2), facets_tags.find(3))
    facets_to_refine = np.union1d(facets_to_refine, facets_tags.find(4))

    cdim = coarse_mesh.topology.dim
    num_cells = coarse_mesh.topology.index_map(cdim).size_global
    dummy_mesh = dfx.mesh.create_submesh(coarse_mesh, cdim, np.arange(num_cells))[0]

    # Mark the cells to refine in the coarse mesh
    dummy_mesh.topology.create_entities(dummy_mesh.topology.dim - 1)
    fdim = cdim - 1
    coarse_mesh.topology.create_connectivity(fdim, cdim)
    f2c_connect_dummy = coarse_mesh.topology.connectivity(fdim, cdim)
    f2c_map_dummy = _reshape_facets_map(f2c_connect_dummy)
    cells_to_refine = f2c_map_dummy[facets_to_refine]
    cells_to_refine = np.unique(cells_to_refine.reshape(-1))

    # Mark the corresponding child cells of the refined cells in the fine mesh
    fine_mesh, parent_cells, _ = dfx.mesh.refine(
        dummy_mesh,
        facets_to_refine,
        option=RefinementOption.parent_cell,
    )
    parent_cells = compute_parent_cells(dummy_mesh, fine_mesh, parent_cells)
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
    dg0_fine_space = dfx.fem.functionspace(fine_mesh, dg0_element)

    dx = ufl.Measure("dx", domain=fine_mesh, subdomain_data=fine_cells_tags)
    v0 = ufl.TestFunction(dg0_fine_space)

    dg0_coarse_space = dfx.fem.functionspace(coarse_mesh, dg0_element)
    h_T_coarse = cell_diameter(dg0_coarse_space)
    h_T_fine = dfx.fem.Function(dg0_fine_space)
    nmm_dg0_coarse_space2dg0_fine_space = dfx.fem.create_interpolation_data(
        dg0_fine_space, dg0_coarse_space, fine_cells, padding=padding
    )
    h_T_fine.interpolate_nonmatching(
        h_T_coarse, fine_cells, nmm_dg0_coarse_space2dg0_fine_space
    )
    correction_function_fine = dfx.fem.Function(fine_space)
    correction_function_fine.x.array[:] = (
        phih_fine.x.array[:] - phi_fine.x.array[:]
    ) * solution_p_fine.x.array[:]
    correction_function_fine = correction_function_fine * h_T_fine ** (-1)
    grad_correction = ufl.grad(correction_function_fine)
    if dual:
        measure_ind = 2
    else:
        measure_ind = (1, 2)

    # eta_{1,z}
    h10_norm_correction = (
        ufl.inner(grad_correction, grad_correction) * v0 * dx(measure_ind)
    )
    h10_norm_correction_form = dfx.fem.form(h10_norm_correction)
    h10_norm_correction_vec = assemble_vector(h10_norm_correction_form)

    h10_norm_dg0_fine = dfx.fem.Function(dg0_fine_space)
    h10_norm_dg0_fine.x.array[:] = h10_norm_correction_vec.array[:]
    h10_norm_vec_coarse = np.bincount(
        parent_cells, weights=h10_norm_correction_vec.array[:]
    )
    dg0_coarse_space = dfx.fem.functionspace(coarse_mesh, dg0_element)
    h10_norm_dg0 = dfx.fem.Function(dg0_coarse_space)
    h10_norm_dg0.x.array[:] = h10_norm_vec_coarse

    # eta_{0,z}
    l2_norm_correction = (
        h_T_fine ** (-2)
        * ufl.inner(correction_function_fine, correction_function_fine)
        * v0
        * dx(measure_ind)
    )
    l2_norm_correction_form = dfx.fem.form(l2_norm_correction)
    l2_norm_correction_vec = assemble_vector(l2_norm_correction_form)

    l2_norm_dg0_fine = dfx.fem.Function(dg0_fine_space)
    l2_norm_dg0_fine.x.array[:] = l2_norm_correction_vec.array[:]
    l2_norm_vec_coarse = np.bincount(
        parent_cells, weights=l2_norm_correction_vec.array[:]
    )
    dg0_coarse_space = dfx.fem.functionspace(coarse_mesh, dg0_element)

    dg0_coarse_space = dfx.fem.functionspace(coarse_mesh, dg0_element)
    l2_norm_dg0 = dfx.fem.Function(dg0_coarse_space)
    l2_norm_dg0.x.array[:] = l2_norm_vec_coarse

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
    return h10_norm_dg0, l2_norm_dg0, fine_submesh_tags, fine_submesh


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

    penalization = (
        pen_coef
        * h_T ** (-2)
        * ufl.inner(uh - h_T ** (-1) * ph * phih, vh - h_T ** (-1) * qh * phih)
    )

    a = (
        stiffness * dx((1, 2))
        - boundary * ds
        + penalization * dx(2)
        + stabilization_cells * dx(2)
        + stabilization_facets * dS(2)
    )

    # Linear form
    rhs = ufl.inner(fh, vh)
    penalization_rhs = (
        pen_coef * h_T ** (-2) * ufl.inner(gh, vh - h_T ** (-1) * qh * phih)
    )
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

    eta_T = h_T**2 * ufl.inner(ufl.inner(rh, rh), w0) * dx_est
    eta_E = ufl.avg(h_T) * ufl.inner(ufl.inner(Jh, Jh), ufl.avg(w0)) * dS_est

    eta_dict = {"eta_T": eta_T, "eta_E": eta_E}

    if dual and dirichlet_estimator:
        if phih is None:
            raise ValueError(
                "You must pass a discrete levelset in order to compute eta_p."
            )
        pen_coef = coefs["penalization"]
        eta_p = (
            pen_coef
            * (
                h_T ** (-2)
                * ufl.inner(
                    solution_u - h_T ** (-1) * solution_p * phih - gh,
                    solution_u - h_T ** (-1) * solution_p * phih - gh,
                )
                * w0
            )
            * dx(2)
        )
        eta_dict["eta_p"] = eta_p

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


def compute_xi_h10_l2(solution_u_ref, g_ref, coarse_h_T, ref_dg0_space):
    ref_space = solution_u_ref.function_space
    xi_source_term = dfx.fem.Function(ref_space)
    xi_dirichlet_data = dfx.fem.Function(ref_space)
    xi_dirichlet_data.x.array[:] = solution_u_ref.x.array[:] - g_ref.x.array[:]
    xi_ref = fem_solve(solution_u_ref.function_space, xi_source_term, xi_dirichlet_data)

    ref_v0 = ufl.TestFunction(ref_dg0_space)
    xi_ref_h10_int = ufl.inner(ufl.grad(xi_ref), ufl.grad(xi_ref)) * ref_v0 * ufl.dx
    xi_ref_h10_form = dfx.fem.form(xi_ref_h10_int)
    xi_ref_h10_vec = assemble_vector(xi_ref_h10_form)
    xi_ref_h10 = dfx.fem.Function(ref_dg0_space)
    xi_ref_h10.x.array[:] = xi_ref_h10_vec.array[:]

    xi_ref_l2_int = coarse_h_T ** (-1) * ufl.inner(xi_ref, xi_ref) * ref_v0 * ufl.dx
    xi_ref_l2_form = dfx.fem.form(xi_ref_l2_int)
    xi_ref_l2_vec = assemble_vector(xi_ref_l2_form)
    xi_ref_l2 = dfx.fem.Function(ref_dg0_space)
    xi_ref_l2.x.array[:] = xi_ref_l2_vec.array[:]
    return xi_ref_h10, xi_ref_l2


def compute_h10_error(solution_u_2_ref, reference_solution, ref_dg0_space, dg0_space):
    reference_space = reference_solution.function_space
    solution_u_2_ref = dfx.fem.Function(reference_space)

    grad_diff = ufl.grad(reference_solution - solution_u_2_ref)
    ref_v0 = ufl.TestFunction(ref_dg0_space)
    h10_norm_diff = ufl.inner(grad_diff, grad_diff) * ref_v0 * ufl.dx
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


def compute_l2_error_p(
    coarse_solution_p_2_ref,
    ref_solution,
    ref_g,
    coarse_levelset_2_ref,
    coarse_h_T_2_ref,
    coarse_cut_indicator_2_ref,
    ref_dg0_space,
):
    ref_space = ref_solution.function_space
    h_T_reference_p = dfx.fem.Function(ref_space)
    h_T_reference_p = ref_solution - ref_g

    p_diff = (
        h_T_reference_p
        - coarse_solution_p_2_ref * coarse_levelset_2_ref / coarse_h_T_2_ref
    )

    ref_v0 = ufl.TestFunction(ref_dg0_space)
    l2_norm_p_int = (
        coarse_h_T_2_ref ** (-2)
        * ufl.inner(p_diff, p_diff)
        * coarse_cut_indicator_2_ref
        * ref_v0
        * ufl.dx
    )
    l2_norm_p_form = dfx.fem.form(l2_norm_p_int)
    l2_p_err_vec = dfx.fem.assemble_vector(l2_norm_p_form)

    ref_l2_p_err = dfx.fem.Function(ref_dg0_space)
    ref_l2_p_err.x.array[:] = l2_p_err_vec.array[:]
    return ref_l2_p_err


def compute_source_term_oscillations(fh, ref_f, coarse_h_T, ref_dg0_space, dg0_space):
    coarse_space = fh.function_space
    coarse_mesh = coarse_space.mesh
    reference_space = ref_f.function_space
    reference_mesh = reference_space.mesh
    ref_fh = dfx.fem.Function(reference_space)

    cdim = reference_mesh.topology.dim
    num_reference_cells = reference_mesh.topology.index_map(cdim).size_global
    reference_cells = np.arange(num_reference_cells)
    nmm_coarse2ref = dfx.fem.create_interpolation_data(
        reference_space, coarse_space, reference_cells, padding=1.0e-14
    )

    ref_fh.interpolate_nonmatching(fh, reference_cells, nmm_coarse2ref)

    diff = ref_f - ref_fh
    ref_v0 = ufl.TestFunction(ref_dg0_space)
    st_osc_norm_diff = coarse_h_T**2 * ufl.inner(diff, diff) * ref_v0 * ufl.dx
    st_osc_norm_form = dfx.fem.form(st_osc_norm_diff)
    st_osc_norm_vec = assemble_vector(st_osc_norm_form)
    ref_st_osc_norm = dfx.fem.Function(ref_dg0_space)
    ref_st_osc_norm.x.array[:] = st_osc_norm_vec.array[:]

    coarse_st_osc_norm = None
    try:
        num_cells = coarse_mesh.topology.index_map(cdim).size_global
        all_cells = np.arange(num_cells)
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, ref_dg0_space, all_cells, padding=1.0e-20
        )
        coarse_st_osc_norm = dfx.fem.Function(dg0_space)
        coarse_st_osc_norm.interpolate_nonmatching(
            ref_st_osc_norm, all_cells, nmm_ref_space2coarse_space
        )
    except RuntimeError:
        print("Failed to interpolate h10_norm to coarse space.")
        pass
    return ref_st_osc_norm, coarse_st_osc_norm


def compute_dirichlet_oscillations(
    gh, ref_g, ref_dg0_space, dg0_space, ref_cut_indicator
):
    coarse_space = gh.function_space
    coarse_mesh = coarse_space.mesh
    reference_space = ref_g.function_space
    reference_mesh = reference_space.mesh
    ref_gh = dfx.fem.Function(reference_space)

    cdim = reference_mesh.topology.dim
    num_reference_cells = reference_mesh.topology.index_map(cdim).size_global
    reference_cells = np.arange(num_reference_cells)
    nmm_coarse2ref = dfx.fem.create_interpolation_data(
        reference_space, coarse_space, reference_cells, padding=1.0e-14
    )

    ref_gh.interpolate_nonmatching(gh, reference_cells, nmm_coarse2ref)

    diff = ufl.grad(ref_g - ref_gh)
    ref_v0 = ufl.TestFunction(ref_dg0_space)
    dir_osc_norm_diff = ref_cut_indicator * ufl.inner(diff, diff) * ref_v0 * ufl.dx
    dir_osc_norm_form = dfx.fem.form(dir_osc_norm_diff)
    dir_osc_norm_vec = assemble_vector(dir_osc_norm_form)
    ref_dir_osc_norm = dfx.fem.Function(ref_dg0_space)
    ref_dir_osc_norm.x.array[:] = dir_osc_norm_vec.array[:]

    coarse_dir_osc_norm = None
    try:
        num_cells = coarse_mesh.topology.index_map(cdim).size_global
        all_cells = np.arange(num_cells)
        nmm_ref_space2coarse_space = dfx.fem.create_interpolation_data(
            dg0_space, ref_dg0_space, all_cells, padding=1.0e-20
        )
        coarse_dir_osc_norm = dfx.fem.Function(dg0_space)
        coarse_dir_osc_norm.interpolate_nonmatching(
            ref_dir_osc_norm, all_cells, nmm_ref_space2coarse_space
        )
    except RuntimeError:
        print("Failed to interpolate h10_norm to coarse space.")
        pass
    return ref_dir_osc_norm, coarse_dir_osc_norm
