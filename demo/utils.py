import adios4dolfinx
import dolfinx as dfx
import numpy as np
import ufl
from basix.ufl import element
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.fem.petsc import assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI


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


def save_function(fct: dfx.fem.Function, file_path: str):
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
        with XDMFFile(mesh.comm, file_path, "w") as of:
            of.write_mesh(mesh)
            of.write_function(cg1_fct)
    else:
        with XDMFFile(mesh.comm, file_path, "w") as of:
            of.write_mesh(mesh)
            of.write_function(fct)


def compute_boundary_local_estimators(
    coarse_mesh, solution_w, levelset, phih, cells_tags, facets_tags, padding=1.0e-14
):
    facets_to_refine = np.union1d(facets_tags.find(2), facets_tags.find(3))
    facets_to_refine = np.union1d(facets_to_refine, facets_tags.find(4))

    cdim = coarse_mesh.topology.dim
    num_cells = coarse_mesh.topology.index_map(cdim).size_global
    dummy_mesh = dfx.mesh.create_submesh(coarse_mesh, cdim, np.arange(num_cells))[0]

    # Mark the cells to refine in the coarse mesh
    dummy_mesh.topology.create_entities(dummy_mesh.topology.dim - 1)
    fdim = cdim - 1
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
    # WARNING: child-parent cells map is not correctly computed from refine! To be able to use it you have to use the following trick, inspired by https://fenicsproject.discourse.group/t/mesh-refinement-using-dolfinx-mesh-refine-plaza/15168/4
    fine_cells_tags = dfx.mesh.transfer_meshtag(cells_tags, fine_mesh, parent_cells)
    num_cells_dummy = dummy_mesh.topology.index_map(cdim).size_global
    parent_ct_indices = np.arange(num_cells_dummy).astype(np.int32)
    parent_ct_markers = parent_ct_indices
    sorted_indices = np.argsort(parent_ct_indices)
    parent_ct = dfx.mesh.meshtags(
        fine_mesh,
        cdim,
        parent_ct_indices[sorted_indices],
        parent_ct_markers[sorted_indices],
    )
    parent_cells = dfx.mesh.transfer_meshtag(parent_ct, fine_mesh, parent_cells).values

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
        fine_space, solution_w.function_space, fine_cells, padding=padding
    )
    solution_w_fine = dfx.fem.Function(fine_space)
    solution_w_fine.interpolate_nonmatching(
        solution_w, fine_cells, nmm_solution_w_space2fine_space
    )
    correction_function_fine = dfx.fem.Function(fine_space)
    correction_function_fine.x.array[:] = (
        phih_fine.x.array[:] - phi_fine.x.array[:]
    ) * solution_w_fine.x.array[:]

    cell_name = fine_mesh.topology.cell_name()
    dg0_element = element("DG", cell_name, 0)
    dg0_fine_space = dfx.fem.functionspace(fine_mesh, dg0_element)

    dx = ufl.Measure("dx", domain=fine_mesh, subdomain_data=fine_cells_tags)
    v0 = ufl.TestFunction(dg0_fine_space)
    grad_correction = ufl.grad(correction_function_fine)
    h10_norm_correction = ufl.inner(grad_correction, grad_correction) * v0 * dx(2)
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
    return h10_norm_dg0
