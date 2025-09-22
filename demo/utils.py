import adios4dolfinx
import dolfinx as dfx
import numpy as np
from basix.ufl import element
from dolfinx.io import XDMFFile
from mpi4py import MPI


def read_function(filepath, element_family, degree, timestamp, name):
    mesh = adios4dolfinx.read_mesh(filepath, comm=MPI.COMM_WORLD, time=timestamp)
    space = dfx.fem.functionspace(mesh, (element_family, degree))
    fct = dfx.fem.Function(space)
    adios4dolfinx.read_function(filepath, fct, time=timestamp, name=name)
    return fct


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


def compute_boundary_correction(mesh, levelset, phih, cells_tags, facets_tags):
    cut_cells = cells_tags.find(2)
    cut_facets = facets_tags.find(2)
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_global
    dummy_mesh = dfx.mesh.create_submesh(mesh, tdim, np.arange(num_cells))[0]
    dummy_mesh.topology.create_entities(dummy_mesh.topology.dim - 1)
    fine_mesh = dfx.mesh.refine(dummy_mesh, cut_facets)[0]

    phih_space = phih.function_space
    print(dir(phih.function_space.ufl_element()))
    fine_element = phih_space.ufl_element()
    fine_space = dfx.fem.functionspace(fine_mesh, fine_element)
    dg0_fine_element = element("DG", mesh.topology.cell_name(), 0)
    dg0_fine_space = dfx.fem.functionspace(fine_mesh, dg0_fine_element)

    nmm_mesh2fine_mesh = dfx.fem.create_interpolation_data(
        fine_space, phih_space, mesh, padding=1.0e-14
    )
    phih_fine = dfx.fem.Function(fine_space)
    phih_fine.interpolate(phih, nmm_interpolation_data=nmm_mesh2fine_mesh)
