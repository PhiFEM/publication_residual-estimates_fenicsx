import dolfinx as dfx
from basix.ufl import element
from dolfinx.io import XDMFFile


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
