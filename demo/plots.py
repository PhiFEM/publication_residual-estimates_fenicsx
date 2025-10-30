import os

import dolfinx as dfx
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap


# Labelled and categorical scalar bars is currently not supported for vtk meshes https://github.com/pyvista/pyvista/issues/5389
def plot_tags(mesh, meshtags, name, line_width=1.0, annotations={}):
    plotter = pv.Plotter()

    dim = meshtags.dim
    mesh.topology.create_connectivity(dim, mesh.topology.dim)
    cells, types, x = dfx.plot.vtk_mesh(mesh, dim)
    grid = pv.UnstructuredGrid(cells, types, x)

    title = "Cells tags"
    show_edges = True
    newcolors = [
        "#2166ac",  # Blue
        "#ef8a62",  # Orange
        "#b2182b",  # Red
    ]

    if meshtags.dim == mesh.topology.dim - 1:
        title = "Facets tags"
        show_edges = False
        newcolors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
    elif meshtags.dim < mesh.topology.dim - 1:
        raise ValueError(
            "Meshtags can only be of same dimension as the mesh or one dimension lower."
        )

    unique_values = np.unique(meshtags.values) - 1
    my_cmap = ListedColormap(np.asarray(newcolors)[unique_values])
    grid.cell_data["Marker"] = meshtags.values
    sargs = dict(
        title=title,
        fmt="%.3E",
    )
    plotter.add_mesh(
        grid,
        scalars="Marker",
        show_edges=show_edges,
        line_width=line_width,
        scalar_bar_args=sargs,
        cmap=my_cmap,
        show_scalar_bar=False,
    )
    legend = [
        [val, color, pv.Box()]
        for val, color in zip(
            np.asarray(list(annotations.values()))[unique_values],
            np.asarray(my_cmap.colors),
        )
    ]
    plotter.add_legend(legend)
    plotter.view_xy()
    plotter.export_html(name + ".html")
    plotter.save_graphic(name + ".svg")

    plotter.close()


def plot_mesh(mesh, name, wireframe=False, linewidth=1.0):
    plotter = pv.Plotter()

    grid = pv.UnstructuredGrid(*dfx.plot.vtk_mesh(mesh))
    if wireframe:
        plotter.add_mesh(
            grid,
            style="wireframe",
            show_edges=False,
            show_scalar_bar=False,
            line_width=linewidth,
            color="black",
        )
    else:
        plotter.add_mesh(
            grid,
            show_edges=True,
            show_scalar_bar=False,
            line_width=linewidth,
        )
    plotter.view_xy()
    plotter.export_html(name + ".html")
    plotter.save_graphic(name + ".svg")

    plotter.close()


def plot_scalar(fct, name, warp_by_scalar=False):
    path = os.path.split(name)[:-1][0]
    file_name = os.path.split(name)[-1]
    fct_space = fct.function_space

    fct_degree = fct_space.ufl_element().basix_element.degree

    if fct_degree == 0:
        mesh = fct_space.mesh
        cells, types, x = dfx.plot.vtk_mesh(mesh)
    else:
        cells, types, x = dfx.plot.vtk_mesh(fct_space)

    grid = pv.UnstructuredGrid(cells, types, x)
    if fct_degree == 0:
        grid.cell_data[file_name] = fct.x.array
    else:
        grid.point_data[file_name] = fct.x.array

    grid.set_active_scalars(file_name)

    plotter = pv.Plotter()
    if not warp_by_scalar:
        plotter.add_text(
            "Scalar contour field", font_size=14, color="black", position="upper_edge"
        )
        plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
        plotter.view_xy()
        plotter.enable_rubber_band_2d_style()
        plotter.add_legend_scale()
        plotter.export_html(name + ".html")
    else:
        fct_max = np.max(np.abs(fct.x.array))
        warped = grid.warp_by_scalar(factor=1.0 / fct_max)
        plotter.add_text(
            "Warped function", position="upper_edge", font_size=14, color="black"
        )
        sargs = dict(
            height=0.8,
            width=0.1,
            vertical=True,
            position_x=0.05,
            position_y=0.05,
            title_font_size=40,
            color="black",
            label_font_size=25,
            interactive=True,
        )
        plotter.add_mesh(warped, show_edges=False, scalar_bar_args=sargs)
        plotter.add_axes()
        plotter.export_html(os.path.join(path, "wbs_" + file_name + ".html"))

    plotter.save_graphic(os.path.join(path, file_name + ".svg"))

    plotter.close()


def write_frame(mesh, plotter, uh, name):
    # Update plot with refined mesh
    grid = pv.UnstructuredGrid(*dfx.plot.vtk_mesh(mesh))
    curved_grid = pv.UnstructuredGrid(*dfx.plot.vtk_mesh(uh.function_space))
    curved_grid.point_data["u"] = uh.x.array
    # curved_grid = curved_grid.tessellate()
    curved_actor = plotter.add_mesh(
        curved_grid,
        show_edges=True,
        line_width=2.0,
    )
    actor = plotter.add_mesh(grid, style="wireframe", color="black")
    plotter.view_xy()
    plotter.export_html(name)
    plotter.remove_actor(actor)
    plotter.remove_actor(curved_actor)
