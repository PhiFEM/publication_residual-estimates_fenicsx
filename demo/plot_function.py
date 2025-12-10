import argparse
import os
import sys

import adios4dolfinx
import dolfinx as dfx
import numpy as np
import plotly.graph_objects as go
from basix.ufl import element
from dolfinx.io import XDMFFile
from mpi4py import MPI

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="plot_function.py", description="Plot a function as a 3D surface."
)

parser.add_argument("path", type=str, help="Choose the demo/parameters to run.")
parser.add_argument("name", type=str, help="Name of the function to plot.")
parser.add_argument(
    "element",
    type=str,
    choices=["cg1", "cg2", "dg0", "dg1"],
    help="The finite element of the function to plot.",
)

args = parser.parse_args()
path = args.path
fct_name = args.name
fct_dir = os.path.join(parent_dir, path)

mesh = adios4dolfinx.read_mesh(fct_dir, comm=MPI.COMM_WORLD)

demo_name = path.split(sep="/")[0]
source_dir = os.path.join(parent_dir, demo_name)
param_name = path.split(sep="/")[1].split(sep="_")[1]

sys.path.append(source_dir)

from data import generate_levelset

levelset = generate_levelset(np)

cell_name = mesh.topology.cell_name()
elements_dct = {
    "cg1": element("Lagrange", cell_name, 1),
    "cg2": element("Lagrange", cell_name, 2),
    "dg0": element("DG", cell_name, 0),
    "dg1": element("DG", cell_name, 1),
}

fct_element = elements_dct[args.element]

fct_space = dfx.fem.functionspace(mesh, fct_element)
fct = dfx.fem.Function(fct_space)

adios4dolfinx.read_function(
    path,
    fct,
    name=fct_name,
)

dofs = fct_space.tabulate_dof_coordinates()[:, :2]

x_fct, y_fct = dofs[:, 0], dofs[:, 1]
x_min, x_max = np.min(x_fct), np.max(x_fct)
y_min, y_max = np.min(y_fct), np.max(y_fct)

z_fct = fct.x.array[:]

z_fct_min, z_fct_max = np.min(fct.x.array[:]), np.max(fct.x.array[:])

xrange = np.abs(x_max - x_min)
yrange = np.abs(y_max - y_min)
maxrange = np.maximum(xrange, yrange)
x = np.linspace(x_min - 0.001, x_max + 0.001, 1000)
y = np.linspace(y_min - 0.001, y_max + 0.001, 1000)
X, Y = np.meshgrid(x, y)
XY = np.vstack([np.ndarray.flatten(X), np.ndarray.flatten(Y)])
z = levelset(XY)

z = np.minimum(z, 0.001)
Z = z.reshape(X.shape)

font_size = 18
fig = go.Figure()

fig.add_trace(
    go.Mesh3d(
        x=x_fct,
        y=y_fct,
        z=z_fct,
        intensity=z_fct,
        colorscale="Viridis",
        opacity=0.85,
        colorbar=dict(tickfont=dict(size=font_size + 5), len=0.5),
    )
)


if "solution" in fct_name:
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            hidesurface=True,
            contours_z=dict(
                size=0,
                width=10,
                show=True,
                color="#fc8d59",
            ),
        ),
    )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200,
        font=dict(size=font_size),
        scene_aspectratio=dict(x=1, y=1, z=0.5),
    )
fig.update_scenes(zaxis=dict(range=[z_fct_min - 0.001, z_fct_max]))

output_name = demo_name + "_" + param_name + "_" + fct_name
fig.write_html(os.path.join(parent_dir, output_name + ".html"))

with XDMFFile(mesh.comm, os.path.join(parent_dir, output_name + ".xdmf"), "w") as of:
    of.write_mesh(mesh)
    of.write_function(fct)
