import argparse
import os

import adios4dolfinx
import dolfinx as dfx
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import MPI
import numpy as np
import polars as pl
import yaml
from meshtagsplot import plot_mesh_tags

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog="plot.py", description="Plot the demo results.")

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

args = parser.parse_args()
demo, parameters = args.parameters.split(sep="/")

fig = plt.figure()
ax = fig.subplots()
parameter_names = []

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

iterations_num = parameters["iterations_number"]

source_dir = os.path.join(parent_dir, demo)
output_dir = os.path.join(source_dir, "output_" + parameters)
meshes_dir = os.path.join(output_dir, "meshes")

for i in range(iterations_num):
    mesh = adios4dolfinx.read_mesh(
        os.path.join(
            output_dir,
            "checkpoints",
            f"checkpoint_{str(i).zfill(2)}.bp",
        ),
        comm=MPI.COMM_WORLD,
    )

    cdim = mesh.geometry.dim()
    num_cells = mesh.topology.index_map(cdim).size_global
    all_cells = np.arange(num_cells)
    tags = np.zeros_like(all_cells)
    dummy_mesh_tags = dfx.mesh.meshtags(mesh, cdim, all_cells, tags)

    fig = plt.figure()
    ax = fig.subplots()

    plot_mesh_tags(mesh, dummy_mesh_tags, ax, display_scalarbar=False)
    plt.savefig(
        os.path.join(meshes_dir, f"mesh_{str(i)}.zfill(2).png"),
        dpi=500,
        bbox_inches="tight",
    )
