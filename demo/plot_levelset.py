import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import yaml
from plots import plot_mesh

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="plot_levelset.py",
    description="Plot the levelset from the given demo.",
)

parser.add_argument("parameter", type=str, help="Choose the demo/parameters to run.")

args = parser.parse_args()
parameter = args.parameter
demo, param_name = parameter.split(sep="/")

source_dir = os.path.join(parent_dir, demo)
sys.path.append(source_dir)

from data import gen_mesh, generate_levelset

levelset = generate_levelset(np)

with open(os.path.join(source_dir, param_name + ".yaml"), "rb") as f:
    parameters = yaml.safe_load(f)

bbox = parameters["bbox"]

xrange = np.abs(bbox[0][1] - bbox[0][0])
yrange = np.abs(bbox[1][1] - bbox[1][0])
maxrange = np.maximum(xrange, yrange)

x = np.linspace(bbox[0][0], bbox[0][0] + maxrange, 800)
y = np.linspace(bbox[1][0], bbox[1][0] + maxrange, 800)
X, Y = np.meshgrid(x, y)
XY = np.vstack([np.ndarray.flatten(X), np.ndarray.flatten(Y)])
z = levelset(XY)
Z = z.reshape(X.shape)


fig = plt.figure()
ax = fig.subplots()
ax.contourf(X, Y, Z, levels=[-10, 0], colors=["lightblue"])
ax.contour(X, Y, Z, levels=[0], colors="black", linewidths=1.25)

ax.set_xlim([bbox[0][0], bbox[0][1]])
ax.set_ylim([bbox[1][0], bbox[1][1]])
plt.gca().set_aspect("equal")
plt.savefig(os.path.join(demo, "levelset"), dpi=500, bbox_inches="tight")

fig = go.Figure(
    data=[
        go.Surface(
            contours={"z": {"show": True, "start": -0.001, "end": 0.0, "size": 0.001}},
            z=Z,
            x=X,
            y=Y,
        )
    ]
)
fig.update_layout(
    title=dict(text=f"Levelset ({demo})"),
    autosize=False,
    scene={"aspectratio": {"x": 1, "y": 1, "z": 1}},
    width=1000,
    height=1000,
)

fig.write_html(os.path.join(demo, "levelset.html"))

mesh, geoModel = gen_mesh(0.05, curved=True)
plot_mesh(mesh, "mesh", wireframe=True, linewidth=1.5)
