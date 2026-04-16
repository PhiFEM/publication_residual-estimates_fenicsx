import argparse
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="plot_efficiency.py", description="Plot the demo results."
)

parser.add_argument("demo", type=str, help="Choose the demo to plot from.")

args = parser.parse_args()
demo = args.demo
parameters_list = ["fem", "phifem-bc-geo"]

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

data_path = os.path.join(
    demo,
    "output_fem",
    "results.csv",
)
df_fem = pl.read_csv(data_path)

xs_fem = df_fem["dof"].to_numpy()
ys_fem = df_fem["estimator"].to_numpy() / df_fem["h10_error"].to_numpy()
mask = np.isnan(ys_fem)
xs_fem = xs_fem[~mask]
ys_fem = ys_fem[~mask]

data_path = os.path.join(
    demo,
    "output_phifem-bc-geo",
    "results.csv",
)
df_phifem = pl.read_csv(data_path)

xs_phifem = df_phifem["dof"].to_numpy()
ys_phifem = (
    df_phifem["estimator"].to_numpy() / df_phifem["triple_norm_error"].to_numpy()
)
mask = np.isnan(ys_phifem)
xs_phifem = xs_phifem[~mask]
ys_phifem = ys_phifem[~mask]

min_xs = min(max(xs_fem), max(xs_phifem))
mask_fem = xs_fem <= min_xs
mask_phifem = xs_phifem <= min_xs
xs_fem = xs_fem[mask_fem]
ys_fem = ys_fem[mask_fem]
xs_phifem = xs_phifem[mask_phifem]
ys_phifem = ys_phifem[mask_phifem]

fig = plt.figure()
ax = fig.subplots()

style = "--"
color = "#2c7bb6"
est = plot_param["fem"]["estimator_name"]
err = "|e|_{{1,\\Omega}}"

ax.semilogx(
    xs_fem,
    ys_fem,
    style + "^",
    color=color,
    label=f"{est}$/{err}$, " + "FEM",
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground="#252525"),
        pe.Normal(),
    ],
)

est = plot_param["phifem-bc-geo"]["estimator_name"]
err = "E"
err_name = "triple_norm_error"
style = "-"
color = "#d7191c"

ax.semilogx(
    xs_phifem,
    ys_phifem,
    style + "^",
    color=color,
    label=f"{est}$/{err}$, " + "$\\varphi$-FEM",
    path_effects=[
        pe.Stroke(linewidth=2.5, foreground="#252525"),
        pe.Normal(),
    ],
)

plt.xlabel("dof")
plt.legend()
plt.savefig(
    os.path.join(demo, "plot_efficiency.png"),
    dpi=500,
    bbox_inches="tight",
)
