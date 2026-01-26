import argparse
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog="plot.py", description="Plot the demo results.")

parser.add_argument("parameters", type=str, help="Choose the demo/parameters to run.")

parser.add_argument(
    "data",
    type=str,
    help="List of quantities to plot.",
)

args = parser.parse_args()
parameters_list = args.parameters.split(sep=",")
data_list = args.data.split(sep=",")

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

fig = plt.figure()
ax = fig.subplots()
parameter_names = []
for parameter in parameters_list:
    demo, param_name = parameter.split(sep="/")
    if "phifem" in param_name:
        style = "-"
        color = "#d7191c"
    else:
        style = "--"
        color = "#2c7bb6"
    scheme_name = plot_param[param_name]["name"].split(sep=",")[0]
    parameter_names.append(param_name)
    with open(os.path.join(parameter + ".yaml"), "rb") as f:
        parameters = yaml.safe_load(f)

    data_path = os.path.join(
        demo,
        "output_" + param_name,
        "results.csv",
    )
    df = pl.read_csv(data_path)
    if "phifem" in param_name:
        error2comp = ["h10_error", "triple_norm_error"]
        est = "\\eta"
        err_names = ["|e|_{1,\\Omega}", "E"]
        markers = ["^", "o"]
    else:
        error2comp = ["h10_error"]
        err_names = ["|e|_{1,\\Omega}"]
        est = "\\eta_1"
        markers = ["^"]
    for err, err_name, marker in zip(error2comp, err_names, markers):
        xs = df["dof"].to_numpy()
        ys = df["estimator"].to_numpy() / df[err].to_numpy()
        mask = np.isnan(ys)
        xs = xs[~mask]
        ys = ys[~mask]
        plt.semilogx(
            xs,
            ys,
            style + marker,
            color=color,
            label=f"${est}/{err_name}$, " + f"{scheme_name}",
            path_effects=[
                pe.Stroke(linewidth=2.5, foreground="#252525"),
                pe.Normal(),
            ],
        )

plt.xlabel("dof")
plt.legend()
plt.savefig(
    os.path.join(demo, "plot_efficiency_" + "-".join(parameter_names)),
    dpi=500,
    bbox_inches="tight",
)
