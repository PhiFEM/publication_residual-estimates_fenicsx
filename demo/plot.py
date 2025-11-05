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

fig = plt.figure()
ax = fig.subplots()
parameter_names = []

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

for parameter in parameters_list:
    demo, param_name = parameter.split(sep="/")

    mstyle = plot_param[param_name]["marker"]
    parameter_names.append(param_name)
    with open(os.path.join(parameter + ".yaml"), "rb") as f:
        parameters = yaml.safe_load(f)

    ref_type = parameters["refinement"]
    mesh_type = parameters["mesh_type"]
    scheme_name = plot_param[param_name]["name"]
    if ref_type == "unif":
        trunc = -3
    elif ref_type == "adap":
        trunc = -10
    data_path = os.path.join(
        demo,
        "output_" + param_name,
        "results.csv",
    )

    df = pl.read_csv(data_path)
    for d in data_list:
        d_name = plot_param[d]["name"]
        lstyle = plot_param[d]["line"]
        color = plot_param[d]["color"]
        xs = df["dof"].to_numpy()
        try:
            ys = df[d].to_numpy()
        except pl.exceptions.ColumnNotFoundError:
            print(f"{d} not found.")
            continue
        mask = np.isnan(ys)
        xs = xs[~mask]
        ys = ys[~mask]
        ys_not_zero = np.max(ys) > 0.0
        if ys_not_zero:
            slope, b = np.polyfit(
                np.log(xs[trunc:]),
                np.log(ys[trunc:]),
                1,
            )
        else:
            continue
        if ys_not_zero:
            plt.loglog(
                xs,
                ys,
                lstyle + mstyle,
                c=color,
                label=rf"{d_name} ({scheme_name}; {np.round(slope, 2)})",
                path_effects=[
                    pe.Stroke(linewidth=2.5, foreground="#252525"),
                    pe.Normal(),
                ],
            )
            plt.loglog(
                xs[trunc:],
                np.exp(b) * xs[trunc:] ** slope,
                "--",
                color="k",
            )
        else:
            continue

plt.xlabel("dof")
plt.legend(prop={"size": plot_param["legend_text_size"]})
plt.savefig(
    os.path.join(
        demo, demo + "_" + "-".join(parameter_names) + "_" + "-".join(data_list)
    ),
    dpi=500,
    bbox_inches="tight",
)
