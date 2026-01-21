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

parser.add_argument(
    "trunc",
    type=int,
    default=1,
    help="Truncation of the top of the data to be plotted.",
)

parser.add_argument("-r", "--rate", action="store_true")

args = parser.parse_args()
parameters_list = args.parameters.split(sep=",")
data_list = args.data.split(sep=",")
trunc_top = -args.trunc
rate = args.rate

fig = plt.figure()
ax = fig.subplots()
parameter_names = []

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

eta_comp = np.all(["eta" in data for data in data_list])
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
        trunc_btm = trunc_top - 3
    elif ref_type == "adap":
        trunc_btm = trunc_top - 8
    data_path = os.path.join(
        demo,
        "output_" + param_name,
        "results.csv",
    )

    df = pl.read_csv(data_path)
    for d in data_list:
        d_name = plot_param[d]["name"]
        if eta_comp:
            label = rf"{d_name}"
        else:
            label = rf"{d_name} ({scheme_name})"
        lstyle = plot_param[d]["line"]
        if eta_comp:
            color = plot_param[d]["color"]
        else:
            color = plot_param[param_name]["color"]
        xs = df["dof"].to_numpy()
        try:
            ys = df[d].to_numpy()
        except pl.exceptions.ColumnNotFoundError:
            print(f"{d} not found.")
            continue
        mask = np.isnan(ys)
        xs = xs[~mask]
        ys = ys[~mask]
        try:
            ys_not_zero = np.max(ys) > 0.0
        except ValueError:
            ys_not_zero = False
        if ys_not_zero:
            if rate:
                trunc_btm = -np.maximum(-(trunc_top - 5), 0)
                a, b = np.polyfit(
                    np.log(xs[trunc_btm:trunc_top]),
                    np.log(ys[trunc_btm:trunc_top]),
                    deg=1,
                )
                rate = " (s=" + str(np.round(a, 1)) + ")"
            else:
                rate = ""
            plt.loglog(
                xs[:trunc_top],
                ys[:trunc_top],
                lstyle + mstyle,
                c=color,
                label=label + rate,
                path_effects=[
                    pe.Stroke(linewidth=2.5, foreground="#252525"),
                    pe.Normal(),
                ],
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
