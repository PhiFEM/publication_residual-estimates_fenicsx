import argparse
import os

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
for parameter in parameters_list:
    demo, param_name = parameter.split(sep="/")
    parameter_names.append(param_name)
    with open(os.path.join(parameter + ".yaml"), "rb") as f:
        parameters = yaml.safe_load(f)

    ref_type = parameters["refinement"]
    mesh_type = parameters["mesh_type"]
    if ref_type == "unif":
        trunc = -3
    elif ref_type == "adap":
        trunc = -10
    data_path = os.path.join(
        demo,
        "output_phifem_" + param_name,
        "results.csv",
    )
    df = pl.read_csv(data_path)
    for d in data_list:
        if d != "Reference_error":
            xs = df["dof"].to_numpy()
            ys = df[d].to_numpy() / df["Reference_error"].to_numpy()
            mask = np.isnan(ys)
            xs = xs[~mask]
            ys = ys[~mask]
            plt.plot(
                xs,
                ys,
                "-^",
                label=f"{param_name}, {d}",
            )

plt.xlabel("dof")
plt.legend()
plt.savefig(
    os.path.join(demo, "plot_efficiency_" + "-".join(parameter_names)),
    dpi=500,
    bbox_inches="tight",
)
