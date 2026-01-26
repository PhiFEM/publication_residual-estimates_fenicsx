import argparse
import os

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

args = parser.parse_args()
parameters_list = args.parameters.split(sep=",")
data_list = args.data.split(sep=",")
trunc_top = -args.trunc

parameter_names = []

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

results = {}
demo = parameters_list[0].split(sep="/")[0]
tex_template_path = os.path.join(
    "_tex_tables_templates",
    "table_template.tex",
)
with open(tex_template_path, "r") as f:
    content = f.read()
    for parameter in parameters_list:
        demo, param_name = parameter.split(sep="/")
        tex_file_path = os.path.join(
            demo,
            demo + "_" + "-".join(parameter_names) + "_" + "-".join(data_list) + ".tex",
        )

        parameter_names.append(param_name)
        with open(os.path.join(parameter + ".yaml"), "rb") as f:
            parameters = yaml.safe_load(f)

        ref_type = parameters["refinement"]
        mesh_type = parameters["mesh_type"]
        scheme_name = plot_param[param_name]["name"]
        if ref_type == "unif":
            trunc_btm = trunc_top - 3
        elif ref_type == "adap":
            trunc_btm = trunc_top - 15
        data_path = os.path.join(
            demo,
            "output_" + param_name,
            "results.csv",
        )

        df = pl.read_csv(data_path)
        for d in data_list:
            if d == "estimator":
                d_name = scheme_name.split(sep=" ")[1][1:-1]
            else:
                d_name = plot_param[d]["name"]
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
                slope, b = np.polyfit(
                    np.log(xs[trunc_btm:trunc_top]),
                    np.log(ys[trunc_btm:trunc_top]),
                    1,
                )
                content = content.replace(
                    "[" + param_name + "-" + d + "]",
                    str(np.round(slope, 2)).ljust(5, "0")
                    + " " * (len("[" + param_name + "-" + d + "]") - 5),
                )
            else:
                continue

    with open(tex_file_path, "w") as of:
        of.write(content)
