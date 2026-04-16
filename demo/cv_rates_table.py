import argparse
import os

import numpy as np
import polars as pl
import yaml

parent_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog="plot.py", description="Plot the demo results.")

parser.add_argument("demo", type=str, help="Choose the demo.")

parser.add_argument(
    "--trunc",
    "-t",
    type=int,
    default=3,
    help="Truncation of the top of the data to be plotted.",
)

args = parser.parse_args()
demo = args.demo
trunc_top = -args.trunc

data_list = [
    "estimator",
    "h10_error",
    "triple_norm_error",
    "eta_r",
    "eta_J",
    "eta_BC",
    "eta_geo",
]
parameters_list = ["phifem-bc-geo", "fem"]

with open(os.path.join(parent_dir, "plot_parameters.yaml"), "rb") as f:
    plot_param = yaml.safe_load(f)

results = {}
tex_template_path = os.path.join(
    "_tex_tables_templates",
    "table_template.tex",
)
with open(tex_template_path, "r") as f:
    content = f.read()
    max_dofs = []
    for param_name in parameters_list:
        data_path = os.path.join(
            demo,
            "output_" + param_name,
            "results.csv",
        )
        df = pl.read_csv(data_path)
        for d in data_list:
            if d == "estimator":
                d_name = plot_param[param_name]["estimator_name"]
            else:
                d_name = plot_param[d]["name"]
            try:
                ys = df[d].to_numpy()
                y_mask = np.logical_not(np.isnan(ys))
                xs = df["dof"].to_numpy()[y_mask]
            except pl.exceptions.ColumnNotFoundError:
                print(f"{d} not found.")
                xs = df["dof"].to_numpy()
                continue
        max_dofs.append(max(xs))

    data_path = os.path.join(
        demo,
        "output_" + "fem",
        "results.csv",
    )
    df = pl.read_csv(data_path)
    ys = df["h10_error"].to_numpy()
    y_mask = np.logical_not(np.isnan(ys))
    xs = df["dof"].to_numpy()[y_mask]
    max_dofs.append(max(xs))
    min_max_dofs = min(max_dofs)

    for param_name in parameters_list:
        tex_file_path = os.path.join(
            demo,
            demo + ".tex",
        )

        with open(os.path.join(demo, param_name + ".yaml"), "rb") as f:
            parameters = yaml.safe_load(f)

        ref_type = parameters["refinement"]
        mesh_type = parameters["mesh_type"]
        scheme_name = plot_param[param_name]["name"]
        if ref_type == "unif":
            trunc_btm = trunc_top - 10
        elif ref_type == "adap":
            trunc_btm = trunc_top - 10
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
            dofs_mask = xs <= min_max_dofs
            xs = xs[dofs_mask]
            ys = ys[dofs_mask]
            mask = np.isnan(ys)
            xs = xs[~mask]
            ys = ys[~mask]
            try:
                ys_not_zero = np.max(ys) > 0.0
            except ValueError:
                ys_not_zero = False
            if ys_not_zero:
                if trunc_top == 0:
                    slope, b = np.polyfit(
                        np.log(xs[trunc_btm:]),
                        np.log(ys[trunc_btm:]),
                        1,
                    )
                else:
                    slope, b = np.polyfit(
                        np.log(xs[trunc_btm:trunc_top]),
                        np.log(ys[trunc_btm:trunc_top]),
                        1,
                    )
                content = content.replace(
                    "[" + param_name + "-" + d + "]",
                    str(np.abs(np.round(slope, 2))).ljust(4, "0")
                    + " " * (len("[" + param_name + "-" + d + "]") - 5),
                )
            else:
                continue

    with open(tex_file_path, "w") as of:
        of.write(content)
