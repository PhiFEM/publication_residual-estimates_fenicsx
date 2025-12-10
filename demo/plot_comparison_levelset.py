import matplotlib.pyplot as plt
import os
import pandas as pd

parent_dir = os.path.dirname(__file__)
pd.set_option('display.expand_frame_repr', False)
plt.style.use(['ggplot',"./plots.mplstyle"])

fig = plt.figure()
ax = fig.add_subplot()
df_p1 = pd.read_csv(os.path.join("pdt_sines_smooth", "output_FEM_levelset_p1", "uniform", "results.csv"))
df_p2 = pd.read_csv(os.path.join("pdt_sines_smooth", "output_FEM_levelset_p2", "uniform", "results.csv"))

dofs      = df_p1["dofs"].values
h10_error = df_p1["H10 error"].values
h10_est   = df_p1["H10 estimator"].values

ax.loglog(dofs, h10_est, "-^", label=r"$\eta$ $\varphi$ $p_1$")
ax.loglog(dofs, h10_error, "--o", label=r"$|u_{\mathrm{ref}} - u_h|_{H^1(\Omega)}$ $\varphi$ $p_1$")

dofs      = df_p2["dofs"].values
h10_error = df_p2["H10 error"].values
h10_est   = df_p2["H10 estimator"].values

ax.loglog(dofs, h10_est, "-^", label=r"$\eta$ $\varphi$ $p_2$")
ax.loglog(dofs, h10_error, "--o", label=r"$|u_{\mathrm{ref}} - u_h|_{H^1(\Omega)}$ $\varphi$ $p_2$")

plt.xlabel("dofs")
plt.legend()
plt.savefig("./comparison_levelset.pdf", bbox_inches="tight")