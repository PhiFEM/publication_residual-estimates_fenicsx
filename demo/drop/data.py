import ngsPETSc.utils.fenicsx as ngfx
import numpy as np
import ufl
from mpi4py import MPI
from netgen.geom2d import SplineGeometry


def generate_levelset(mode):
    def minimum(f1, f2):
        if mode.__name__ == "numpy":
            return np.minimum(f1, f2)
        elif mode.__name__ == "ufl":
            return ufl.conditional(f1 < f2, f1, f2)

    def maximum(f1, f2):
        if mode.__name__ == "numpy":
            return np.maximum(f1, f2)
        elif mode.__name__ == "ufl":
            return ufl.conditional(f1 > f2, f1, f2)

    def phi_circle(x):
        val = mode.sqrt(x[0] ** 2 + x[1] ** 2) - 1.0
        return val

    def phi_ellipse(x):
        val = 2.0 * (mode.sqrt((0.5 * (x[0] + 3.0)) ** 2 + x[1] ** 2) - 1.0)
        return val

    def phi_rectangle(x):
        val_x = maximum(-x[0] - 3.0, x[0])
        val_y = maximum(x[1], -x[1] - 1.0)
        val = maximum(val_x, val_y)
        return val

    def levelset(x):
        val = maximum(phi_rectangle(x), -phi_ellipse(x))
        val = minimum(val, phi_circle(x))
        return val

    return levelset


def generate_source_term(mode):
    def source_term(x):
        return np.ones_like(x[0])

    return source_term


def generate_dirichlet_data(mode):
    def dirichlet_data(x):
        return np.zeros_like(x[0])

    return dirichlet_data


def gen_mesh(hmax, curved=False):
    geo = SplineGeometry()
    pnts = [
        (-3, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]

    pts = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = [
        [["line", pts[0], pts[1]], "line"],
        [["spline3", pts[1], pts[2], pts[3]], "curve"],
        [["spline3", pts[3], pts[4], pts[5]], "curve"],
        [["spline3", pts[5], pts[6], pts[7]], "curve"],
        [["spline3", pts[7], pts[8], pts[0]], "curve"],
    ]
    for c, bc in curves:
        geo.Append(c, bc=bc)

    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    mesh = geoModel.model_to_mesh(gdim=2, hmax=hmax)[0]
    if curved:
        mesh = geoModel.curveField(3)

    return mesh, geoModel
