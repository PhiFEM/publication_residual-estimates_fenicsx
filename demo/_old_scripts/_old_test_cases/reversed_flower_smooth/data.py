import numpy as np
from netgen.geom2d import SplineGeometry


def atan_r(x, radius=1.0, slope=1.0):
    r = np.sqrt(np.square(x[0, :]) + np.square(x[1, :]))
    r0 = np.full_like(r, radius)
    val = np.arctan(slope * (r - r0))
    return val


# Implementation of a graded smooth-min function inspired from: https://iquilezles.org/articles/smin/
def smax(x, y_1, y_2, kmin=0.0, kmax=1.0):
    k = kmax * ((np.pi / 2.0 - atan_r(x, radius=3.0, slope=15.0)) / np.pi / 2.0) + kmin
    return np.maximum(k, np.maximum(y_1, y_2)) - np.linalg.norm(
        np.maximum(np.vstack([k, k]) - np.vstack([y_1, y_2]), 0.0), axis=0
    )


# Levelset and RHS expressions taken from: https://academic.oup.com/imajna/article-abstract/42/1/333/6041856?redirectedFrom=fulltext
def generate_levelset(mode):
    def minimum(f1, f2):
        if mode.__name__ == "numpy":
            return mode.minimum(f1, f2)
        elif mode.__name__ == "ufl":
            return mode.conditional(f1 < f2, f1, f2)

    def maximum(f1, f2):
        if mode.__name__ == "numpy":
            return mode.maximum(f1, f2)
        elif mode.__name__ == "ufl":
            return mode.conditional(f1 > f2, f1, f2)

    def def_phi0(x):
        return x[0] ** 2 + x[1] ** 2 - 4.0

    def levelset(x):
        val = def_phi0(x)
        R = 2.0
        r = 0.9
        num_petals = 8
        for i in range(1, num_petals + 1):
            xi = R * np.cos(2.0 * i * np.pi / num_petals)
            yi = R * np.sin(2.0 * i * np.pi / num_petals)

            def def_phi_i(x):
                return (x[0] - xi) ** 2 + (x[1, :] - yi) ** 2 - r**2

            val = maximum(val, -def_phi_i(x))
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
        (-3.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
    ]
    for i, pt in enumerate(pnts):
        rot_pt = _rotate([pt[0], pt[1]], -_angle)
        pnts[i] = (float(rot_pt[0]), float(rot_pt[1]))

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
