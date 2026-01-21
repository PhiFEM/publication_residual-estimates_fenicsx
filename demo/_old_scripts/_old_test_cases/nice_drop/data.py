import ngsPETSc.utils.fenicsx as ngfx
import numpy as np
import ufl
from mpi4py import MPI
from netgen.geom2d import SplineGeometry

_small_radius = 0.01
_large_radius = 0.5
_angle = -np.pi / 16.0
INITIAL_MESH_SIZE = 0.1
MAXIMUM_DOF = 5.0e4


def _rotate(x, angle):
    _sine_angle = np.sin(angle)
    _cosine_angle = np.cos(angle)
    rot_x = np.zeros_like(x)
    rot_x[0] = _cosine_angle * x[0] + _sine_angle * x[1]
    rot_x[1] = -_sine_angle * x[0] + _cosine_angle * x[1]
    return rot_x


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

    def phi_large_circle(x):
        val = x[0] ** 2 + x[1] ** 2 - _large_radius**2
        return val

    def phi_small_circle(x):
        val = (
            (x[0] + 2 * _large_radius - 2 * _small_radius) ** 2
            + (x[1] - _large_radius + _small_radius) ** 2
            - _small_radius**2
        )
        return val

    def phi_out_circle(x):
        val = (
            (x[0] + 2 * _large_radius - 2 * _small_radius) ** 2
            + x[1] ** 2
            - (_large_radius - 2 * _small_radius) ** 2
        )
        return val

    def phi_rectangle(x):
        val = maximum(-x[0] + (-2 * _large_radius + 2 * _small_radius), x[0])
        val = maximum(val, -x[1])
        val = maximum(val, x[1] - _large_radius)
        return val

    def levelset(x):
        x = _rotate(x, -_angle)
        val = maximum(phi_rectangle(x), -phi_out_circle(x))
        val = minimum(val, phi_large_circle(x))
        val = minimum(val, phi_small_circle(x))
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
        (0.0, 0.0),  #                                                        A
        (-_large_radius, 0.0),  #                                             B
        (-_large_radius, -_large_radius),  #                                  C
        (0.0, -_large_radius),  #                                             D
        (_large_radius, -_large_radius),  #                                   E
        (_large_radius, 0.0),  #                                              F
        (_large_radius, _large_radius),  #                                    G
        (0.0, _large_radius),  #                                              H
        (-2.0 * _large_radius + 2.0 * _small_radius, _large_radius),  #       I
        (-2.0 * _large_radius + _small_radius, _large_radius - _small_radius),  # J
        (-2.0 * _large_radius + 2.0 * _small_radius, 0.0),  #  M
        (
            -2.0 * _large_radius + 2.0 * _small_radius,
            _large_radius - 2.0 * _small_radius,
        ),  # N
        (
            -2.0 * _large_radius + 2.0 * _small_radius,
            _large_radius - _small_radius,
        ),  # O
    ]
    for i, pt in enumerate(pnts):
        rot_pt = _rotate([pt[0], pt[1]], _angle)
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
