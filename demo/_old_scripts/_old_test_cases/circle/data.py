import gmsh
import ngsPETSc.utils.fenicsx as ngfx
import numpy as np
import ufl
from mpi4py import MPI
from netgen.geom2d import SplineGeometry


def generate_levelset(mode):
    def levelset(x):
        return mode.sqrt(x[0] ** 2 + x[1] ** 2) - 1.0

    return levelset


def generate_source_term(mode):
    def source_term(x):
        return np.zeros_like(x[0])

    return source_term


def generate_dirichlet_data(mode):
    def dirichlet_data(x):
        return mode.sqrt((x[0] - 1.0) ** 2 + x[1] ** 2) ** (2.0 / 3.0)

    return dirichlet_data


def gen_mesh(hmax, curved=False):
    geo = SplineGeometry()
    pnts = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]
    p1, p2, p3, p4, p5, p6, p7, p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = [
        [["spline3", p1, p2, p3], "curve"],
        [["spline3", p3, p4, p5], "curve"],
        [["spline3", p5, p6, p7], "curve"],
        [["spline3", p7, p8, p1], "curve"],
    ]
    for c, bc in curves:
        geo.Append(c, bc=bc)

    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    mesh = geoModel.model_to_mesh(gdim=2, hmax=hmax)[0]
    if curved:
        mesh = geoModel.curveField(3)

    return mesh, geoModel
