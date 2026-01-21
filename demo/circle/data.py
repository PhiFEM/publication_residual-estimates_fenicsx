import ngsPETSc.utils.fenicsx as ngfx
from mpi4py import MPI
from netgen.geom2d import SplineGeometry

INITIAL_MESH_SIZE = 0.1
MAXIMUM_DOF = 5.0e4


def generate_levelset(mode):
    def levelset(x):
        return mode.sqrt(x[0] ** 2 + x[1] ** 2) - 1.0

    return levelset


def generate_exact_solution(mode):
    def exact_solution(x):
        return mode.sqrt((x[0] - 1.0) ** 2 + x[1] ** 2) ** (2.0 / 3.0)

    return exact_solution


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
