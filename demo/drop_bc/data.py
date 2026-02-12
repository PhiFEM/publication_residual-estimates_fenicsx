import numpy as np

_angle = -np.pi / 16.0
INITIAL_MESH_SIZE = 0.1
MAXIMUM_DOF = 1.0e5
REFERENCE = "phifem-bc-geo"
MAX_EXTRA_STEP_ADAP = 2
MAX_EXTRA_STEP_UNIF = 2


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
            return mode.minimum(f1, f2)
        elif mode.__name__ == "ufl":
            return mode.conditional(f1 < f2, f1, f2)

    def maximum(f1, f2):
        if mode.__name__ == "numpy":
            return mode.maximum(f1, f2)
        elif mode.__name__ == "ufl":
            return mode.conditional(f1 > f2, f1, f2)

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
        x = _rotate(x, _angle)
        val = maximum(phi_rectangle(x), -phi_ellipse(x))
        val = minimum(val, phi_circle(x))
        return val

    return levelset


def generate_exact_solution(mode):
    c = _rotate(np.asarray([-3.0, -1.0]), -_angle)
    sigma = 5.0
    # def exact_solution(x):
    #     x = _rotate(x, _angle)
    #     return ((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2) ** (1.0 / 3.0)

    def exact_solution(x):
        x = _rotate(x, _angle)
        return mode.exp(-sigma * ((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2))

    return exact_solution


def gen_mesh(hmax, curved=False):
    import ngsPETSc.utils.fenicsx as ngfx
    from mpi4py import MPI
    from netgen.geom2d import SplineGeometry

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
