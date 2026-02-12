import numpy as np

tilt_angle = np.pi / 6.0
shift = np.array([np.pi / 32.0, np.pi / 32.0])
INITIAL_MESH_SIZE = 0.1
MAXIMUM_DOF = 1.0e5
REFERENCE = "phifem-bc-geo"
MAX_EXTRA_STEP_ADAP = 2
MAX_EXTRA_STEP_UNIF = 2


def rotation(angle, x, y):
    return (
        np.cos(angle) * x - np.sin(angle) * y,
        np.sin(angle) * x + np.cos(angle) * y,
    )


def line(x, y, a, b, c):
    rotated = rotation(tilt_angle, x, y)
    return a * rotated[0] + b * rotated[1] + c


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

    def expression_levelset(x):
        x_shift = x[0] - shift[0]
        y_shift = x[1] - shift[1]

        line_1 = line(x_shift, y_shift, -1.0, 0.0, 0.0)
        line_2 = line(x_shift, y_shift, 0.0, -1.0, 0.0)
        line_3 = line(x_shift, y_shift, 1.0, 0.0, -0.5)
        line_4 = line(x_shift, y_shift, 0.0, 1.0, -0.5)
        line_5 = line(x_shift, y_shift, 0.0, -1.0, -0.5)
        line_6 = line(x_shift, y_shift, -1.0, 0.0, -0.5)

        reentrant_corner = minimum(line_1, line_2)
        top_right_corner = maximum(line_3, line_4)
        corner = maximum(reentrant_corner, top_right_corner)
        horizontal_leg = maximum(corner, line_5)
        vertical_leg = maximum(horizontal_leg, line_6)
        return vertical_leg

    return expression_levelset


def generate_source_term(mode):
    def source_term(x):
        return np.ones_like(x[0])

    return source_term


def generate_dirichlet_data(mode):
    def dirichlet_data(x):
        return np.zeros_like(x[0])

    return dirichlet_data


def rotate_shift(angle, shift, x):
    return (
        np.cos(angle) * x[0] - np.sin(angle) * x[1] - shift[0],
        np.sin(angle) * x[0] + np.cos(angle) * x[1] - shift[1],
    )


def gen_mesh(hmax, curved=False):
    import ngsPETSc.utils.fenicsx as ngfx
    from mpi4py import MPI
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    pnts = [
        (0.0, 0.0),
        (0.0, -0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.0),
    ]
    for i, pt in enumerate(pnts):
        pnts[i] = rotate_shift(-tilt_angle, -shift, pt)

    p1, p2, p3, p4, p5, p6 = [geo.AppendPoint(*pnt) for pnt in pnts]
    lines = [
        [["line", p1, p2], "line"],
        [["line", p2, p3], "line"],
        [["line", p3, p4], "line"],
        [["line", p4, p5], "line"],
        [["line", p5, p6], "line"],
        [["line", p6, p1], "line"],
    ]
    for c, bc in lines:
        geo.Append(c, bc=bc)

    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    mesh = geoModel.model_to_mesh(gdim=2, hmax=hmax)[0]
    return mesh, geoModel
