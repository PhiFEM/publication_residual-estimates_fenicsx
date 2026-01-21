import dolfinx as dfx
import numpy as np
import ufl

delta = 1.0e-20


# Triangle function approximation
def _trg(x):
    return 1.0 - 2.0 * ufl.acos((1 - delta) * ufl.sin(2.0 * np.pi * x)) / np.pi


# Square function approximation
def _sqr(x):
    return 2.0 * ufl.atan(ufl.sin(2.0 * np.pi * x) / delta) / np.pi


# Sawtooth function approximation
def _swt(x):
    return 1.0 + _trg((2.0 * x - 1.0) / 4.0) * _sqr(x / 2.0) / 2.0


def generate_levelset(mode):
    if mode.__name__ == "numpy":

        def floor(x):
            return np.floor(x)
    elif mode.__name__ == "ufl":

        def floor(x):
            return x - _swt(x) + 0.5

    def levelset(x):
        r = mode.sqrt(x[0] ** 2 + x[1] ** 2)
        R = 2.0
        radius = 0.5
        theta = mode.atan2(x[1], x[0])

        sigma = 0.35

        val = (
            r
            * (
                1.0
                + 2.0
                * abs(theta)
                * mode.sqrt(
                    radius**2
                    - (
                        1.0 / (mode.sqrt(abs(theta)) + 1.0e-18)
                        - floor(1.0 / (mode.sqrt(abs(theta)) + 1.0e-18) + radius)
                    )
                    ** 2
                )
            )
            - R
        )

        val = mode.exp(-(((r - R) / sigma) ** 2)) * val + (
            1.0 - mode.exp(-(((r - R) / sigma) ** 2))
        ) * (r - R)

        return val

    return levelset


def generate_source_term(mode):
    def source_term(x):
        sigma = 0.1
        x0 = 1.9
        return np.exp(-((x[0] - x0) ** 2 + x[1] ** 2) / sigma)

    return source_term


def generate_dirichlet_data(mode):
    def dirichlet_data(x):
        return np.zeros_like(x[0])

    return dirichlet_data
