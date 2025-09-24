import numpy as np
import ufl


def levelset(x):
    values = np.sqrt(
        (x[0] + np.pi / 6.0) ** 2 + (x[1] + np.pi / 6.0) ** 2
    ) - np.ones_like(x[0, :])
    return values


def generate_exact_solution(mode):
    def exact_solution(x):
        r = mode.sqrt((x[0] + np.pi / 6.0) ** 2 + (x[1] + np.pi / 6.0) ** 2)
        val = mode.cos((np.pi / 2.0) * r)
        return val


def exact_solution(x):
    exact_sol = generate_exact_solution(np)
    return exact_sol(x)


def source_term(x):
    exact_sol = generate_exact_solution(ufl)
    return -ufl.div(ufl.grad(exact_sol(x)))
