import numpy as np

# def expression_levelset(x):
#     m = 3
#     vals = (x[0,:]*x[0,:] + x[1,:]*x[1,:] - np.ones_like(x[0,:]))**m - x[0,:]**2 * x[1,:]**m
#     return vals


def levelset(x):
    xs = x[0, :]
    ys = x[1, :]

    vals = (
        xs**2
        + (
            ys
            - (2.0 * (xs**2 + np.abs(xs) - np.full_like(xs, 6.0)))
            / (3.0 * (xs**2 + np.abs(xs) + np.full_like(xs, 2.0)))
        )
        ** 2
        - 36.0
    )
    return vals


def detection_levelset(x):
    return expression_levelset(x)


def source_term(x):
    return np.ones_like(x[0, :])
