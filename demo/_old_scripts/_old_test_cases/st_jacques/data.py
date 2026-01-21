import numpy as np


# Levelset taken from 1st test case in: https://onlinelibrary.wiley.com/doi/abs/10.1002/num.22878
def generate_levelset(mode):
    def levelset(x):
        R = 0.47
        size_osc = 2.0
        r = mode.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = mode.atan2(x[1], x[0])
        sigma = 0.35
        val = (
            r**4 * (5.0 + size_osc * theta * mode.sin((5.0 / (theta + 1.0e-10)))) / 2.0
            - R**4
        )

        val = mode.exp(-(((r - R) / sigma) ** 2)) * val + (
            1.0 - mode.exp(-(((r - R) / sigma) ** 2))
        ) * (r - R)
        return val

    return levelset


def source_term(x):
    sigma = 0.1
    x0 = np.sqrt(0.47 / np.sqrt(2.5))
    return np.exp(-((x[0] - x0) ** 2 + x[1] ** 2) / sigma)
