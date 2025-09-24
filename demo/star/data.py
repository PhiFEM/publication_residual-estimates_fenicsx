import numpy as np


# Levelset taken from 1st test case in: https://onlinelibrary.wiley.com/doi/abs/10.1002/num.22878
def levelset(x):
    R = 0.47
    size_osc = 2.0
    r = np.sqrt(x[0, :] ** 2 + x[1, :] ** 2)
    theta = np.arctan2(x[1], x[0])
    val = (
        r**4 * (np.full_like(r, 5.0) + size_osc * theta * np.sin((5.0 / theta))) / 2.0
        - R**4
    )
    return val


def source_term(x):
    sigma = 0.1
    x0 = np.sqrt(0.47 / np.sqrt(2.5))
    return np.exp(-((x[0] - x0) ** 2 + x[1] ** 2) / sigma)
