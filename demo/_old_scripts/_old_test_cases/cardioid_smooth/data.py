import numpy as np

# def expression_levelset(x):
#     m = 3
#     vals = (x[0,:]*x[0,:] + x[1,:]*x[1,:] - np.ones_like(x[0,:]))**m - x[0,:]**2 * x[1,:]**m
#     return vals

def expression_levelset(x):
    xs = x[0,:]
    ys = x[1,:]

    vals = (xs**2 + ys**2 + xs)**2 - xs**2 - ys**2
    return vals

def expression_detection_levelset(x):
    return expression_levelset(x)

def expression_rhs(x):
    return np.ones_like(x[0,:])