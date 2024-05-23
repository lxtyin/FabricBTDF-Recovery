from utils import *


# region sggx ================

# input: w(..., 3)
def sigma(w, alpha):
    return torch.sqrt(w[..., [0]] ** 2 + w[..., [1]] ** 2 + (w[..., [2]] * alpha) ** 2)


def sigmaT(w, alpha, density=1.0):
    return sigma(w, alpha) * density


def D(w, alpha):
    s2 = w[..., [0]] ** 2 + w[..., [1]] ** 2 + (w[..., [2]] / alpha) ** 2
    result = 1 / (PI * alpha * torch.square(s2))
    return result


def eval(wi, half, alpha):
    return D(half, alpha) * 0.25 / sigma(wi, alpha)

# endregion
