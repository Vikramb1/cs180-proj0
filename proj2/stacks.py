import numpy as np

from frequencies import blur


def gaussian_stack(image, levels, sigma):
    return [image.copy()] + [blur(image, sigma * 2 ** (k - 1)) for k in range(1, levels)]


def laplacian_stack(image, levels, sigma):
    g = gaussian_stack(image, levels, sigma)
    return [g[k] - g[k + 1] for k in range(levels - 1)] + [g[-1]]


def blend(a, b, mask, levels, sigma):
    la = laplacian_stack(a, levels, sigma)
    lb = laplacian_stack(b, levels, sigma)
    gm = gaussian_stack(mask, levels, sigma)
    return sum(m[..., None] * x + (1 - m[..., None]) * y for m, x, y in zip(gm, la, lb))
