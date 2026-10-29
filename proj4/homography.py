import numpy as np


def system(im1_pts, im2_pts):
    rows, rhs = [], []
    for (x, y), (u, v) in zip(im1_pts, im2_pts):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    return np.array(rows, float), np.array(rhs, float)


def computeH(im1_pts, im2_pts):
    A, b = system(im1_pts, im2_pts)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def apply_h(H, pts):
    pts = np.asarray(pts, float)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def corners(shape):
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)


def bounds(images, Hs):
    pts = np.vstack([apply_h(H, corners(im.shape)) for im, H in zip(images, Hs)])
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def target_grid(H, shape, box=None):
    if box is None:
        box = bounds([np.zeros(shape)], [H])
    x0, y0, w, h = box
    xs, ys = np.meshgrid(np.arange(w) + x0, np.arange(h) + y0)
    src = apply_h(np.linalg.inv(H), np.c_[xs.ravel(), ys.ravel()])
    return src[:, 0].reshape(h, w), src[:, 1].reshape(h, w), box


def inside(im, sx, sy):
    h, w = im.shape[:2]
    return (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)


def sample_nn(im, sx, sy):
    h, w = im.shape[:2]
    xi = np.clip(np.round(sx).astype(int), 0, w - 1)
    yi = np.clip(np.round(sy).astype(int), 0, h - 1)
    return im[yi, xi]


def sample_bilinear(im, sx, sy):
    h, w = im.shape[:2]
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    x0 = np.minimum(np.floor(sx).astype(int), w - 2)
    y0 = np.minimum(np.floor(sy).astype(int), h - 2)
    a = (sx - x0)[..., None]
    b = (sy - y0)[..., None]
    top = im[y0, x0] * (1 - a) + im[y0, x0 + 1] * a
    bottom = im[y0 + 1, x0] * (1 - a) + im[y0 + 1, x0 + 1] * a
    return top * (1 - b) + bottom * b


def warp(im, H, sampler, box=None):
    sx, sy, box = target_grid(H, im.shape, box)
    alpha = inside(im, sx, sy).astype(float)
    out = sampler(im, sx, sy) * alpha[..., None]
    return out, alpha, box


def warpImageNearestNeighbor(im, H, box=None):
    return warp(im, H, sample_nn, box)


def warpImageBilinear(im, H, box=None):
    return warp(im, H, sample_bilinear, box)
