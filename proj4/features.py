import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from harris import dist2, get_harris_corners
from homography import apply_h, computeH, sample_bilinear


def gray(im):
    return im @ np.array([0.299, 0.587, 0.114])


def corners(im, edge=24):
    h, coords = get_harris_corners(gray(im), edge_discard=edge)
    ys, xs = coords
    return np.c_[xs, ys].astype(float), h[ys, xs]


def anms(pts, strength, n=500, robust=0.9, chunk=2048, pool=8000):
    order = np.argsort(-strength)[:pool]
    pts, strength = pts[order], strength[order]
    radius = np.full(len(pts), np.inf)
    for start in range(1, len(pts), chunk):
        stop = min(start + chunk, len(pts))
        d = dist2(pts[start:stop], pts[:stop])
        weaker = strength[start:stop, None] < robust * strength[None, :stop]
        radius[start:stop] = np.where(weaker, d, np.inf).min(1)
    keep = np.argsort(-radius)[:n]
    return pts[keep], np.sqrt(radius[keep])


def normalize(patches):
    patches = patches - patches.mean(1, keepdims=True)
    return patches / (patches.std(1, keepdims=True) + 1e-8)


def orientations(im, pts, sigma=4.5):
    g = gaussian_filter(gray(im), 1)
    gx = gaussian_filter(sobel(g, axis=1), sigma)
    gy = gaussian_filter(sobel(g, axis=0), sigma)
    xi, yi = np.round(pts).astype(int).T
    return np.arctan2(gy[yi, xi], gx[yi, xi])


def descriptors(im, pts, angles=None, window=40, size=8):
    g = gaussian_filter(gray(im), window / size / 2)
    step = window / size
    offsets = (np.arange(size) - (size - 1) / 2) * step
    ox, oy = np.meshgrid(offsets, offsets)
    if angles is None:
        angles = np.zeros(len(pts))
    c, s = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    sx = pts[:, 0, None, None] + c * ox - s * oy
    sy = pts[:, 1, None, None] + s * ox + c * oy
    patches = sample_bilinear(g[..., None], sx, sy)[..., 0]
    return normalize(patches.reshape(len(pts), -1))


def keep_inside(im, pts, margin=30):
    h, w = im.shape[:2]
    ok = (pts[:, 0] > margin) & (pts[:, 0] < w - margin) & (pts[:, 1] > margin) & (pts[:, 1] < h - margin)
    return pts[ok]


def match(d1, d2, ratio=0.6):
    d = np.sqrt(np.maximum(dist2(d1, d2), 0))
    nearest = np.argsort(d, 1)[:, :2]
    first = d[np.arange(len(d1)), nearest[:, 0]]
    second = d[np.arange(len(d1)), nearest[:, 1]]
    good = first < ratio * second
    return np.c_[np.nonzero(good)[0], nearest[good, 0]], first / second


def ransac(p1, p2, iters=3000, eps=2.0, seed=0):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(p1), bool)
    for _ in range(iters):
        pick = rng.choice(len(p1), 4, replace=False)
        try:
            H = computeH(p1[pick], p2[pick])
        except np.linalg.LinAlgError:
            continue
        inliers = np.linalg.norm(apply_h(H, p1) - p2, axis=1) < eps
        if inliers.sum() > best.sum():
            best = inliers
    return computeH(p1[best], p2[best]), best
