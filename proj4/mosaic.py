import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from homography import bounds, sample_bilinear, warpImageBilinear


def warp_all(images, Hs):
    box = bounds(images, Hs)
    warped = [warpImageBilinear(im, H, box)[:2] for im, H in zip(images, Hs)]
    return [w for w, _ in warped], [a for _, a in warped], box


def feather_weights(alphas):
    return [distance_transform_edt(a > 0.5) for a in alphas]


def overwrite(warped, alphas):
    out = np.zeros_like(warped[0])
    for w, a in zip(warped, alphas):
        out[a > 0.5] = w[a > 0.5]
    return out


def feather(warped, weights):
    total = sum(weights) + 1e-9
    return sum(w * (k / total)[..., None] for w, k in zip(warped, weights))


def blur(im, sigma):
    return np.stack([gaussian_filter(im[..., c], sigma) for c in range(im.shape[2])], 2)


def two_band(warped, alphas, weights, sigma=4):
    total = sum(weights) + 1e-9
    best = np.argmax(np.stack(weights), 0)
    low = np.zeros_like(warped[0])
    high = np.zeros_like(warped[0])
    for i, (w, a, k) in enumerate(zip(warped, alphas, weights)):
        smooth = blur(w, sigma) / np.maximum(gaussian_filter(a, sigma), 1e-6)[..., None]
        low += smooth * (k / total)[..., None]
        high += np.where((best == i)[..., None], w - smooth, 0)
    covered = (sum(alphas) > 0.5)[..., None]
    return np.clip(low + high, 0, 1) * covered


def cylinder(images, Hs, f, center, weights_from=None):
    cx, cy = center
    planes = [np.linalg.inv(H) for H in Hs]
    xs = []
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        edge = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [0, h / 2], [w - 1, h / 2]], float)
        p = np.c_[edge, np.ones(len(edge))] @ H.T
        xs.append(p[:, 0] / p[:, 2])
    xs = np.concatenate(xs) - cx
    t0, t1 = np.arctan(xs.min() / f), np.arctan(xs.max() / f)
    width = int((t1 - t0) * f)
    theta = t0 + np.arange(width) / f
    heights = []
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        for y in (0, h - 1):
            row = np.c_[np.arange(w), np.full(w, y), np.ones(w)] @ H.T
            px, py = row[:, 0] / row[:, 2] - cx, row[:, 1] / row[:, 2] - cy
            heights.append(py * np.cos(np.arctan(px / f)))
    heights = np.concatenate(heights)
    v = np.arange(int(heights.min()), int(heights.max()) + 1)
    T, V = np.meshgrid(theta, v)
    plane = np.stack([f * np.tan(T) + cx, V / np.cos(T) + cy, np.ones_like(T)], -1)
    warped, alphas = [], []
    for im, inv in zip(images, planes):
        p = plane @ inv.T
        sx, sy = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
        h, w = im.shape[:2]
        a = ((sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)).astype(float)
        warped.append(sample_bilinear(im, sx, sy) * a[..., None])
        alphas.append(a)
    return warped, alphas
