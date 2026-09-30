import cv2
import numpy as np

DX = np.array([[1.0, 0.0, -1.0]])
DY = np.array([[1.0], [0.0], [-1.0]])


def pad_zero(image, kh, kw):
    top, left = kh // 2, kw // 2
    bottom, right = kh - 1 - top, kw - 1 - left
    padded = np.zeros((image.shape[0] + kh - 1, image.shape[1] + kw - 1), dtype=np.float64)
    padded[top:top + image.shape[0], left:left + image.shape[1]] = image
    return padded


def conv2d_four_loops(image, kernel):
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    padded = pad_zero(image, kh, kw)
    out = np.zeros(image.shape, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += padded[i + u, j + v] * flipped[u, v]
            out[i, j] = total
    return out


def conv2d_two_loops(image, kernel):
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    padded = pad_zero(image, kh, kw)
    out = np.zeros(image.shape, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * flipped)
    return out


def box_filter(size):
    return np.ones((size, size)) / (size * size)


def gaussian_1d(sigma):
    size = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.getGaussianKernel(size, sigma)


def gaussian_2d(sigma):
    g = gaussian_1d(sigma)
    return g @ g.T


def gradient_magnitude(gx, gy):
    return np.sqrt(gx * gx + gy * gy)


def gradient_orientation(gx, gy):
    safe_gx = np.where(gx == 0, 1e-12, gx)
    theta = np.arctan(gy / safe_gx)
    theta = np.where(gx < 0, theta + np.pi, theta)
    return np.mod(theta, 2 * np.pi)
