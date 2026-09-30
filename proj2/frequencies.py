import numpy as np
from scipy.signal import convolve2d

from filters import gaussian_1d, gaussian_2d


def blur(image, sigma):
    if sigma <= 0:
        return image.copy()
    if image.ndim == 3:
        return np.dstack([blur(image[..., c], sigma) for c in range(image.shape[2])])
    g = gaussian_1d(sigma)
    rows = convolve2d(image, g, mode="same", boundary="symm")
    return convolve2d(rows, g.T, mode="same", boundary="symm")


def unsharp_kernel(sigma, amount):
    blur_kernel = gaussian_2d(sigma)
    impulse = np.zeros_like(blur_kernel)
    impulse[blur_kernel.shape[0] // 2, blur_kernel.shape[1] // 2] = 1.0
    return (1 + amount) * impulse - amount * blur_kernel


def sharpen(image, sigma, amount):
    kernel = unsharp_kernel(sigma, amount)
    channels = [convolve2d(image[..., c], kernel, mode="same", boundary="symm") for c in range(image.shape[2])]
    return np.clip(np.dstack(channels), 0, 1)


def high_pass(image, sigma):
    return image - blur(image, sigma)


def to_gray(image):
    return image @ np.array([0.299, 0.587, 0.114])


def as_three_channels(gray):
    return np.dstack([gray, gray, gray])


def hybrid(high_image, low_image, sigma_high, sigma_low, color_high=True, color_low=True):
    if not color_high:
        high_image = as_three_channels(to_gray(high_image))
    if not color_low:
        low_image = as_three_channels(to_gray(low_image))
    return np.clip(blur(low_image, sigma_low) + high_pass(high_image, sigma_high), 0, 1)


def cutoff_frequency(sigma):
    return np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma)


def log_spectrum(gray):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse)
