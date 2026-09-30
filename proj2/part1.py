import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d

from filters import (DX, DY, box_filter, conv2d_four_loops, conv2d_two_loops,
                     gaussian_2d, gradient_magnitude, gradient_orientation)

DATA = Path("data")
MEDIA = Path("media")
SIGMA = 2.0
THRESHOLD_FINITE = 0.25
THRESHOLD_GAUSSIAN = 0.06


def load_gray(path):
    im = iio.imread(path).astype(np.float64) / 255.0
    if im.ndim == 3:
        im = im[..., :3].mean(axis=2)
    return im


def save(name, im):
    iio.imwrite(MEDIA / name, (np.clip(im, 0, 1) * 255).astype(np.uint8))


def show_signed(im):
    return 0.5 + im / (2 * np.abs(im).max())


def show_positive(im):
    return im / im.max()


def show_filter(kernel, scale=24):
    return np.kron(show_signed(kernel), np.ones((scale, scale)))


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def part_1_1():
    selfie = load_gray(DATA / "selfie.jpg")
    box = box_filter(9)
    four, t_four = timed(conv2d_four_loops, selfie, box)
    two, t_two = timed(conv2d_two_loops, selfie, box)
    ref, t_scipy = timed(convolve2d, selfie, box, mode="same", boundary="fill", fillvalue=0)
    dx = conv2d_two_loops(selfie, DX)
    dy = conv2d_two_loops(selfie, DY)
    uneven = np.arange(12.0).reshape(3, 4)
    save("selfie_gray.jpg", selfie)
    save("selfie_box.jpg", two)
    save("selfie_dx.jpg", show_signed(dx))
    save("selfie_dy.jpg", show_signed(dy))
    return {
        "image_shape": list(selfie.shape),
        "seconds_four_loops": round(t_four, 3),
        "seconds_two_loops": round(t_two, 3),
        "seconds_scipy": round(t_scipy, 5),
        "max_diff_four_vs_scipy": float(np.abs(four - ref).max()),
        "max_diff_two_vs_scipy": float(np.abs(two - ref).max()),
        "max_diff_dx_vs_scipy": float(np.abs(dx - convolve2d(selfie, DX, mode="same")).max()),
        "max_diff_dy_vs_scipy": float(np.abs(dy - convolve2d(selfie, DY, mode="same")).max()),
        "max_diff_even_kernel_vs_scipy": float(np.abs(conv2d_two_loops(selfie, uneven) - convolve2d(selfie, uneven, mode="same")).max()),
    }


def part_1_2(cam):
    gx = convolve2d(cam, DX, mode="same", boundary="symm")
    gy = convolve2d(cam, DY, mode="same", boundary="symm")
    mag = gradient_magnitude(gx, gy)
    save("cam_dx.png", show_signed(gx))
    save("cam_dy.png", show_signed(gy))
    save("cam_mag.png", show_positive(mag))
    save("cam_edges.png", mag > THRESHOLD_FINITE)
    return {"threshold": THRESHOLD_FINITE}


def part_1_3(cam):
    g = gaussian_2d(SIGMA)
    blurred = convolve2d(cam, g, mode="same", boundary="symm")
    bx = convolve2d(blurred, DX, mode="same", boundary="symm")
    by = convolve2d(blurred, DY, mode="same", boundary="symm")
    blur_mag = gradient_magnitude(bx, by)

    dog_x = convolve2d(g, DX)
    dog_y = convolve2d(g, DY)
    sx = convolve2d(cam, dog_x, mode="same", boundary="symm")
    sy = convolve2d(cam, dog_y, mode="same", boundary="symm")
    dog_mag = gradient_magnitude(sx, sy)

    save("cam_blur.png", blurred)
    save("blur_dx.png", show_signed(bx))
    save("blur_dy.png", show_signed(by))
    save("blur_mag.png", show_positive(blur_mag))
    save("blur_edges.png", blur_mag > THRESHOLD_GAUSSIAN)
    save("gaussian.png", np.kron(show_positive(g), np.ones((24, 24))))
    save("dog_x.png", show_filter(dog_x))
    save("dog_y.png", show_filter(dog_y))
    save("dog_dx.png", show_signed(sx))
    save("dog_dy.png", show_signed(sy))
    save("dog_mag.png", show_positive(dog_mag))
    save("dog_edges.png", dog_mag > THRESHOLD_GAUSSIAN)

    m = g.shape[0]
    inner = (slice(m, -m), slice(m, -m))
    orientation(sx, sy, dog_mag)
    return {
        "sigma": SIGMA,
        "kernel_size": m,
        "threshold": THRESHOLD_GAUSSIAN,
        "max_diff_interior": float(np.abs(blur_mag - dog_mag)[inner].max()),
        "max_diff_everywhere": float(np.abs(blur_mag - dog_mag).max()),
        "edge_pixels_that_differ": int(np.sum((blur_mag > THRESHOLD_GAUSSIAN) != (dog_mag > THRESHOLD_GAUSSIAN))),
    }


def orientation(gx, gy, mag):
    theta = gradient_orientation(gx, gy)
    hsv = np.dstack([theta / (2 * np.pi), np.ones_like(theta), show_positive(mag)])
    save("orientation.png", hsv_to_rgb(hsv))
    wheel_y, wheel_x = np.mgrid[-1:1:200j, -1:1:200j]
    wheel_theta = gradient_orientation(wheel_x, wheel_y)
    wheel = hsv_to_rgb(np.dstack([wheel_theta / (2 * np.pi), np.ones_like(wheel_theta), np.ones_like(wheel_theta)]))
    wheel[wheel_x ** 2 + wheel_y ** 2 > 1] = 1.0
    save("orientation_wheel.png", wheel)


def main():
    MEDIA.mkdir(exist_ok=True)
    cam = load_gray(DATA / "cameraman.png")
    results = {"part_1_1": part_1_1(), "part_1_2": part_1_2(cam), "part_1_3": part_1_3(cam)}
    (MEDIA / "part1_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
