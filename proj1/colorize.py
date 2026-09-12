import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.transform import rescale


def load(path):
    im = img_as_float(iio.imread(path)).astype(np.float32)
    if im.ndim == 3:
        im = im.mean(axis=2)
    h = im.shape[0] // 3
    return im[:h], im[h:2 * h], im[2 * h:3 * h]


def crop(im, frac):
    h, w = im.shape
    dy, dx = int(h * frac), int(w * frac)
    return im[dy:h - dy, dx:w - dx]


def ssd(a, b):
    return float(np.sum((a - b) ** 2))


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return -float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


METRICS = {"ssd": ssd, "ncc": ncc}
def grad(im):
    gy = np.zeros_like(im)
    gx = np.zeros_like(im)
    gy[1:-1, :] = im[2:, :] - im[:-2, :]
    gx[:, 1:-1] = im[:, 2:] - im[:, :-2]
    return np.sqrt(gx * gx + gy * gy)


FEATURES = {"raw": lambda im: im, "grad": grad}


def shift(im, dx, dy):
    return np.roll(im, (dy, dx), axis=(0, 1))


def exhaustive(im, ref, metric, radius, center=(0, 0), margin=0.1):
    ref_c = crop(ref, margin)
    best = None
    for dy in range(center[1] - radius, center[1] + radius + 1):
        for dx in range(center[0] - radius, center[0] + radius + 1):
            score = metric(crop(shift(im, dx, dy), margin), ref_c)
            if best is None or score < best[0]:
                best = (score, dx, dy)
    return best[1], best[2], best[0]


def pyramid(im, min_size, feature):
    levels = [im]
    while min(levels[-1].shape) > min_size:
        levels.append(rescale(levels[-1], 0.5, anti_aliasing=True).astype(np.float32))
    return [feature(l) for l in levels]


def align(im, ref, metric, radius, refine, min_size, feature):
    ims = pyramid(im, min_size, feature)
    refs = pyramid(ref, min_size, feature)
    dx, dy, loss = exhaustive(ims[-1], refs[-1], metric, radius)
    for level in range(len(ims) - 2, -1, -1):
        dx, dy, loss = exhaustive(ims[level], refs[level], metric, refine, (2 * dx, 2 * dy))
    return dx, dy, loss, len(ims)


def to_web(im, width):
    if im.shape[1] <= width:
        return im
    return rescale(im, width / im.shape[1], anti_aliasing=True, channel_axis=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--metric", choices=METRICS, default="ncc")
    p.add_argument("--feature", choices=FEATURES, default="raw")
    p.add_argument("--tag")
    p.add_argument("--suffix", default="")
    p.add_argument("--radius", type=int, default=15)
    p.add_argument("--refine", type=int, default=3)
    p.add_argument("--min-size", type=int, default=400)
    p.add_argument("--out", default="out")
    p.add_argument("--web", default="media")
    p.add_argument("--web-width", type=int, default=900)
    args = p.parse_args()

    out_dir = Path(args.out)
    web_dir = Path(args.web)
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)
    results_path = web_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    metric = METRICS[args.metric]
    feature = FEATURES[args.feature]
    tag = args.tag or (args.metric if args.feature == "raw" else f"{args.metric}_{args.feature}")

    for path in args.inputs:
        name = Path(path).stem
        t0 = time.time()
        b, g, r = load(path)
        gx, gy, gloss, levels = align(g, b, metric, args.radius, args.refine, args.min_size, feature)
        rx, ry, rloss, _ = align(r, b, metric, args.radius, args.refine, args.min_size, feature)
        color = np.dstack([shift(r, rx, ry), shift(g, gx, gy), b])
        iio.imwrite(out_dir / f"{name}{args.suffix}.jpg", img_as_ubyte(np.clip(color, 0, 1)), quality=92)
        iio.imwrite(web_dir / f"{name}{args.suffix}.jpg", img_as_ubyte(np.clip(to_web(color, args.web_width), 0, 1)), quality=88)
        elapsed = time.time() - t0
        results.setdefault(name, {})[tag] = {
            "g": [gx, gy],
            "r": [rx, ry],
            "g_loss": round(gloss, 4),
            "r_loss": round(rloss, 4),
            "levels": levels,
            "seconds": round(elapsed, 2),
            "shape": list(b.shape),
        }
        print(f"{name:20s} {tag:9s} G=({gx:4d},{gy:4d})  R=({rx:4d},{ry:4d})  levels={levels}  {elapsed:.1f}s", flush=True)
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
