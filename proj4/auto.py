import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from features import anms, corners, descriptors, gray, keep_inside, match, orientations, ransac
from homography import apply_h, computeH, warpImageBilinear
from images import load, save
from main import MOSAICS
from mosaic import feather_weights, two_band, warp_all

OUT = Path("media")
N, RATIO = 3000, 0.7
YELLOW, RED, GREEN, CYAN = (255, 230, 0), (255, 60, 60), (60, 230, 90), (0, 220, 255)


def to_pil(im):
    return Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))


def dots(im, pts, color, r=3):
    pil = to_pil(im)
    d = ImageDraw.Draw(pil)
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return pil


def side_by_side(im1, im2, p1, p2, colors):
    a, b = to_pil(im1), to_pil(im2)
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    d = ImageDraw.Draw(canvas)
    for (x1, y1), (x2, y2), c in zip(p1, p2, colors):
        d.line([x1, y1, x2 + a.width, y2], fill=c, width=2)
        d.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], outline=c, width=2)
        d.ellipse([x2 + a.width - 4, y2 - 4, x2 + a.width + 4, y2 + 4], outline=c, width=2)
    return canvas


def image(name, cache):
    return cache[("image", name)] if ("image", name) in cache else load(name)


def features(name, cache, rotate=False):
    if (name, rotate) not in cache:
        im = image(name, cache)
        pts, strength = corners(im)
        if ("valid", name) in cache:
            xi, yi = pts.astype(int).T
            keep = cache[("valid", name)][yi, xi]
            pts, strength = pts[keep], strength[keep]
        picked, _ = anms(pts, strength, N)
        picked = keep_inside(im, picked)
        angles = orientations(im, picked) if rotate else None
        cache[(name, rotate)] = (pts, strength, picked, descriptors(im, picked, angles), angles)
    return cache[(name, rotate)]


def corner_figures(name, cache):
    im = load(name)
    pts, strength, picked, desc, _ = features(name, cache)
    dots(im, pts, RED, 1).save(OUT / f"b_harris_{name}.jpg", quality=90)
    top = pts[np.argsort(-strength)[:500]]
    dots(im, top, YELLOW).save(OUT / f"b_top500_{name}.jpg", quality=90)
    dots(im, anms(pts, strength, 500)[0], YELLOW).save(OUT / f"b_anms500_{name}.jpg", quality=90)
    dots(im, picked, YELLOW, 2).save(OUT / f"b_anms_{name}.jpg", quality=90)
    return len(pts), len(picked)


def descriptor_figure(name, cache, count=8):
    im = load(name)
    _, _, picked, desc, _ = features(name, cache)
    rng = np.random.default_rng(3)
    chosen = rng.choice(len(picked), count, replace=False)
    g = gray(im)
    tiles = []
    for i in chosen:
        x, y = np.round(picked[i]).astype(int)
        window = g[y - 20:y + 20, x - 20:x + 20]
        patch = desc[i].reshape(8, 8)
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        tiles.append((np.kron(window, np.ones((4, 4))), np.kron(patch, np.ones((20, 20)))))
    for k, (window, patch) in enumerate(tiles):
        save(OUT / f"b_window_{k}.jpg", np.repeat(window[..., None], 3, 2))
        save(OUT / f"b_patch_{k}.jpg", np.repeat(patch[..., None], 3, 2))
    box = dots(im, picked[chosen], CYAN, 6)
    d = ImageDraw.Draw(box)
    for k, (x, y) in enumerate(picked[chosen]):
        d.rectangle([x - 20, y - 20, x + 20, y + 20], outline=CYAN, width=2)
        d.text((x + 23, y - 20), str(k + 1), fill=CYAN, stroke_width=2, stroke_fill="black")
    box.save(OUT / f"b_desc_{name}.jpg", quality=90)


def pair(a, b, cache, rotate=False):
    _, _, p1, d1, _ = features(a, cache, rotate)
    _, _, p2, d2, _ = features(b, cache, rotate)
    pairs, ratios = match(d1, d2, RATIO)
    m1, m2 = p1[pairs[:, 0]], p2[pairs[:, 1]]
    H, inliers = ransac(m1, m2)
    return H, m1, m2, inliers


def manual_h(a, b):
    d = json.loads(Path(f"points/{a}_{b}.json").read_text())
    return computeH(d[a], d[b]), np.array(d[a], float)


def ratio_figure(a, b, cache):
    _, _, p1, d1, _ = features(a, cache)
    _, _, p2, d2, _ = features(b, cache)
    pairs, ratios = match(d1, d2, 1.01)
    H, _ = manual_h(a, b)
    err = np.linalg.norm(apply_h(H, p1[pairs[:, 0]]) - p2[pairs[:, 1]], axis=1)
    correct = ratios[pairs[:, 0]][err < 8]
    wrong = ratios[pairs[:, 0]][err > 30]
    bins = np.linspace(0, 1, 21)
    return {"bins": bins.tolist(), "correct": np.histogram(correct, bins)[0].tolist(),
            "wrong": np.histogram(wrong, bins)[0].tolist()}


def run_mosaic(name, names, ref, cache, results):
    images = [load(n) for n in names]
    Hs, info = [], {}
    for n, im in zip(names, images):
        if n == ref:
            Hs.append(np.eye(3))
            continue
        start = time.perf_counter()
        H, m1, m2, inliers = pair(n, ref, cache)
        seconds = time.perf_counter() - start
        Hm, pm = manual_h(n, ref)
        gap = np.linalg.norm(apply_h(H, pm) - apply_h(Hm, pm), axis=1)
        colors = [GREEN if k else RED for k in inliers]
        side_by_side(im, images[names.index(ref)], m1, m2, [YELLOW] * len(m1)).save(OUT / f"b_matches_{name}_{n}.jpg", quality=88)
        side_by_side(im, images[names.index(ref)], m1, m2, colors).save(OUT / f"b_ransac_{name}_{n}.jpg", quality=88)
        info[n] = {"H": H.round(6).tolist(), "matches": int(len(m1)), "inliers": int(inliers.sum()),
                   "gap_mean": round(float(gap.mean()), 1), "gap_max": round(float(gap.max()), 1)}
        Hs.append(H)
    warped, alphas, box = warp_all(images, Hs)
    save(OUT / f"b_auto_{name}.jpg", two_band(warped, alphas, feather_weights(alphas)))
    results["mosaics"][name] = {"images": names, "reference": ref, "pairs": info, "size": [int(v) for v in box[2:]]}


def rotation_demo(cache, results, degrees=45):
    a, b = "IMG_4051", "IMG_4052"
    im = load(b)
    h, w = im.shape[:2]
    t = np.radians(degrees)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    center = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])
    H = center @ R @ np.linalg.inv(center)
    turned, alpha, _ = warpImageBilinear(im, H)
    noise = np.random.default_rng(0).normal(0.5, 0.02, turned.shape)
    turned = np.where(alpha[..., None] > 0.99, turned, noise)
    cache[("image", b + "_rot")] = turned
    cache[("valid", b + "_rot")] = binary_erosion(alpha > 0.99, iterations=30)
    rows = {}
    for rotate in (False, True):
        Hr, m1, m2, inliers = pair(a, b + "_rot", cache, rotate)
        colors = [GREEN if k else RED for k in inliers]
        side_by_side(load(a), turned, m1, m2, colors).save(
            OUT / f"b_rot_{'oriented' if rotate else 'axis'}.jpg", quality=88)
        rows["oriented" if rotate else "axis"] = {"matches": int(len(m1)), "inliers": int(inliers.sum())}
    _, _, picked, _, angles = features(b + "_rot", cache, True)
    pil = dots(turned, picked[::12], YELLOW, 3)
    d = ImageDraw.Draw(pil)
    for (x, y), ang in zip(picked[::12], angles[::12]):
        d.line([x, y, x + 18 * np.cos(ang), y + 18 * np.sin(ang)], fill=CYAN, width=2)
    pil.save(OUT / "b_rot_orient.jpg", quality=88)
    results["rotation"] = {"degrees": degrees, **rows}


if __name__ == "__main__":
    cache, results = {}, {"mosaics": {}}
    def save_results():
        (OUT / "results_b.json").write_text(json.dumps(results, indent=1))

    harris, kept = corner_figures("IMG_4052", cache)
    results["corners"] = {"harris": harris, "anms": kept, "n": N, "ratio": RATIO}
    descriptor_figure("IMG_4052", cache)
    results["ratio_hist"] = ratio_figure("IMG_4051", "IMG_4052", cache)
    save_results()
    for name, (names, ref) in MOSAICS.items():
        run_mosaic(name, names, ref, cache, results)
        save_results()
        print(name, results["mosaics"][name]["pairs"], flush=True)
    rotation_demo(cache, results)
    save_results()
    print(results["rotation"], flush=True)
