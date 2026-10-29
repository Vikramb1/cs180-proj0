import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from homography import computeH, system, warpImageBilinear, warpImageNearestNeighbor, apply_h
from images import load, save
from mosaic import cylinder, feather, feather_weights, overwrite, two_band, warp_all

OUT = Path("media")
MOSAICS = {
    "gate": (["IMG_4048", "IMG_4049", "IMG_4050"], "IMG_4049"),
    "wheeler": (["IMG_4051", "IMG_4052", "IMG_4053"], "IMG_4052"),
    "lamps": (["IMG_4057", "IMG_4058"], "IMG_4057"),
}
RECTIFY = json.loads(Path("points/rectify.json").read_text())
FOCAL = 26 / np.hypot(36, 24)
SEAMS = {"gate": (1300, 700, 600), "wheeler": (2350, 1150, 600)}


def pair_points(a, b):
    d = json.loads(Path(f"points/{a}_{b}.json").read_text())
    return np.array(d[a], float), np.array(d[b], float)


def draw_points(im, pts, color):
    pil = Image.fromarray((im * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for i, (x, y) in enumerate(pts):
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=color, width=3)
        d.text((x + 9, y - 9), str(i + 1), fill=color, stroke_width=2, stroke_fill="black")
    return np.asarray(pil) / 255


def correspondences(name, a, b):
    pa, pb = pair_points(a, b)
    save(OUT / f"{name}_{a}_pts.jpg", draw_points(load(a), pa, (255, 230, 0)))
    save(OUT / f"{name}_{b}_pts.jpg", draw_points(load(b), pb, (255, 230, 0)))


def run_mosaic(name, names, ref, results):
    images = [load(n) for n in names]
    Hs, info = [], {}
    for n in names:
        if n == ref:
            Hs.append(np.eye(3))
            continue
        pa, pb = pair_points(n, ref)
        H = computeH(pa, pb)
        A, b = system(pa, pb)
        err = np.linalg.norm(apply_h(H, pa) - pb, axis=1)
        Hs.append(H)
        info[n] = {"H": H.round(6).tolist(), "A": A.round(2).tolist(), "b": b.round(2).tolist(),
                   "n": len(pa), "err_mean": round(float(err.mean()), 2), "err_max": round(float(err.max()), 2)}
        correspondences(name, n, ref)
    for n, im in zip(names, images):
        save(OUT / f"{name}_src_{n}.jpg", im)
    warped, alphas, box = warp_all(images, Hs)
    weights = feather_weights(alphas)
    blends = {"overwrite": overwrite(warped, alphas), "feather": feather(warped, weights),
              "mosaic": two_band(warped, alphas, weights)}
    for label, im in blends.items():
        save(OUT / f"{name}_{label}.jpg", im)
        if name in SEAMS:
            x, y, s = SEAMS[name]
            save(OUT / f"{name}_{label}_seam.jpg", im[y:y + s, x:x + s])
    for n, w in zip(names, warped):
        save(OUT / f"{name}_warped_{n}.jpg", w)
    tint = np.eye(3)[[0, 2, 1]]
    save(OUT / f"{name}_weights.jpg", sum((k / k.max())[..., None] * tint[i] for i, k in enumerate(weights)))
    shape = images[names.index(ref)].shape
    f = FOCAL * np.hypot(*shape[:2])
    center = (shape[1] / 2, shape[0] / 2)
    cw, ca = cylinder(images, Hs, f, center)
    save(OUT / f"{name}_cylinder.jpg", two_band(cw, ca, feather_weights(ca)))
    results["mosaics"][name] = {"images": names, "reference": ref, "pairs": info, "size": [int(v) for v in box[2:]],
                                "focal": round(float(f), 1)}


def rectify(name, spec, results):
    im = load(spec["image"])
    src = np.array(spec["corners"], float)
    w, h = spec["size"]
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) + src.min(0)
    H = computeH(src, dst)
    times = {}
    for label, fn in (("nn", warpImageNearestNeighbor), ("bilinear", warpImageBilinear)):
        start = time.perf_counter()
        out, alpha, box = fn(im, H)
        times[label] = round(time.perf_counter() - start, 3)
        save(OUT / f"rect_{name}_{label}.jpg", out)
        x0, y0 = int(dst[0, 0] - box[0]), int(dst[0, 1] - box[1])
        save(OUT / f"rect_{name}_{label}_crop.jpg", out[y0:y0 + int(h), x0:x0 + int(w)])
        zx, zy, zs = spec["zoom"]
        zx, zy = int(zx - box[0]), int(zy - box[1])
        patch = out[zy:zy + zs, zx:zx + zs]
        save(OUT / f"rect_{name}_{label}_zoom.jpg", np.kron(patch, np.ones((6, 6, 1))))
    save(OUT / f"rect_{name}_src.jpg", draw_points(im, src, (255, 230, 0)))
    results["rectify"][name] = {"image": spec["image"], "H": H.round(6).tolist(), "times": times,
                                "corners": src.tolist(), "target": dst.tolist(),
                                "pixels": int(np.prod(box[2:]))}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = {"mosaics": {}, "rectify": {}}
    for name, spec in RECTIFY.items():
        rectify(name, spec, results)
    for name, (names, ref) in MOSAICS.items():
        run_mosaic(name, names, ref, results)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
