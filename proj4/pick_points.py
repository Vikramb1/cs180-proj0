import json
import sys
from pathlib import Path

import numpy as np

from homography import apply_h, computeH, warpImageBilinear
from images import load

RAW, SNAPPED = Path("points/raw"), Path("points")


def click(name, n):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 12))
    plt.imshow(load(name))
    plt.title(f"{name}: click {n} points, in order")
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return [[round(x), round(y)] for x, y in pts]


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)


def snap(im1, im2, p, guess, half=12, search=30):
    g1, g2 = im1.mean(2), im2.mean(2)
    x, y = p
    patch = g1[y - half:y + half + 1, x - half:x + half + 1]
    best, where = -2, guess
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            u, v = guess[0] + dx, guess[1] + dy
            window = g2[v - half:v + half + 1, u - half:u + half + 1]
            if window.shape == patch.shape:
                score = ncc(patch, window)
                if score > best:
                    best, where = score, [u, v]
    return where, best


def snap_pair(a, b):
    raw = json.loads((RAW / f"{a}_{b}.json").read_text())
    im1, im2 = load(a), load(b)
    pts1, pts2 = raw[a], raw[b]
    box = (0, 0, im2.shape[1], im2.shape[0])
    for search in (30, 8, 8):
        H = computeH(pts1, pts2)
        moved = warpImageBilinear(im1, H, box)[0]
        centers = np.round(apply_h(H, pts1)).astype(int)
        start = pts2 if search == 30 else centers
        found = [snap(moved, im2, c, list(s), search=search) for c, s in zip(centers, start)]
        pts2 = [[int(q[0]), int(q[1])] for q, _ in found]
        scores = [round(float(s), 3) for _, s in found]
    (SNAPPED / f"{a}_{b}.json").write_text(json.dumps({a: pts1, b: pts2, "ncc": scores}))
    return pts1, pts2, scores


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        n = int(sys.argv[3])
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f"{a}_{b}.json").write_text(json.dumps({a: click(a, n), b: click(b, n)}))
    for p, q, s in zip(*snap_pair(a, b)):
        print(p, q, s)
