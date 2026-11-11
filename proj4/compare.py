import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from features import gray
from homography import computeH, warpImageBilinear
from images import load

OUT = Path("media")


def overlap_score(name, ref, H):
    im, target = load(name), load(ref)
    box = (0, 0, target.shape[1], target.shape[0])
    warped, alpha, _ = warpImageBilinear(im, H, box)
    inside = alpha > 0.99
    a = gaussian_filter(gray(warped), 2)[inside]
    b = gaussian_filter(gray(target), 2)[inside]
    return {"diff": round(float(np.abs(a - b).mean()), 4), "corr": round(float(np.corrcoef(a, b)[0, 1]), 3)}


def compare(results):
    rows = []
    for mosaic, info in results["mosaics"].items():
        ref = info["reference"]
        for name, pair in info["pairs"].items():
            pts = json.loads(Path(f"points/{name}_{ref}.json").read_text())
            manual = computeH(pts[name], pts[ref])
            rows.append({"mosaic": mosaic, "image": name, "reference": ref,
                         "manual": overlap_score(name, ref, manual),
                         "auto": overlap_score(name, ref, np.array(pair["H"])),
                         "gap_mean": pair["gap_mean"]})
    return rows


def ratio_plot(hist):
    bins = np.array(hist["bins"])[:-1]
    width = 0.05
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(bins, hist["correct"], width=width * 0.9, align="edge", color="#2a7a4b", label="correct matches")
    ax.bar(bins, hist["wrong"], width=width * 0.9, align="edge", color="#b3412f", alpha=0.85, label="wrong matches")
    ax.axvline(0.7, color="#1b1b1b", linestyle="--", linewidth=1)
    ax.set_xlabel("ratio of nearest to second-nearest distance")
    ax.set_ylabel("number of matches")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "b_ratio.png", dpi=130)


if __name__ == "__main__":
    results = json.loads((OUT / "results_b.json").read_text())
    results["compare"] = compare(results)
    (OUT / "results_b.json").write_text(json.dumps(results, indent=1))
    ratio_plot(results["ratio_hist"])
    for row in results["compare"]:
        print(row)
