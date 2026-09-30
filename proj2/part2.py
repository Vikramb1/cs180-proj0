import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from align import align
from frequencies import blur, cutoff_frequency, high_pass, hybrid, log_spectrum, psnr, sharpen, to_gray
from stacks import blend, gaussian_stack, laplacian_stack

DATA, MEDIA = Path("data"), Path("media")
STARTER = DATA / "hybrid_starter/hybrid_python"
PROJ0, PROJ1 = Path("../proj0/media"), Path("../proj1/out")
SIGMA_HIGH, SIGMA_LOW = 4, 12

HYBRIDS = {
    "nutmeg_derek": (STARTER / "nutmeg.jpg", [(600, 285), (752, 368)], STARTER / "DerekPicture.jpg",
                     [(298, 345), (440, 332)], (640, 760), [(220, 300), (420, 300)]),
    "me_emir": (PROJ0 / "close-up-selfie.jpg", [(440, 757), (770, 762)], PROJ1 / "emir.jpg",
                [(1908, 967), (2003, 958)], (640, 760), [(240, 300), (400, 300)]),
    "library": (PROJ0 / "far-away-building.jpg", [(1055, 288), (1470, 212)], PROJ0 / "zoom-in-building.jpg",
                [(850, 303), (1262, 215)], (700, 700), [(200, 300), (615, 224)]),
}


def load(path):
    return iio.imread(path)[..., :3] / 255.0


def save(name, im):
    iio.imwrite(MEDIA / name, (np.clip(im, 0, 1) * 255).astype(np.uint8), quality=92)


def resize(im, width):
    return cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)


def signed_view(im):
    return 0.5 + im / (2 * np.percentile(np.abs(im), 99.5))


def strip(tiles, gap=6):
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, gap), (0, 0)), constant_values=1) for t in tiles]
    return np.concatenate(tiles, axis=1)[:, :-gap]


def gray3(im):
    return np.dstack([to_gray(im)] * 3)


def sharpening():
    out = {}
    library = resize(load(PROJ0 / "zoom-in-building.jpg"), 900)
    for name, im, sigma in [("taj", load(DATA / "taj.jpg"), 2.0), ("library", library, 3.0)]:
        low = blur(im, sigma)
        save(f"sharp_{name}_original.jpg", im)
        save(f"sharp_{name}_blur.jpg", low)
        save(f"sharp_{name}_high.jpg", signed_view(im - low))
        for amount in [0.5, 1, 2, 4]:
            save(f"sharp_{name}_a{amount}.jpg", sharpen(im, sigma, amount))
        out[name] = sigma

    blurred = blur(library, 2.0)
    amounts = [0.5, 1, 1.5, 2, 3, 4, 6]
    scores = {"blurred": psnr(blurred, library)}
    scores.update({str(a): psnr(sharpen(blurred, 2.0, a), library) for a in amounts})
    best = max(amounts, key=lambda a: scores[str(a)])
    crop = (slice(100, 260), slice(430, 730))
    zoom = lambda im: np.kron(im[crop], np.ones((2, 2, 1)))
    save("eval_original.jpg", zoom(library))
    save("eval_blurred.jpg", zoom(blurred))
    save("eval_best.jpg", zoom(sharpen(blurred, 2.0, best)))
    save("eval_strong.jpg", zoom(sharpen(blurred, 2.0, 6)))
    return {"sigmas": out, "psnr": scores, "best_amount": best}


def spectra(images):
    s = [log_spectrum(im) for im in images]
    lo, hi = min(np.percentile(x, 1) for x in s), max(x.max() for x in s)
    return [(x - lo) / (hi - lo) for x in s]


def hybrids():
    out = {}
    for name, (hp, hpts, lp, lpts, canvas, eyes) in HYBRIDS.items():
        high_raw, low_raw = load(hp), load(lp)
        high, low = align(high_raw, hpts, eyes, canvas), align(low_raw, lpts, eyes, canvas)
        result = hybrid(high, low, SIGMA_HIGH, SIGMA_LOW)
        save(f"hybrid_{name}_high_input.jpg", resize(high_raw, 600))
        save(f"hybrid_{name}_low_input.jpg", resize(low_raw, 600))
        save(f"hybrid_{name}.jpg", result)
        save(f"hybrid_{name}_scales.jpg", strip([resize(result, canvas[0] // 2 ** k) for k in range(4)]))
        out[name] = canvas[0]
        if name != "nutmeg_derek":
            continue
        save(f"hybrid_{name}_high_aligned.jpg", high)
        save(f"hybrid_{name}_low_aligned.jpg", low)
        gh, gl = to_gray(high), to_gray(low)
        filtered = high_pass(gh, SIGMA_HIGH), blur(gl, SIGMA_LOW)
        save(f"hybrid_{name}_high_filtered.jpg", signed_view(filtered[0]))
        save(f"hybrid_{name}_low_filtered.jpg", filtered[1])
        names = ["high_input", "low_input", "high_filtered", "low_filtered", "hybrid"]
        for n, view in zip(names, spectra([gh, gl, *filtered, to_gray(result)])):
            save(f"fft_{n}.jpg", view)
        variants = {"gray_both": (False, False), "color_low": (False, True),
                    "color_high": (True, False), "color_both": (True, True)}
        made = {k: hybrid(high, low, SIGMA_HIGH, SIGMA_LOW, *v) for k, v in variants.items()}
        for k, v in made.items():
            save(f"hybrid_{name}_{k}.jpg", v)
        out["color_diffs"] = {
            "high_only_vs_gray": float(np.abs(made["color_high"] - made["gray_both"]).mean()),
            "low_only_vs_gray": float(np.abs(made["color_low"] - made["gray_both"]).mean()),
            "both_vs_low_only": float(np.abs(made["color_both"] - made["color_low"]).mean()),
        }
    return out


def process_images(prefix, a, b, mask, levels, sigma):
    la, lb = laplacian_stack(a, levels, sigma), laplacian_stack(b, levels, sigma)
    gm = gaussian_stack(mask, levels, sigma)
    pa = [m[..., None] * x for m, x in zip(gm, la)]
    pb = [(1 - m[..., None]) * y for m, y in zip(gm, lb)]
    panels = []
    for k in (0, 2, levels - 1):
        for p in (pa[k], pb[k], pa[k] + pb[k]):
            panels.append(p if k == levels - 1 else 0.5 + p / (2 * np.percentile(np.abs(pa[k] + pb[k]), 99.5)))
    panels += [sum(pa), sum(pb), sum(pa) + sum(pb)]
    for letter, panel in zip("abcdefghijkl", panels):
        save(f"{prefix}_{letter}.jpg", resize(panel, 420) if panel.shape[1] > 420 else panel)


def scene(name, margin=0.07):
    im = load(PROJ1 / f"{name}.jpg")
    h, w = im.shape[:2]
    im = im[int(h * margin):h - int(h * margin), int(w * margin):w - int(w * margin)]
    return cv2.resize(im, (700, 560), interpolation=cv2.INTER_AREA)


def blends():
    apple, orange = load(DATA / "spline/apple.jpeg"), load(DATA / "spline/orange.jpeg")
    mask = np.zeros(apple.shape[:2])
    mask[:, :150] = 1
    for name, im in [("apple", apple), ("orange", orange)]:
        save(f"stack_{name}_gaussian.jpg", strip(gaussian_stack(im, 5, 2.0)))
        laplacian = laplacian_stack(im, 5, 2.0)
        save(f"stack_{name}_laplacian.jpg", strip([signed_view(x) for x in laplacian[:-1]] + [laplacian[-1]]))
    save("stack_mask_gaussian.jpg", strip(gaussian_stack(np.dstack([mask] * 3), 5, 2.0)))
    save("oraple.jpg", blend(apple, orange, mask, 5, 2.0))
    save("oraple_gray.jpg", blend(gray3(apple), gray3(orange), mask, 5, 2.0))
    process_images("fig342", apple, orange, mask, 5, 2.0)
    error = float(np.abs(sum(laplacian_stack(apple, 5, 2.0)) - apple).max())

    church, wharf = scene("church"), scene("wharf")
    seam = np.zeros(church.shape[:2])
    seam[:int(0.62 * 560)] = 1
    save("blend_dock_a.jpg", church)
    save("blend_dock_b.jpg", wharf)
    save("blend_dock_mask.jpg", np.dstack([seam] * 3))
    save("blend_dock.jpg", blend(church, wharf, seam, 6, 2.0))

    ox, oy = 1450, 600
    emir = load(PROJ1 / "emir.jpg")[oy:oy + 800, ox:ox + 1000]
    me = align(load(PROJ0 / "close-up-selfie.jpg"), [(440, 757), (770, 762)],
               [(1908 - ox, 967 - oy), (2003 - ox, 958 - oy)], (1000, 800), cv2.BORDER_CONSTANT)
    yy, xx = np.mgrid[:800, :1000]
    face = (((xx - 505) / 84) ** 2 + ((yy - 403) / 88) ** 2 < 1).astype(float)
    save("blend_face_a.jpg", me)
    save("blend_face_b.jpg", emir)
    save("blend_face_mask.jpg", np.dstack([face] * 3))
    save("blend_face.jpg", blend(me, emir, face, 6, 2.0))
    process_images("swap", me, emir, face, 6, 2.0)
    return error


def main():
    MEDIA.mkdir(exist_ok=True)
    results = {"sharpening": sharpening(), "hybrids": hybrids(), "sigma_high": SIGMA_HIGH, "sigma_low": SIGMA_LOW,
               "cutoff_high": cutoff_frequency(SIGMA_HIGH), "cutoff_low": cutoff_frequency(SIGMA_LOW)}
    results["reconstruction_error"] = blends()
    (MEDIA / "part2_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
