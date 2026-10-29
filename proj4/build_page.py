import html
import inspect
import json
from pathlib import Path

import homography
import mosaic
import pick_points

R = json.loads(Path("media/results.json").read_text())
M, RECT = R["mosaics"], R["rectify"]
CSS = Path("../proj2/build_page.py").read_text().split('CSS = """')[1].split('"""')[0] + """
  .matrix { display: inline-block; margin-right: 24px; }
"""


def code(*objs):
    src = "\n\n".join(inspect.getsource(o).rstrip() for o in objs)
    return f"<pre><code>{html.escape(src)}</code></pre>"


def fig(name, caption):
    return f'<figure><img src="media/{name}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def grid(cols, *figs):
    return f'<div class="grid c{cols}">\n' + "\n".join(figs) + "\n</div>"


def matrix(H):
    rows = ["  ".join(f"{v:>11.5g}" for v in row) for row in H]
    return "<pre class=\"matrix\"><code>" + "\n".join(rows) + "</code></pre>"


def sources(name):
    return grid(len(M[name]["images"]), *[fig(f"{name}_src_{n}.jpg", n + (" (reference)" if n == M[name]["reference"] else ""))
                                          for n in M[name]["images"]])


def pairs(name):
    ref = M[name]["reference"]
    out = []
    for n, p in M[name]["pairs"].items():
        out.append(f"<h3>{n} → {ref}</h3>")
        out.append(grid(2, fig(f"{name}_{n}_pts.jpg", f"{n}, {p['n']} points"), fig(f"{name}_{ref}_pts.jpg", ref)))
        out.append(f"<p>Recovered H, mapping {n} into {ref}. Mean leftover error {p['err_mean']} px, largest {p['err_max']} px.</p>")
        out.append(matrix(p["H"]))
    return "\n".join(out)


def system_text(p):
    lines = []
    for row, rhs in zip(p["A"], p["b"]):
        lines.append("[" + " ".join(f"{v:>9.0f}" for v in row) + " ]   [" + f"{rhs:>6.0f}" + " ]")
    return "<pre><code>" + "\n".join(lines) + "</code></pre>"


def rect_block(name, title):
    r = RECT[name]
    return f"""<h3>{title}</h3>
  {grid(3, fig(f"rect_{name}_src.jpg", f"{r['image']} with the four corners I picked"),
        fig(f"rect_{name}_nn.jpg", f"Nearest neighbor, {r['times']['nn']} s"),
        fig(f"rect_{name}_bilinear.jpg", f"Bilinear, {r['times']['bilinear']} s"))}
  {grid(4, fig(f"rect_{name}_nn_crop.jpg", "Rectified, nearest neighbor"),
        fig(f"rect_{name}_bilinear_crop.jpg", "Rectified, bilinear"),
        fig(f"rect_{name}_nn_zoom.jpg", "Zoom, nearest neighbor"),
        fig(f"rect_{name}_bilinear_zoom.jpg", "Zoom, bilinear"))}
  {matrix(r["H"])}"""


def mosaic_block(name, title, note):
    m = M[name]
    width, height = m["size"]
    return f"""<h3>{title}</h3>
  <p>{note}</p>
  {sources(name)}
  {fig(f"{name}_mosaic.jpg", f"Mosaic, {width}×{height}, two-band blend")}"""


LAMP = M["lamps"]["pairs"]["IMG_4058"]

A1 = f"""<section id="a1">
  <h2>A.1 · Shooting the pictures</h2>
  <p>I shot everything on an iPhone main camera (26 mm equivalent), handheld, turning the phone in place between shots. Neighboring shots overlap by roughly 20 to 50 percent. Here are the three sets I used for mosaics.</p>
  <h3>Sather Gate</h3>
  {sources("gate")}
  <h3>Wheeler Hall and the Campanile</h3>
  {sources("wheeler")}
  <h3>Lamp posts by Doe</h3>
  {sources("lamps")}
  <p>The lamp set started as three photos, shot at a diagonal angle on purpose. The first one (IMG_4056) shares only a thin band of the building with the next shot, and the railing in front sits about a meter from the camera. Small hand movements shift that railing much more than the building behind it, so no single homography fits both. The fit from that thin band blew up the warp, so I left that photo out and kept the two that share the lamp and the tower.</p>
</section>"""

A2 = f"""<section id="a2">
  <h2>A.2 · Recovering homographies</h2>
  <p>A homography maps a point (x, y) in one image to (u, v) in another:</p>
  <pre><code>u = (h1 x + h2 y + h3) / (h7 x + h8 y + 1)
v = (h4 x + h5 y + h6) / (h7 x + h8 y + 1)</code></pre>
  <p>Multiplying out the denominator makes each equation linear in the eight unknowns. So each point pair gives two rows of a system Ah = b:</p>
  <pre><code>[ x  y  1  0  0  0  -u·x  -u·y ] h = u
[ 0  0  0  x  y  1  -v·x  -v·y ] h = v</code></pre>
  <p>With more than four pairs the system is overdetermined, and I solve it with least squares. h9 is fixed to 1.</p>
  {code(homography.system, homography.computeH, homography.apply_h)}
  <h3>Picking the points</h3>
  <p>I picked every point by hand. For each pair I looked at zoomed crops of the overlap with a coordinate grid drawn on, chose features that don't move and are easy to find in both images (corners of buildings and signs, the lamp heads, the spire, the urns), and wrote down where each one sits in both photos. Then a small refinement snaps the second image's point to the exact pixel. It warps the first image by the current H, compares a 25×25 patch around the feature with normalized cross-correlation in a small window, refits H, and repeats. Points on people were never used, since people walked between shots. <code>pick_points.py</code> also has a matplotlib <code>ginput</code> mode for clicking points directly.</p>
  {code(pick_points.ncc, pick_points.snap)}
  <h3>Correspondences and recovered matrices</h3>
  <p>Each mosaic warps the outer images into the middle (reference) image.</p>
  {pairs("gate")}
  {pairs("wheeler")}
  {pairs("lamps")}
  <p>The leftover error is a few pixels, not a fraction of one, because I held the phone by hand. The lens moved a little between shots, so nearby things (the urns, the planter, the lamp) shift by a different amount than faraway things (the trees, the Campanile). The least-squares fit splits the difference.</p>
  <h3>The system for IMG_4058 → IMG_4057</h3>
  <p>The full system Ah = b for the lamp pair, {LAMP["n"]} points and {2 * LAMP["n"]} equations. Coordinates are in pixels of the quarter-size images.</p>
  {system_text(LAMP)}
</section>"""

A3 = f"""<section id="a3">
  <h2>A.3 · Warping and rectification</h2>
  <p>Both warps use inverse warping. I push the four image corners through H to get the output bounding box. Then, for every output pixel, H<sup>-1</sup> gives the spot in the source image to sample from. Pixel centers sit at integer coordinates. Output pixels that land outside the source get alpha 0, and that alpha later serves as the mask for blending.</p>
  {code(homography.bounds, homography.target_grid, homography.inside, homography.sample_nn, homography.sample_bilinear,
        homography.warp, homography.warpImageNearestNeighbor, homography.warpImageBilinear)}
  <p>For rectification there is only one image, so the target points come from what I know about the object. I picked the four corners of something rectangular and sent them to an upright rectangle with the object's proportions.</p>
  {rect_block("poster", "Recycling bin poster")}
  <p>The recycling logo comes out round and the text lines come out level, so the poster is now seen straight on.</p>
  {rect_block("sign", "Room 190J sign")}
  <p>I shot this one at an angle with the 5× lens (121 mm equivalent). After the warp, the sign is a rectangle and the "Slide" text under it is level as well, since it lies on the same glass plane.</p>
  <h3>Nearest neighbor against bilinear</h3>
  <div class="tablewrap"><table>
    <tr><th>Image</th><th>Output pixels</th><th>Nearest neighbor</th><th>Bilinear</th></tr>
    <tr><td>Poster</td><td>{RECT["poster"]["pixels"]:,}</td><td>{RECT["poster"]["times"]["nn"]} s</td><td>{RECT["poster"]["times"]["bilinear"]} s</td></tr>
    <tr><td>Sign</td><td>{RECT["sign"]["pixels"]:,}</td><td>{RECT["sign"]["times"]["nn"]} s</td><td>{RECT["sign"]["times"]["bilinear"]} s</td></tr>
  </table></div>
  <p>Nearest neighbor copies one source pixel per output pixel. It is about two to three times faster here, but slanted edges turn into stair steps. You can see this on the letters of "Berkeley" and on the 9 and 0. Bilinear reads four pixels and mixes them by distance. It costs a few more array operations per pixel, and the edges come out smooth. Both are fully vectorized with NumPy, so even bilinear takes well under a second at this size. Bilinear softens the image slightly where the warp stretches it, but I used it for every mosaic because stair steps show up much more than the slight blur.</p>
</section>"""

A4 = f"""<section id="a4">
  <h2>A.4 · Blending into a mosaic</h2>
  <p>The procedure for each mosaic:</p>
  <ol>
    <li>Keep the middle photo fixed and compute H from each other photo into it.</li>
    <li>Push every image's corners through its H to find the bounding box of the whole mosaic.</li>
    <li>Warp every image into that box with bilinear inverse warping. Each warp also returns its alpha mask.</li>
    <li>Give each image a weight map: the distance from each pixel to the edge of its own mask. The weight is highest in the middle of each photo and falls to 0 at its border.</li>
    <li>Blend in two bands. The low frequencies (Gaussian blur, σ = 4) are averaged with the weights, so exposure differences fade across the whole overlap. The high frequencies come only from the image with the largest weight at each pixel, so fine detail stays sharp instead of doubling.</li>
  </ol>
  {code(mosaic.warp_all, mosaic.feather_weights, mosaic.feather, mosaic.two_band)}
  <h3>Why two bands</h3>
  {grid(3, fig("gate_overwrite_seam.jpg", "Overwrite: hard seam and a jump in brightness"),
        fig("gate_feather_seam.jpg", "Weighted average: smooth, but people and leaves ghost"),
        fig("gate_mosaic_seam.jpg", "Two-band: smooth tone, sharp detail"))}
  {grid(3, fig("wheeler_overwrite_seam.jpg", "Overwrite"),
        fig("wheeler_feather_seam.jpg", "Weighted average"),
        fig("wheeler_mosaic_seam.jpg", "Two-band"))}
  {grid(2, fig("gate_weights.jpg", "Sather Gate weight maps, one color per image"),
        fig("wheeler_weights.jpg", "Wheeler weight maps"))}
  <p>Plain overwriting leaves a visible line wherever the exposure changed between shots. A weighted average hides the line, but anything that doesn't line up exactly (people who walked, branches in the wind, the parallax above) shows up twice, faintly. Two bands keep the smooth tone of the average and pick the detail from one image. Some ghosting is still visible where a person walked across the seam itself.</p>
  <h3>Results</h3>
  {mosaic_block("gate", "Sather Gate", "Three photos warped into the middle one.")}
  {mosaic_block("wheeler", "Wheeler Hall and the Campanile", "Three photos covering about 115 degrees. The edges stretch a lot on a flat plane, which is what A.5 fixes.")}
  {mosaic_block("lamps", "Lamp posts by Doe", "Two photos shot at a diagonal angle, warped into the first one.")}
</section>"""

A5 = f"""<section id="a5">
  <h2>A.5 · Bells and whistles: cylindrical mosaics</h2>
  <p>A flat mosaic stretches without bound as the view widens. Instead, I project onto a cylinder around the camera. The focal length comes from the photo metadata: 26 mm equivalent on a 43.3 mm full-frame diagonal, which is {M["wheeler"]["focal"]} px for the quarter-size images. That also sets the cylinder radius.</p>
  <p>The sampling is fully inverse. Each output pixel (θ, h) on the cylinder is a ray, which hits the reference image plane at x = f tan θ + cx, y = h / cos θ + cy. That point goes through H<sup>-1</sup> into each original photo, which is then sampled with bilinear interpolation. So every pixel is read straight from the original photos, never from an already-warped mosaic. Blending is the same two-band blend as above.</p>
  {code(mosaic.cylinder)}
  {grid(2, fig("wheeler_mosaic.jpg", "Wheeler, planar"), fig("wheeler_cylinder.jpg", "Wheeler, cylindrical"))}
  {grid(2, fig("gate_mosaic.jpg", "Sather Gate, planar"), fig("gate_cylinder.jpg", "Sather Gate, cylindrical"))}
  {grid(2, fig("lamps_mosaic.jpg", "Lamps, planar"), fig("lamps_cylinder.jpg", "Lamps, cylindrical"))}
  <p>The Wheeler panorama gains the most. The plaza no longer fans out at the edges, and the lamp posts on the left and right keep a sensible size. The price is that straight horizontal lines bend into curves, which is easiest to see on the lamp set, where the camera was tilted.</p>
</section>"""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cs180 project 4 · image warping and mosaicing</title>
<style>{CSS}</style>
</head>
<body>

<header>
  <p class="back"><a href="../">← All projects</a></p>
  <p class="eyebrow">CS180 · Fall 2026 · Project 4A</p>
  <h1>Image Warping and Mosaicing</h1>
  <p>Vikram Bhamre · UC Berkeley</p>
  <nav>
    <a href="#a1">A.1 Pictures</a>
    <a href="#a2">A.2 Homographies</a>
    <a href="#a3">A.3 Warping</a>
    <a href="#a4">A.4 Mosaics</a>
    <a href="#a5">A.5 Cylinders</a>
  </nav>
</header>

<main>
{A1}

{A2}

{A3}

{A4}

{A5}
</main>

<footer>
  CS180 Project 4A, October 2026. All homography, warping, interpolation and blending code is my own (NumPy, plus SciPy for the Gaussian blur and the distance transform).
</footer>

</body>
</html>
"""

Path("index.html").write_text(PAGE)
