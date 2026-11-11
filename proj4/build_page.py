import html
import inspect
import json
from pathlib import Path

import auto
import features
import homography
import mosaic
import pick_points

R = json.loads(Path("media/results.json").read_text())
M, RECT = R["mosaics"], R["rectify"]
B = json.loads(Path("media/results_b.json").read_text())
BM, BC, ROT = B["mosaics"], B["corners"], B["rotation"]
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

def comparison_rows():
    rows = []
    for r in B["compare"]:
        rows.append(f"    <tr><td>{r['mosaic']}</td><td>{r['image']} → {r['reference']}</td>"
                    f"<td>{r['manual']['diff']}</td><td>{r['auto']['diff']}</td>"
                    f"<td>{r['manual']['corr']}</td><td>{r['auto']['corr']}</td><td>{r['gap_mean']} px</td></tr>")
    return "\n".join(rows)


def match_rows():
    rows = []
    for mosaic_name, m in BM.items():
        for n, p in m["pairs"].items():
            rows.append(f"    <tr><td>{mosaic_name}</td><td>{n} → {m['reference']}</td>"
                        f"<td>{p['matches']}</td><td>{p['inliers']}</td></tr>")
    return "\n".join(rows)


def pair_figures(prefix, caption):
    out = []
    for mosaic_name, m in BM.items():
        for n in m["pairs"]:
            out.append(fig(f"{prefix}_{mosaic_name}_{n}.jpg", f"{mosaic_name}: {n} → {m['reference']}. {caption}"))
    return out


def descriptor_tiles():
    return grid(4, *[fig(f"b_window_{k}.jpg", f"Feature {k + 1}: 40×40 window") for k in range(4)],
                *[fig(f"b_patch_{k}.jpg", f"Feature {k + 1}: 8×8 descriptor") for k in range(4)])


B_INTRO = """<section id="b0">
  <h2>Part B · Automatic stitching</h2>
  <p>Part A used points I picked by hand. Part B finds them automatically, following Brown, Szeliski and Winder, "Multi-Image Matching using Multi-Scale Oriented Patches", with the simplifications from the assignment: one scale, no sub-pixel accuracy, axis-aligned descriptors and no wavelet step. The pipeline for each pair of photos is:</p>
  <ol>
    <li>Find Harris corners in both photos.</li>
    <li>Keep an evenly spread subset with adaptive non-maximal suppression (ANMS).</li>
    <li>Describe each kept corner with a blurred, normalized 8×8 patch.</li>
    <li>Match patches between the photos, and keep a match only if it passes the nearest-neighbor ratio test.</li>
    <li>Run RANSAC on the matches to find a homography that most of them agree with.</li>
    <li>Warp and blend exactly as in Part A.</li>
  </ol>
  <p>Everything below uses the same quarter-size photos as Part A, and the same photo pairs, so the two parts compare directly.</p>
</section>"""

B1 = f"""<section id="b1">
  <h2>B.1 · Harris corners and ANMS</h2>
  <p>For corner detection I used the course's <code>harris.py</code>, which the assignment allows. It computes the Harris response at a single scale (σ = 1), keeps local maxima, and drops corners within 24 pixels of the border. On this photo of Wheeler Hall it finds {BC["harris"]:,} corners. Most of them are weak, and many sit on sky noise and leaves.</p>
  {fig("b_harris_IMG_4052.jpg", f"All {BC['harris']:,} Harris corners (red)")}
  <p>Matching needs corners that are spread over the whole photo. If you keep the 500 strongest, they pile up on the trees and the people. ANMS fixes this. For each corner it finds the radius to the nearest corner that is clearly stronger (the neighbor's response times 0.9 must beat the corner's own), and then it keeps the corners with the largest radii. A corner survives if it is either very strong or far from anything stronger.</p>
  {code(features.corners, features.anms)}
  {grid(2, fig("b_top500_IMG_4052.jpg", "The 500 strongest corners: clumped on the trees and people"),
        fig("b_anms500_IMG_4052.jpg", "500 corners chosen by ANMS: spread out"))}
  <p>Two choices differ from the paper. First, I keep {BC["n"]:,} corners per photo instead of 500. With 500 corners and a ratio of 0.6, I got only 13 to 42 matches per pair, and RANSAC was unstable: it recovered several homographies that were far off. With {BC["n"]:,} corners I get 100 to 932 matches per pair. Second, ANMS compares every corner with every stronger corner, which gets slow with tens of thousands of corners. So I run it on the 8,000 strongest corners only. The weaker ones are mostly noise. After ANMS, I drop corners within 30 pixels of the border, because their 40×40 windows would run off the photo, leaving {BC["anms"]:,} on this photo.</p>
  {fig("b_anms_IMG_4052.jpg", f"The {BC['anms']:,} corners I keep (yellow)")}
</section>"""

B2 = f"""<section id="b2">
  <h2>B.2 · Feature descriptors</h2>
  <p>Each descriptor starts with a 40×40 window around the corner. I blur the grayscale photo with a Gaussian (σ = 2.5), then sample an 8×8 grid from the window, 5 pixels apart. Sampling from the blurred, larger window makes the descriptor tolerant to small errors in where the corner sits. Last, I subtract the mean and divide by the standard deviation of the 64 values (bias/gain normalization), so a patch looks the same whether the photo is brighter or has more contrast.</p>
  {code(features.gray, features.normalize, features.descriptors)}
  {fig("b_desc_IMG_4052.jpg", "Eight corners chosen at random (cyan), with their 40×40 windows")}
  {descriptor_tiles()}
  <p>The top row shows each 40×40 window at the photo's own sharpness. The bottom row shows the normalized 8×8 descriptor made from it, stretched to full contrast. The descriptors keep the big light and dark shapes of the window and drop the fine detail.</p>
</section>"""

B3 = f"""<section id="b3">
  <h2>B.3 · Feature matching</h2>
  <p>For each descriptor in the first photo, I find its nearest and second-nearest descriptors in the second photo, by Euclidean distance. A true match should be much closer than the runner-up. A wrong match usually isn't: it is about as close as many other patches. So I keep a match only if the nearest distance is less than a set fraction of the second-nearest distance. This is Lowe's ratio test.</p>
  {code(features.match)}
  <p>To choose the threshold I did what Figure 6b of the paper does. For one pair (Wheeler, left and middle photos) I computed the ratio for every corner's best match. I called a match correct if it landed within 8 pixels of where my hand-picked homography from Part A predicts, and wrong if it was more than 30 pixels away. The plot shows both groups.</p>
  <figure class="narrow"><img src="media/b_ratio.png" alt="Histogram of the ratio for correct and wrong matches" loading="lazy"><figcaption>Correct matches (green) sit mostly below 0.7. Wrong matches (red) pile up near 1. The dashed line is my threshold, 0.7.</figcaption></figure>
  <p>Below 0.65 there is a single wrong match. From 0.65 to 0.7 a few appear (3 against 111 correct), and above 0.8 they outnumber the correct ones. I chose 0.7. It keeps about 70% of the correct matches and under 1% of the wrong ones, and RANSAC removes the few wrong ones that get through. The matches that pass for each pair are drawn below, before any outlier removal.</p>
  {grid(2, *pair_figures("b_matches", "All matches that pass the ratio test."))}
</section>"""

B4 = f"""<section id="b4">
  <h2>B.4 · RANSAC and automatic mosaics</h2>
  <p>Some matches are still wrong, and a least-squares fit pulls toward them. RANSAC avoids this. It repeats 3,000 times: pick 4 matches at random, compute the exact homography from them, and count how many of all the matches that homography maps to within 2 pixels of their partners (the inliers). The homography with the most inliers wins, and I refit it by least squares on all of its inliers, using the same <code>computeH</code> as Part A.</p>
  {code(features.ransac)}
  <p>In the figures, green lines are inliers and red lines are outliers that RANSAC threw away.</p>
  {grid(2, *pair_figures("b_ransac", "Green = inlier, red = outlier."))}
  <div class="tablewrap"><table>
    <tr><th>Mosaic</th><th>Pair</th><th>Matches</th><th>Inliers</th></tr>
{match_rows()}
  </table></div>
  <h3>Manual against automatic</h3>
  <p>Each row shows the Part A mosaic on the left and the automatic mosaic on the right. Both use the same photos, the same reference photo, and the same two-band blend.</p>
  {grid(2, fig("gate_mosaic.jpg", "Sather Gate, hand-picked points"), fig("b_auto_gate.jpg", "Sather Gate, automatic"))}
  {grid(2, fig("wheeler_mosaic.jpg", "Wheeler Hall, hand-picked points"), fig("b_auto_wheeler.jpg", "Wheeler Hall, automatic"))}
  {grid(2, fig("lamps_mosaic.jpg", "Lamp posts, hand-picked points"), fig("b_auto_lamps.jpg", "Lamp posts, automatic"))}
  <p>The two mosaics agree well on Wheeler Hall, where the automatic and hand-picked homographies for the left photo differ by only 0.7 pixels on average. They disagree on the other pairs, so I wanted a measurement and not just my eyes. For each pair I warped the photo into the reference with each homography, blurred both, and compared them over the area they share. Lower difference and higher correlation both mean a better alignment.</p>
  <div class="tablewrap"><table>
    <tr><th>Mosaic</th><th>Pair</th><th>Difference, manual</th><th>Difference, auto</th><th>Correlation, manual</th><th>Correlation, auto</th><th>H differs by</th></tr>
{comparison_rows()}
  </table></div>
  <p>The automatic homography is as good or better on every pair (the lamp pair is a tie) and clearly better on three of them. The biggest gap is the right photo of the gate. I had only 9 hand-picked points there, from a narrow overlap that is mostly dark trees. The automatic fit uses {BM["gate"]["pairs"]["IMG_4050"]["inliers"]} inliers, and it lines up the overlap much better (difference 0.054 against 0.115). The "H differs by" column is the mean distance, at my hand-picked points, between where the two homographies send them.</p>
  <p>Some of the remaining error is not a bug. I held the phone by hand, so the camera moved a little between shots. Nearby objects (the urns, the railing) shift more than far-away ones (the trees, the Campanile), and no single homography fits both. You can still see faint ghosts of people who walked between shots, in both versions. The overlap score includes those people too.</p>
</section>"""

B5 = f"""<section id="b5">
  <h2>B.5 · Bells and whistles: rotation invariance</h2>
  <p>I chose rotation invariance. The axis-aligned descriptor above fails if one photo is turned relative to the other, because the patch it samples is turned too. The fix is to give each corner a direction and sample the 40×40 window along it. I find the direction from the image gradient. I smooth the Sobel gradients with a Gaussian (σ = 4.5) and take the angle of the smoothed gradient at the corner. Then the 8×8 sample grid is rotated by that angle, so a turned photo yields the same patch.</p>
  {code(features.orientations, features.descriptors)}
  <p>To test it, I rotated the middle Wheeler photo by {ROT["degrees"]} degrees and matched it against the left Wheeler photo. I filled the empty corners of the rotated image with faint noise, and I dropped any corner near the fake border. The first figure shows the direction I found at some of the corners, drawn as cyan lines.</p>
  {fig("b_rot_orient.jpg", "Corner directions on the rotated photo (cyan lines)")}
  {grid(2, fig("b_rot_axis.jpg", f"Axis-aligned descriptors: {ROT['axis']['matches']} matches, {ROT['axis']['inliers']} inliers"),
        fig("b_rot_oriented.jpg", f"Oriented descriptors: {ROT['oriented']['matches']} matches, {ROT['oriented']['inliers']} inliers"))}
  <p>Without the direction, almost nothing matches: {ROT["axis"]["matches"]} matches pass the ratio test and only {ROT["axis"]["inliers"]} agree with a homography. With it, {ROT["oriented"]["matches"]} matches pass and {ROT["oriented"]["inliers"]} are inliers. The mosaics above use the plain axis-aligned descriptors, because my photos aren't rotated against each other, and the oriented version only matters when they are.</p>
</section>"""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cs180 project 4 · [auto]stitching photo mosaics</title>
<style>{CSS}</style>
</head>
<body>

<header>
  <p class="back"><a href="../">← All projects</a></p>
  <p class="eyebrow">CS180 · Fall 2026 · Project 4</p>
  <h1>[Auto]Stitching Photo Mosaics</h1>
  <p>Vikram Bhamre · UC Berkeley</p>
  <nav>
    <a href="#a1">A.1 Pictures</a>
    <a href="#a2">A.2 Homographies</a>
    <a href="#a3">A.3 Warping</a>
    <a href="#a4">A.4 Mosaics</a>
    <a href="#a5">A.5 Cylinders</a>
    <a href="#b1">B.1 Corners</a>
    <a href="#b2">B.2 Descriptors</a>
    <a href="#b3">B.3 Matching</a>
    <a href="#b4">B.4 RANSAC</a>
    <a href="#b5">B.5 Rotation</a>
  </nav>
</header>

<main>
{A1}

{A2}

{A3}

{A4}

{A5}

{B_INTRO}

{B1}

{B2}

{B3}

{B4}

{B5}
</main>

<footer>
  CS180 Project 4, parts A and B, October and November 2026. The homography, warping, interpolation, blending, ANMS, descriptor, matching and RANSAC code is my own (NumPy, plus SciPy for the Gaussian blur and the distance transform). The Harris detector is the course's <code>harris.py</code>, with its comments removed.
</footer>

</body>
</html>
"""

Path("index.html").write_text(PAGE)
