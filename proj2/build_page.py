import html
import inspect
import json
from pathlib import Path

import filters

R = json.loads(Path("media/part1_results.json").read_text())
P11, P12, P13 = R["part_1_1"], R["part_1_2"], R["part_1_3"]
Q = json.loads(Path("media/part2_results.json").read_text())
SH, HY = Q["sharpening"], Q["hybrids"]


def code(*objs):
    src = "\n\n".join(inspect.getsource(o).rstrip() for o in objs)
    return f"<pre><code>{html.escape(src)}</code></pre>"


def fig(name, caption):
    return f'<figure><img src="media/{name}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def grid(cols, *figs):
    return f'<div class="grid c{cols}">\n' + "\n".join(figs) + "\n</div>"


CSS = """
  :root { --bg: #faf9f6; --ink: #1b1b1b; --rule: #e3e0d9; --max: 960px; }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.6 Georgia, "Iowan Old Style", "Times New Roman", serif; }
  header, main, footer { max-width: var(--max); margin: 0 auto; padding: 0 24px; }
  header { padding-top: 64px; padding-bottom: 40px; border-bottom: 1px solid var(--rule); }
  a { color: var(--ink); }
  .back, .eyebrow, nav, figcaption, table, footer { font-size: 15px; }
  .back { margin: 0 0 20px; }
  .back a { text-decoration: none; }
  .eyebrow { margin: 0 0 16px; }
  h1 { font-size: 40px; line-height: 1.15; font-weight: 400; margin: 0 0 12px; }
  header p { margin: 0; }
  nav { margin-top: 24px; }
  nav a { text-decoration: none; margin-right: 20px; border-bottom: 1px solid var(--rule); }
  section { padding: 56px 0; border-bottom: 1px solid var(--rule); }
  section:last-of-type { border-bottom: 0; }
  h2 { font-size: 28px; font-weight: 400; margin: 0 0 24px; }
  h3 { font-size: 20px; font-weight: 400; margin: 32px 0 12px; }
  p { margin: 0 0 16px; }
  .grid { display: grid; gap: 20px 16px; margin: 20px 0 28px; }
  .c2 { grid-template-columns: repeat(2, 1fr); }
  .c3 { grid-template-columns: repeat(3, 1fr); }
  .c4 { grid-template-columns: repeat(4, 1fr); }
  .c5 { grid-template-columns: repeat(5, 1fr); }
  .narrow { max-width: 460px; margin: 20px auto 28px; }
  figure { margin: 0; }
  figure img { width: 100%; height: auto; display: block; border-radius: 4px; background: #ddd; }
  figcaption { margin-top: 6px; }
  pre { overflow-x: auto; background: #fff; border: 1px solid var(--rule); border-radius: 4px; padding: 14px 16px; margin: 0 0 20px; }
  code { font: inherit; font-size: 14px; white-space: pre; }
  .tablewrap { overflow-x: auto; margin: 0 0 20px; }
  table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 8px 14px 8px 0; border-bottom: 1px solid var(--rule); font-weight: normal; white-space: nowrap; }
  footer { padding: 40px 24px 64px; }
  body { text-transform: lowercase; }
  pre, code { text-transform: none; }
  @media (max-width: 640px) { h1 { font-size: 30px; } .c3, .c4 { grid-template-columns: repeat(2, 1fr); } }
"""

h, w = P11["image_shape"]
speed_two = P11["seconds_four_loops"] / P11["seconds_two_loops"]
speed_scipy = P11["seconds_two_loops"] / P11["seconds_scipy"]

PSNR = SH["psnr"]
P_ROWS = "\n".join(f"    <tr><td>Sharpened, amount {a}</td><td>{PSNR[a]:.2f} dB</td></tr>" for a in ["0.5", "1", "1.5", "2", "3", "4", "6"])
CD = HY["color_diffs"]
CUT_H, CUT_L = Q["cutoff_high"], Q["cutoff_low"]

PART_1_1 = f"""<section id="p11">
  <h2>Part 1.1 · Convolutions from scratch</h2>
  <p>A 2D convolution slides a filter over the image and sums the products of the filter's weights and the pixels underneath. Padding is done by hand, as shown below.</p>
  {code(filters.pad_zero, filters.conv2d_four_loops, filters.conv2d_two_loops)}
  <p>I compared both against scipy.signal.convolve2d on my photo ({w}×{h}) with a 9×9 box filter. The outputs match.</p>
  <div class="tablewrap"><table>
    <tr><th>Implementation</th><th>Time</th><th>Largest difference from SciPy</th></tr>
    <tr><td>Four loops</td><td>{P11["seconds_four_loops"]:.2f} s</td><td>{P11["max_diff_four_vs_scipy"]:.1e}</td></tr>
    <tr><td>Two loops</td><td>{P11["seconds_two_loops"]:.2f} s</td><td>{P11["max_diff_two_vs_scipy"]:.1e}</td></tr>
    <tr><td>scipy.signal.convolve2d</td><td>{P11["seconds_scipy"]:.3f} s</td><td></td></tr>
  </table></div>
  <p>Moving the inner two loops into NumPy makes the convolution about {speed_two:.0f} times faster. SciPy is still about {speed_scipy:.0f} times faster since its outer loops are compiled too.</p>
  <p>For boundaries, I fill with zeros, so everything outside the photo counts as black. SciPy offers other options, like boundary="symm" (mirror) and "wrap". The cameraman sections below use "symm".</p>
  <p>Here is the photo with the 9×9 box filter, then with D<sub>x</sub> and D<sub>y</sub>.</p>
  {code(filters.box_filter)}
  <pre><code>DX = np.array([[1.0, 0.0, -1.0]])
DY = np.array([[1.0], [0.0], [-1.0]])</code></pre>
  {grid(4, fig("selfie_gray.jpg", "Grayscale input"), fig("selfie_box.jpg", "9×9 box filter"), fig("selfie_dx.jpg", "Convolved with Dx"), fig("selfie_dy.jpg", "Convolved with Dy"))}
</section>"""

PART_1_2 = f"""<section id="p12">
  <h2>Part 1.2 · Finite difference operator</h2>
  <p>I convolved the cameraman image with D<sub>x</sub> and D<sub>y</sub> to get the partial derivatives. The gradient magnitude is √(dx² + dy²). I then set a threshold to turn it into an edge image.</p>
  {grid(4, fig("cam_dx.png", "Partial derivative in x"), fig("cam_dy.png", "Partial derivative in y"), fig("cam_mag.png", "Gradient magnitude"), fig("cam_edges.png", f"Edges, threshold {P12['threshold']:.2f}"))}
  <p>I tried thresholds from 0.15 to 0.40 and chose {P12["threshold"]:.2f}. Below that, the grass fills with noise. Above that, the tripod legs break up and the coat disappears. At {P12["threshold"]:.2f} the man, camera and tripod are complete and a little grass noise is left. The faint buildings in the background are mostly lost.</p>
</section>"""

PART_1_3 = f"""<section id="p13">
  <h2>Part 1.3 · Derivative of Gaussian filter</h2>
  <p>The finite difference is noisy, so I blur first. I made the Gaussian with cv2.getGaussianKernel (σ = {P13["sigma"]:.0f}, size {P13["kernel_size"]}) and took an outer product to get 2D. Then I repeated Part 1.2 on the blurred image.</p>
  {grid(4, fig("cam_blur.png", "Blurred with G"), fig("blur_dx.png", "dx of blurred"), fig("blur_dy.png", "dy of blurred"), fig("blur_mag.png", "Gradient magnitude"))}
  {grid(2, fig("cam_edges.png", f"Finite difference only, threshold {P12['threshold']:.2f}"), fig("blur_edges.png", f"Gaussian then finite difference, threshold {P13['threshold']:.2f}"))}
  <p><strong>What is different?</strong> The grass noise is gone. The edges are thicker and smoother. The buildings and the horizon now show up as clean edges. The small details in the camera and face merge together. The gradient values are smaller, so I lowered the threshold to {P13["threshold"]:.2f}.</p>
  <p>Next I did it with one convolution. I convolved G with D<sub>x</sub> and D<sub>y</sub> to get the two DoG filters, and applied them to the image directly.</p>
  {grid(3, fig("gaussian.png", "Gaussian G"), fig("dog_x.png", "G ∗ Dx"), fig("dog_y.png", "G ∗ Dy"))}
  {grid(4, fig("dog_dx.png", "dx with DoG"), fig("dog_dy.png", "dy with DoG"), fig("dog_mag.png", "Gradient magnitude"), fig("dog_edges.png", f"Edges, threshold {P13['threshold']:.2f}"))}
  <p>The result is the same as blurring first. The largest difference between the two gradient magnitude images is {P13["max_diff_everywhere"]:.1e}, and the two edge images differ in {P13["edge_pixels_that_differ"]} pixels.</p>
</section>"""

BELLS = f"""<section id="bells">
  <h2>Bells and whistles · Gradient orientation</h2>
  <p>The orientation is the angle of the vector (dx, dy) from the DoG filters. I did not use a built-in angle function. I used arctan(dy / dx), which only gives angles between −90° and 90°, and added 180° where dx is negative. Then I wrapped the result to 0° to 360°.</p>
  <p>The angle is the hue and the gradient magnitude is the brightness, so flat areas are black. Opposite sides of an object get opposite hues, like the two sides of the coat.</p>
  {grid(2, fig("orientation.png", "Orientation as hue, magnitude as brightness"), fig("orientation_wheel.png", "Hue for each direction"))}
</section>"""

def panels(prefix, first, second):
    caps = [f"(a) {first}, level 0", f"(b) {second}, level 0", "(c) Sum, level 0",
            f"(d) {first}, level 2", f"(e) {second}, level 2", "(f) Sum, level 2",
            f"(g) {first}, last level", f"(h) {second}, last level", "(i) Sum, last level",
            f"(j) {first}, all levels", f"(k) {second}, all levels", "(l) Blend"]
    return grid(3, *[fig(f"{prefix}_{l}.jpg", c) for l, c in zip("abcdefghijkl", caps)])

PART_2_1 = f"""<section id="p21">
  <h2>Part 2.1 · Image sharpening</h2>
  <p>A Gaussian blur keeps only the low frequencies. If you subtract the blur from the original, you get the high frequencies, which are the edges and fine texture. Adding extra high frequencies makes the image look sharper: sharpened = I + α (I − G ∗ I). This is one convolution with the filter (1 + α) δ − α G, where δ is the impulse filter and α is the amount.</p>
  <p><strong>Taj Mahal</strong> (σ = {SH["sigmas"]["taj"]:.0f})</p>
  {grid(3, fig("sharp_taj_original.jpg", "Original"), fig("sharp_taj_blur.jpg", "Blurred"), fig("sharp_taj_high.jpg", "High frequencies"))}
  {grid(4, fig("sharp_taj_a0.5.jpg", "Sharpened, α = 0.5"), fig("sharp_taj_a1.jpg", "α = 1"), fig("sharp_taj_a2.jpg", "α = 2"), fig("sharp_taj_a4.jpg", "α = 4"))}
  <p>At α = 0.5 the change is small. At α = 1 and 2 the carving and the trees look clearer. At α = 4 there are halos around strong edges, like the dome against the sky, and the image looks harsh.</p>
  <p><strong>Doe Library</strong> (my photo, σ = {SH["sigmas"]["library"]:.0f})</p>
  {grid(3, fig("sharp_library_original.jpg", "Original"), fig("sharp_library_blur.jpg", "Blurred"), fig("sharp_library_high.jpg", "High frequencies"))}
  {grid(4, fig("sharp_library_a0.5.jpg", "Sharpened, α = 0.5"), fig("sharp_library_a1.jpg", "α = 1"), fig("sharp_library_a2.jpg", "α = 2"), fig("sharp_library_a4.jpg", "α = 4"))}
  <p><strong>Blur a sharp image, then sharpen it.</strong> I blurred the sharp library photo (σ = 2), sharpened it again (σ = 2), and compared it to the original with PSNR. Higher PSNR means closer. The crops are enlarged 2×.</p>
  {grid(4, fig("eval_original.jpg", "Original"), fig("eval_blurred.jpg", "Blurred"), fig("eval_best.jpg", f"Sharpened, amount {SH['best_amount']}"), fig("eval_strong.jpg", "Sharpened, amount 6"))}
  <div class="tablewrap"><table>
    <tr><th>Compared to the original</th><th>PSNR</th></tr>
    <tr><td>Blurred</td><td>{PSNR["blurred"]:.2f} dB</td></tr>
{P_ROWS}
  </table></div>
  <p>Sharpening helps a bit. The best amount is {SH["best_amount"]}, with {PSNR[str(SH["best_amount"])]:.2f} dB against {PSNR["blurred"]:.2f} dB for the blurred image. The letters are readable again but still softer than the original. Blurring removes detail, and sharpening cannot bring it back. At amount 6 the score drops to {PSNR["6"]:.2f} dB, which is worse than the blurred image, because the halos and noise count as errors.</p>
</section>"""

PART_2_2 = f"""<section id="p22">
  <h2>Part 2.2 · Hybrid images</h2>
  <p>A hybrid image adds the low frequencies of one image to the high frequencies of another. Up close you see the high frequency image. From far away you only see the low frequency one. The low pass is a Gaussian blur. The high pass is the image minus its blur. The hybrid is the sum.</p>
  <p>To align the images, I picked two points in each one (the eyes) and rotated, scaled and shifted each image so the points match on one canvas.</p>
  <p>I tried σ pairs of (3, 8), (4, 12) and (6, 16), and used σ<sub>high</sub> = {Q["sigma_high"]} and σ<sub>low</sub> = {Q["sigma_low"]} for every hybrid. The cutoff is where the Gaussian gain falls to one half, f = √(2 ln 2) / (2πσ). That is {CUT_H:.3f} cycles per pixel for the high pass and {CUT_L:.3f} for the low pass. On a 640 pixel wide image, that is about {CUT_H * 640:.0f} and {CUT_L * 640:.0f} cycles across.</p>

  <h3>Nutmeg and Derek, step by step</h3>
  <p>Nutmeg gives the high frequencies and Derek gives the low ones. The starter code does it the other way, but Derek's skin is smooth and Nutmeg's fur has strong contrast, so Derek almost disappeared.</p>
  {grid(4, fig("hybrid_nutmeg_derek_high_input.jpg", "Nutmeg, original"), fig("hybrid_nutmeg_derek_low_input.jpg", "Derek, original"), fig("hybrid_nutmeg_derek_high_aligned.jpg", "Nutmeg, aligned"), fig("hybrid_nutmeg_derek_low_aligned.jpg", "Derek, aligned"))}
  {grid(5, fig("hybrid_nutmeg_derek_high_aligned.jpg", "Nutmeg"), fig("hybrid_nutmeg_derek_low_aligned.jpg", "Derek"), fig("hybrid_nutmeg_derek_high_filtered.jpg", "Nutmeg, high pass"), fig("hybrid_nutmeg_derek_low_filtered.jpg", "Derek, low pass"), fig("hybrid_nutmeg_derek.jpg", "Hybrid"))}
  {grid(5, fig("fft_high_input.jpg", "Nutmeg, Fourier transform"), fig("fft_low_input.jpg", "Derek"), fig("fft_high_filtered.jpg", "High pass"), fig("fft_low_filtered.jpg", "Low pass"), fig("fft_hybrid.jpg", "Hybrid"))}
  <p>The bottom row is the log magnitude of the Fourier transform of the grayscale images. The high pass has a dark hole in the middle, since the low frequencies are gone. The low pass is only bright near the center. The hybrid has both. The bright cross comes from the image borders.</p>
  <div class="narrow">{fig("hybrid_nutmeg_derek.jpg", "Hybrid: Nutmeg up close, Derek from far away")}</div>
  <p>The same hybrid at 100%, 50%, 25% and 12.5% of the size:</p>
  {fig("hybrid_nutmeg_derek_scales.jpg", "The hybrid at four sizes")}

  <h3>Me and the Emir of Bukhara</h3>
  <p>My selfie gives the high frequencies and the Emir (from Project 1) gives the low ones. From far away you see the beard and turban. Up close you see my face. It only partly works, because the busy background of my selfie shows up as faint lines on the turban.</p>
  {grid(3, fig("hybrid_me_emir_high_input.jpg", "Me, high frequencies"), fig("hybrid_me_emir_low_input.jpg", "Emir, low frequencies"), fig("hybrid_me_emir.jpg", "Hybrid"))}
  {fig("hybrid_me_emir_scales.jpg", "The hybrid at four sizes")}

  <h3>Failure: the library from two distances</h3>
  <p>I used the two library photos from Project 0, one far away with zoom and one close up. I aligned them on the words "THE UNIVERSITY LIBRARY". It fails for two reasons. It is the same building at every size, so there is no second picture to see. Also, the perspective is different, so the columns and windows show up twice. Two points can fix rotation, scale and shift, but not perspective.</p>
  {grid(3, fig("hybrid_library_high_input.jpg", "Far away, high frequencies"), fig("hybrid_library_low_input.jpg", "Close up, low frequencies"), fig("hybrid_library.jpg", "Hybrid"))}
  {fig("hybrid_library_scales.jpg", "The hybrid at four sizes")}

  <h3>Bells and whistles: color</h3>
  {grid(4, fig("hybrid_nutmeg_derek_gray_both.jpg", "Gray in both"), fig("hybrid_nutmeg_derek_color_low.jpg", "Color in low only"), fig("hybrid_nutmeg_derek_color_high.jpg", "Color in high only"), fig("hybrid_nutmeg_derek_color_both.jpg", "Color in both"))}
  <p>Color works best in the low frequency component. Color in the high frequency component alone looks the same as no color, and color in both looks the same as color in the low one only. The average pixel difference from the gray hybrid is {CD["low_only_vs_gray"]:.3f} with color in the low component and {CD["high_only_vs_gray"]:.3f} with color in the high one. The high pass image has almost no color in it, and the eye does not see color in fine detail.</p>
</section>"""

PART_2_3 = f"""<section id="p23">
  <h2>Part 2.3 · Gaussian and Laplacian stacks</h2>
  <p>A stack is a pyramid without downsampling, so every level has the full size. In the Gaussian stack, level 0 is the image and level k is the image blurred with σ = 2 · 2<sup>k−1</sup>, so σ is 2, 4, 8 and 16 for levels 1 to 4. In the Laplacian stack, each level is the difference of two neighbors, G<sub>k</sub> − G<sub>k+1</sub>. The last level is the blurriest Gaussian, so adding all the levels gives the image back. The largest error on the apple is {Q["reconstruction_error"]:.1e}.</p>
  <p>Levels 0 to 4 go from left to right. Band pass levels are stretched so gray means zero, each level on its own.</p>
  {fig("stack_apple_gaussian.jpg", "Apple, Gaussian stack")}
  {fig("stack_apple_laplacian.jpg", "Apple, Laplacian stack")}
  {fig("stack_orange_gaussian.jpg", "Orange, Gaussian stack")}
  {fig("stack_orange_laplacian.jpg", "Orange, Laplacian stack")}
  {fig("stack_mask_gaussian.jpg", "Gaussian stack of the mask")}
</section>"""

PART_2_4 = f"""<section id="p24">
  <h2>Part 2.4 · Multiresolution blending</h2>
  <p>At each level I take M<sub>k</sub> L<sub>k</sub>(A) + (1 − M<sub>k</sub>) L<sub>k</sub>(B), where M<sub>k</sub> is the level of the mask's Gaussian stack, and then add up all the levels. The mask gets blurrier at each level, so fine detail switches from A to B over a few pixels and coarse colors switch over many. Each color channel is blended on its own.</p>

  <h3>The oraple</h3>
  <p>The mask is 1 on the left half, with five levels. The color version is my bells and whistles. It is more convincing than gray, because the two fruits mostly differ in hue.</p>
  {grid(2, fig("oraple_gray.jpg", "Gray"), fig("oraple.jpg", "Color"))}
  <p>Figure 3.42 recreated. The rows are level 0, level 2 and the last level. The columns are the apple times its mask, the orange times its mask, and the sum. The bottom row adds up all the levels.</p>
  {panels("fig342", "Apple", "Orange")}

  <h3>A dock that leads to a church</h3>
  <p>This is a horizontal seam. The top is the lake and church from one Prokudin-Gorskii photo, and the bottom is the dock from another. The mask is 1 above 62% of the height. At 55% a boat from the dock photo showed at the left edge.</p>
  {grid(3, fig("blend_dock_a.jpg", "Church"), fig("blend_dock_b.jpg", "Wharf"), fig("blend_dock_mask.jpg", "Mask"))}
  <div class="narrow" style="max-width: 720px">{fig("blend_dock.jpg", "Blend")}</div>

  <h3>My face on the Emir, with an irregular mask</h3>
  <p>I aligned my selfie to the Emir by the eyes. The mask is an ellipse around the face. My first, bigger ellipse pulled the yellow couch from my selfie into the beard, so I made it smaller. Below the blend is the Laplacian stack of this result, in the layout of Figure 10 in the paper. The rows are level 0, level 2 and the last of six levels.</p>
  {grid(3, fig("blend_face_a.jpg", "My selfie, aligned"), fig("blend_face_b.jpg", "Emir of Bukhara"), fig("blend_face_mask.jpg", "Mask"))}
  <div class="narrow" style="max-width: 720px">{fig("blend_face.jpg", "Blend")}</div>
  {panels("swap", "Selfie", "Emir")}
</section>"""

LEARNED = """<section id="learned">
  <h2>What I learned</h2>
  <p>A hybrid or a blended image is all about the frequencies, not the individual pixels in the image. If you split an image into separate bands and treat each band as its own, then one picture can morph into another at a further distance.</p>
</section>"""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cs180 project 2 · fun with filters and frequencies</title>
<style>{CSS}</style>
</head>
<body>

<header>
  <p class="back"><a href="../">← All projects</a></p>
  <p class="eyebrow">CS180 · Fall 2026 · Project 2</p>
  <h1>Fun with Filters and Frequencies</h1>
  <p>Vikram Bhamre · UC Berkeley</p>
  <nav>
    <a href="#p11">1.1 Convolution</a>
    <a href="#p12">1.2 Finite difference</a>
    <a href="#p13">1.3 Derivative of Gaussian</a>
    <a href="#bells">Gradient orientation</a>
    <a href="#p21">2.1 Sharpening</a>
    <a href="#p22">2.2 Hybrid images</a>
    <a href="#p23">2.3 Stacks</a>
    <a href="#p24">2.4 Blending</a>
  </nav>
</header>

<main>

{PART_1_1}

{PART_1_2}

{PART_1_3}

{BELLS}

{PART_2_1}

{PART_2_2}

{PART_2_3}

{PART_2_4}

{LEARNED}

</main>

<footer>
  CS180 Project 2, September 2026.
</footer>

</body>
</html>
"""

Path("index.html").write_text(PAGE)
print(len(PAGE))
