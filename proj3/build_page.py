import ast
import html
import json
import shutil
from pathlib import Path

from PIL import Image

A, B, MEDIA = Path("results/parta"), Path("results/partb"), Path("media")
NB_A = json.loads(Path("parta.ipynb").read_text())
NB_B = json.loads(Path("partb.ipynb").read_text())
MA = json.loads((A / "metrics.json").read_text())
MB = json.loads((B / "metrics.json").read_text())
STARTS = ["0.02", "0.04", "0.06", "0.10", "0.14", "0.25", "0.50"]


def code(nb, *names):
    found = {}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or src.startswith("#@title"):
            continue
        lines = src.splitlines()
        for node in ast.parse(src).body:
            if getattr(node, "name", None) in names:
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                found[node.name] = "\n".join(lines[start - 1:node.end_lineno])
    return f"<pre><code>{html.escape(chr(10).join(found[n] + chr(10) for n in names).rstrip())}</code></pre>"


def jpg(src, name, rotate=False):
    image = Image.open(src).convert("RGB")
    if rotate:
        image = image.rotate(180)
    (MEDIA / name).parent.mkdir(parents=True, exist_ok=True)
    image.save(MEDIA / name, quality=90)
    return name


def png(src, name):
    (MEDIA / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, MEDIA / name)
    return name


def fig(name, caption, cls=""):
    return (f'<figure{f" class={chr(34)}{cls}{chr(34)}" if cls else ""}><img src="media/{name}" alt="{html.escape(caption)}" '
            f'loading="lazy"><figcaption>{caption}</figcaption></figure>')


def grid(cols, *figs):
    return f'<div class="grid c{cols}">\n' + "\n".join(figs) + "\n</div>"


def sweep(prefix):
    figs = [fig(jpg(A / f"{prefix}_original.png", f"a/{prefix}_original.jpg"), "Original")]
    figs += [fig(jpg(A / f"{prefix}_t={t}.png", f"a/{prefix}_t{t}.jpg"), f"t = {t}") for t in STARTS]
    return grid(4, *figs)


def panel(name, caption):
    return fig(jpg(A / f"{name}.png", f"a/{name}.jpg"), caption)


def plot(name, caption):
    return fig(png(B / f"{name}.png", f"b/{name}.png"), caption)


CSS = Path("../proj2/build_page.py").read_text().split('CSS = """')[1].split('"""')[0] + """
  .hybrid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; align-items: end; margin: 20px 0 28px; }
  .far { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; }
  .small img { width: 64px; }
  .tiny img { width: 32px; }
"""

PB = MB["class"]
PURE = MB["pure_noise"]

B1 = f"""<section id="b1">
  <h2>Part B.1 · A one-step denoising UNet</h2>
  <h3>1.1 The UNet</h3>
  <p>The UNet has two downsampling blocks, a bottleneck that pools the 7×7 map to 1×1 and back, and two upsampling blocks. Each upsampling step concatenates the matching encoder output (the skip connections). The simple ops are a 3×3 convolution, a stride-2 convolution, a stride-2 transposed convolution, a 7×7 average pool and a 7×7 transposed convolution, each followed by batch norm and GELU (the pool has no batch norm).</p>
  {code(NB_B, "Conv", "DownConv", "UpConv", "Flatten", "Unflatten", "ConvBlock", "DownBlock", "UpBlock")}
  <p>I wrote the forward pass once, in <code>run</code>. It takes two functions that change the bottleneck output and the first upsampling output. The plain UNet passes functions that do nothing. The conditioned UNets in Part B.2 reuse it.</p>
  {code(NB_B, "UnconditionalUNet")}
  <h3>1.2 Noising MNIST</h3>
  <p>To make training pairs, I add Gaussian noise to a clean digit x: z = x + σε, with ε ~ N(0, I). Here are four test digits at each σ. The noise is drawn once and scaled, so each column only changes σ.</p>
  {plot("noise_levels", "z = x + σε for σ from 0 to 1")}
  <h3>1.2.1 Training a denoiser at σ = 0.5</h3>
  <p>I trained the UNet (D = 128) to map z back to x with an L2 loss, for 5 epochs with Adam (learning rate 1e-4) and batches of 256. The noise is drawn fresh for every batch, so the model never sees the same noisy image twice.</p>
  {plot("denoise_loss", "Training loss, one point per step. The orange line is a 100-step moving average.")}
  {grid(1, plot("denoise_epoch1", "Test digits after epoch 1"), plot("denoise_epoch5", "Test digits after epoch 5"))}
  <p>After one epoch the digits are already readable, but some strokes are broken (the 1) or smudged (the 5). After five epochs the strokes are clean and continuous. The outputs are a little softer than the originals, which is what an L2 loss does when it is unsure.</p>
  <h3>1.2.2 Out-of-distribution noise levels</h3>
  <p>The same model, trained only at σ = 0.5, on three test digits with σ from 0 to 1. Each pair of rows is one digit.</p>
  {plot("ood", "Denoising at noise levels the model was not trained on")}
  <p>For σ ≤ 0.5 the output is clean. At σ = 0.6 it still works, with small artifacts. At σ = 0.8 and 1.0 the model invents strokes that are not there. The 1 turns into a scribble, since a thin digit is easy to lose in strong noise.</p>
  <h3>1.2.3 Denoising pure noise</h3>
  <p>Next I trained a fresh UNet with the same settings, but its input was pure noise z = ε, with no digit in it at all.</p>
  {plot("pure_noise_loss", "Training loss for the pure-noise model")}
  {grid(1, plot("pure_noise_epoch1", "Outputs after epoch 1"), plot("pure_noise_epoch5", "Outputs after epoch 5"))}
  {plot("pure_noise_vs_mean", "The mean of all training digits next to the model's outputs")}
  <p><strong>What the outputs look like.</strong> Every input gives the same blurry blob. It looks like a 0, 3, 8 and 9 drawn on top of each other. The loss drops to about 0.067 in the first 100 steps and then stays flat.</p>
  <p><strong>Why.</strong> The input says nothing about which digit to produce, so the best the model can do under an L2 loss is predict the point with the smallest total squared distance to all training digits. That point is their mean, the centroid of the training set. I checked this. Over 1,000 random inputs, the average output differs from the pixel-wise mean of the 60,000 training digits by a mean squared error of {PURE["mse_between_mean_output_and_training_mean"]:.1e}, and the outputs barely vary (pixel standard deviation {PURE["pixel_std_across_outputs"]:.4f}). So one-step denoising cannot generate new digits. It can only average.</p>
</section>"""

B2 = f"""<section id="b2">
  <h2>Part B.2 · Flow matching</h2>
  <p>Instead of jumping from noise to a digit in one step, flow matching learns a velocity field and follows it in small steps. For a clean digit x<sub>1</sub> and noise x<sub>0</sub>, the path is x<sub>t</sub> = (1 − t) x<sub>0</sub> + t x<sub>1</sub>, so t = 0 is noise and t = 1 is the digit. Its velocity is x<sub>1</sub> − x<sub>0</sub>, and the UNet u<sub>θ</sub>(x<sub>t</sub>, t) learns to predict it.</p>
  <h3>2.1 Time conditioning</h3>
  <p>Each FCBlock is Linear, GELU, Linear. Two of them turn t into a 2D-channel vector and a D-channel vector. These multiply the bottleneck output and the first upsampling output.</p>
  {code(NB_B, "FCBlock", "TimeConditionalUNet")}
  <h3>2.2 Training</h3>
  <p>Each step draws a batch of digits, a fresh noise image for each, and a time t ~ U[0, 1] for each. The loss is the squared error between the predicted and true velocity.</p>
  {code(NB_B, "time_fm_forward", "train_flow")}
  <p>I trained with D = 64, batches of 64, and Adam starting at 1e-2, with an exponential decay that multiplies the learning rate by 0.1<sup>1/10</sup> after each of the 10 epochs.</p>
  {plot("time_loss", "Training loss for the time-conditioned UNet. The first steps are much higher, so most of the curve sits near the bottom.")}
  <h3>2.3 Sampling</h3>
  <p>Sampling starts at pure noise and takes 50 Euler steps: x ← x + (1/50) u<sub>θ</sub>(x, t), for t = 0, 1/50, …, 49/50. All three grids use the same starting noise.</p>
  {code(NB_B, "time_fm_sample")}
  {grid(1, plot("time_epoch1", "Epoch 1"), plot("time_epoch5", "Epoch 5"), plot("time_epoch10", "Epoch 10"))}
  <p>After one epoch the samples are digit-like strokes but not digits. After five, about half are readable. After ten, most are clear digits, and a few still look like two digits blended together. Since the samples share their starting noise, you can see the same sample sharpen from epoch 5 to epoch 10.</p>
  <h3>2.4 Class conditioning</h3>
  <p>Two more FCBlocks take the class as a one-hot vector. The bottleneck becomes c<sub>1</sub> · unflatten + t<sub>1</sub> and the first upsampling output becomes c<sub>2</sub> · up1 + t<sub>2</sub>. During training I zero the class vector 10% of the time, so the same model also learns the unconditional velocity.</p>
  {code(NB_B, "ClassConditionalUNet", "class_fm_forward")}
  <h3>2.5 Training</h3>
  <p>The settings are the same as the time-only model.</p>
  {plot("class_loss", "Training loss for the class-conditioned UNet")}
  <h3>2.6 Sampling with classifier-free guidance</h3>
  <p>Each of the 300 Euler steps evaluates the model twice, with and without the class, and mixes them: u = u<sub>uncond</sub> + γ (u<sub>cond</sub> − u<sub>uncond</sub>), with γ = 5. Each column is one digit, with four samples each.</p>
  {code(NB_B, "class_fm_sample")}
  {grid(1, plot("class_epoch1", "Epoch 1"), plot("class_epoch5", "Epoch 5"), plot("class_epoch10", "Epoch 10"))}
  <p>The class label helps a lot. Every digit is right and readable after one epoch, if a bit blotchy. By epoch 5 the strokes are clean, and epoch 10 adds variety in slant and thickness.</p>
  <h3>Removing the learning rate scheduler</h3>
  <p>The scheduler takes the learning rate from 1e-2 down to about 1.3e-3. I removed it and used a constant learning rate of 1e-3 instead, close to where the scheduler ends. The large early steps are what the decay protects against. Starting low means the model never needs to be calmed down later. I kept everything else the same, including the seed.</p>
  {plot("class_no_scheduler_loss", "Training loss without the scheduler, constant learning rate 1e-3")}
  {grid(1, plot("class_no_scheduler_epoch1", "No scheduler, epoch 1"), plot("class_no_scheduler_epoch5", "No scheduler, epoch 5"), plot("class_no_scheduler_epoch10", "No scheduler, epoch 10"))}
  <div class="tablewrap"><table>
    <tr><th>Run</th><th>Mean training loss, last epoch</th><th>Loss on 2,000 test digits</th></tr>
    <tr><td>Exponential decay from 1e-2</td><td>{PB["class"]["last_epoch_loss"]:.4f}</td><td>{PB["class"]["held_out_loss"]:.4f}</td></tr>
    <tr><td>Constant 1e-3</td><td>{PB["class_no_scheduler"]["last_epoch_loss"]:.4f}</td><td>{PB["class_no_scheduler"]["held_out_loss"]:.4f}</td></tr>
  </table></div>
  <p>The test loss uses the same digits, noise and times for both models. The constant learning rate ends about 3% higher, and the samples at epoch 10 are just as clean. So a single well-chosen learning rate does nearly as well as the schedule.</p>
</section>"""


MSE = MA["reconstruction_mse"]
PROMPTS = {"campanile": "a photograph of a lighthouse at sunset",
           "my_image_1": "a photograph of a library covered in snow",
           "my_image_2": "a photograph of a small robot standing in the dirt"}

A1 = f"""<section id="a1">
  <h2>Part A.1 · Training and sampling with a pretrained flow model</h2>
  <p>Part A uses PixNerd-XXL-P16-T2I, a text-to-image flow model that works directly on 512×512 pixels. A prompt is turned into an embedding with Qwen3, and the model predicts a velocity v<sub>θ</sub>(x<sub>t</sub>, t, c). Time t = 0 is pure noise and t = 1 is a clean image. I ran everything on a free Colab T4 in float32. For Sections 1.1 to 1.5 I used my photo of Doe Library from Project 0.</p>
  <h3>1.1 Noisy training inputs</h3>
  <p>A training input is a straight-line mix of noise and image: x<sub>t</sub> = (1 − t) ε + t x<sub>1</sub>, with ε ~ N(0, I). This scales the image down as well as adding noise. I drew one ε and reused it for all three times.</p>
  {code(NB_A, "interpolate")}
  {panel("training_inputs", "My photo and x<sub>t</sub> at t = 0.25, 0.50 and 0.75, all with the same noise")}
  <p>At t = 0.25 you can only just make out the building. At t = 0.75 the columns and windows are clear.</p>
  <h3>1.2 Gaussian blur as a baseline</h3>
  <p>Since the image part of x<sub>t</sub> is scaled by t, I first divide by t, which gives x<sub>1</sub> + ((1 − t)/t) ε. Then I blur. The notebook tries σ = 1, 2, 4, 8 and 16 and keeps the one closest to the clean photo. The chosen σ is in each title.</p>
  {grid(1, panel("blur_0.25", "t = 0.25"), panel("blur_0.5", "t = 0.50"), panel("blur_0.75", "t = 0.75"))}
  <p>Blurring trades noise for detail, and the best σ shrinks as t grows: 8, then 4, then 1. At t = 0.25 even σ = 8 leaves colored blobs and soft columns. At t = 0.75 the best choice barely blurs at all, so most of the noise is still there. A blur cannot tell noise from fine detail, so it cannot remove one without the other.</p>
  <h3>1.3 One-step denoising</h3>
  <p>With one velocity prediction I can jump straight to the end of the path: x̂<sub>1</sub> = x<sub>t</sub> + (1 − t) v<sub>θ</sub>(x<sub>t</sub>, t, c), with the prompt "a high quality photo".</p>
  {code(NB_A, "one_step_denoise")}
  {grid(1, panel("one_step_0.25", "t = 0.25"), panel("one_step_0.5", "t = 0.50"), panel("one_step_0.75", "t = 0.75"))}
  <p>This is far better than the blur. Even at t = 0.25 the result is a clean building in the right place. But it is smooth and generic: the carved lettering, lamps and plants are smeared or missing. The estimate is close to an average of the images that could have made this x<sub>t</sub>, so detail the noise has hidden comes out blurred.</p>
  <h3>1.4 What the model adds</h3>
  <p>The one-step update is (1 − t) v̂. Since I know the clean photo and the noise, I also know the true update, (1 − t)(x<sub>1</sub> − ε) = x<sub>1</sub> − x<sub>t</sub>. Both updates are shown on the same symmetric color scale, with zero at middle gray. The error map is the RMS difference over RGB at each pixel.</p>
  {code(NB_A, "velocity_diagnostics")}
  <div class="narrow">{panel("velocity_original", "Original")}</div>
  {grid(1, panel("velocity_0.25", "t = 0.25"), panel("velocity_0.50", "t = 0.50"), panel("velocity_0.75", "t = 0.75"))}
  <p>The predicted and true updates look almost the same. Both hold the noise to remove plus the image structure to add, so they are mostly gray with the building faintly showing. The error map shows where they differ. I set its color limit to 0.3 at all three times, about the 99th percentile at t = 0.50. The flat sky and walls are dark, so the model gets those right. The error sits on edges and fine detail: the carved lettering, the column tops, window frames, plants and the pavement. That is the detail the noise has hidden, so the model can only guess it. The error shrinks as t grows (the 90th percentile goes 0.21, 0.13, 0.08), because less of the image is hidden. Where the model is unsure, it predicts an average, which is why the one-step results look smooth.</p>
  <h3>1.5 Iterative denoising with Euler steps</h3>
  <p>The velocity changes along the path, so a single jump is inaccurate. Euler sampling takes a small step, x<sub>t+h</sub> = x<sub>t</sub> + h v<sub>θ</sub>(x<sub>t</sub>, t, c), predicts again, and repeats. I precompute the step times so the last step is shortened to land exactly on t = 1. The notebook checks this with a constant velocity field: 50 steps from 0, 25 from 0.5, and 4 from 0.93, the last of which is 0.01.</p>
  {code(NB_A, "euler_sample")}
  {panel("trajectory", "Denoising my t = 0.50 input with h = 0.02: the start, every fifth step, and the end")}
  {panel("comparison", "The same t = 0.50 input, denoised three ways")}
  <div class="tablewrap"><table>
    <tr><th>Method</th><th>Mean squared error to the original</th></tr>
    <tr><td>Gaussian blur</td><td>{MSE["blur"]:.4f}</td></tr>
    <tr><td>One step</td><td>{MSE["one_step"]:.4f}</td></tr>
    <tr><td>Euler, 25 steps</td><td>{MSE["euler"]:.4f}</td></tr>
  </table></div>
  <p>The table looks backwards at first. One step has the lowest error, but Euler looks best. A one-step estimate is close to the average of every image that could have made this x<sub>t</sub>, and that average is exactly what minimizes squared error. So it wins on the number while looking smooth. Euler commits to one specific, sharp image. Wherever its invented detail does not line up with the photo, like the lettering or the pavement, squared error counts it twice: once for the missing real detail and once for the invented one. By eye, Euler is the closest. The trajectory shows the noise fading out while the carvings, lamps and plants come back, and the blur is the worst on both counts.</p>
  <h3>1.6 Generating images from pure noise</h3>
  <p>Starting Euler sampling at t = 0 from pure noise turns the denoiser into a generator. These are five seeds with "a high quality photo" and no guidance.</p>
  {panel("conditional_samples", "Seeds 180 to 184, no guidance")}
  <p>All five are portraits of people. The model seems to treat "a high quality photo" as "a portrait". The images are soft, and two have badly broken faces (seeds 180 and 183). Without guidance the prompt only pulls weakly, so the samples drift toward something that is only roughly a photo.</p>
</section>"""

A2 = f"""<section id="a2">
  <h2>Part A.2 · Classifier-free guidance</h2>
  <p>CFG evaluates the model twice at the same x<sub>t</sub> and t, once with the prompt (v<sub>c</sub>) and once with the empty prompt (v<sub>u</sub>), and pushes past the conditional prediction: v = v<sub>c</sub> + w (v<sub>c</sub> − v<sub>u</sub>). With w = 0 this is the plain conditional prediction, which the notebook checks. I used w = 7.</p>
  {code(NB_A, "cfg_velocity")}
  {panel("cfg_samples", "The same five seeds with CFG, w = 7")}
  <p>Every sample is now a sharp, well-lit studio portrait with a correct face. The cost is variety. All five are centered, front-facing faces on plain backgrounds, where the unguided samples had different poses and framings. Each seed still keeps its rough layout and colors. Seed 182 has a hand by the face in both versions.</p>
</section>"""


A31 = f"""<section id="a31">
  <h2>Part A.3.1 · Image-to-image editing</h2>
  <p>This is SDEdit. I noise a real image to a starting time t with the interpolation above, then run the CFG Euler sampler from t to 1. A small t means lots of noise, so the model has to invent most of the image. A large t keeps most of the original. All sweeps use the prompt "a high quality photo", w = 7, h = 0.02 and the same noise.</p>
  {code(NB_A, "edit")}
  <h3>Campanile</h3>
  {sweep("edit_campanile")}
  <p>At t = 0.02 only a pale rounded column against blue survives. From 0.04 to 0.14 it is a plain tapered tower or obelisk with trees at the bottom. The belfry arches come back at t = 0.25, and the clock at 0.50.</p>
  <h3>My photo: Doe Library</h3>
  {sweep("edit_my_image_1")}
  <p>The low starting times keep only the gray and white palette. t = 0.02 is a portrait of a woman, and 0.04 is a still life of cups and canisters. At 0.06 it becomes a small classical temple, which is the first time the columns are used. By 0.14 the real colonnade is back in the right perspective.</p>
  <h3>My photo: a fire hydrant</h3>
  {sweep("edit_my_image_2")}
  <p>The tall white shape gets read as a person: a man in a white tank top at 0.04, then a leg in white trousers and a sneaker at 0.06. At 0.10 and 0.14 it is a white bottle with a red label, keeping the red cap as a logo. At 0.25 the hydrant is back, but its side caps look like fleshy lumps.</p>
  <h3>Editing a web image and drawings</h3>
  <p>The web image is Van Gogh's The Starry Night, from Wikimedia Commons.</p>
  {sweep("edit_web_image")}
  <p>This is the most interesting sweep. From 0.02 to 0.06 the painting becomes a photo of an old CRT monitor, and only the blue and violet colors survive. At 0.10 it is a photo through a window, with mountains, clouds and dark cypress-like trees. At 0.14 it is halfway between photo and painting, with a stone arch, real-looking trees and glowing stars. At 0.25 the painting is back, but the cypress has turned into a rocky crag over a photographic town.</p>
  {sweep("edit_drawing_1")}
  {sweep("edit_drawing_2")}
  <p>The drawings never become photos. At the lowest starting time each one turns into an unrelated face. Then the flat colors and white background get read as graphic design: at 0.04 and 0.06 the sailboat becomes a poster with a striped pyramid, fake text and a logo, and the mountains get fake lettering and a button for a sun. From 0.10 on, the model returns a cleaned-up version of the drawing, with smooth outlines and a perfectly round sun. It even adds a watermark to the sailboat. The mountains at 0.10 are the closest to "real", with a felt or paper texture in the fills. The generic prompt is not strong enough to push a cartoon all the way to a photo.</p>
</section>"""

A32 = f"""<section id="a32">
  <h2>Part A.3.2 · Text-guided editing</h2>
  <p>Same procedure, but with a prompt that describes the edit instead of "a high quality photo".</p>
  <h3>Campanile: "{PROMPTS["campanile"]}"</h3>
  {sweep("text_edit_campanile")}
  <p>At 0.02 and 0.04 it is a whole new lighthouse, with a red lantern, pink sunset clouds and boulders. Between 0.06 and 0.14 the edit is the best mix: the Campanile's framing and trees stay, while the tower becomes a lighthouse with a gallery at the top, lit warm by the sun. At 0.25 it is the Campanile again with lighthouse windows.</p>
  <h3>Doe Library: "{PROMPTS["my_image_1"]}"</h3>
  {sweep("text_edit_my_image_1")}
  <p>The model takes "library" literally. At low starting times it draws bookshelves full of books, buried in snow. At 0.10 and 0.14 Doe's colonnade is back under heavy snow, but with bookshelves where the windows should be. From 0.25 on it is the real building with snow on the ledges and the ground.</p>
  <h3>Fire hydrant: "{PROMPTS["my_image_2"]}"</h3>
  {sweep("text_edit_my_image_2")}
  <p>At 0.02 and 0.04 it is a cute toy robot unrelated to the hydrant. From 0.06 to 0.14 the robot is built out of the hydrant: the red cap becomes a head or chest, the cream body stays, and arms grow out of the side outlets. At 0.25 it is a hydrant again with a few robot parts.</p>
</section>"""


ANAGRAMS = [(1, 180, "a lithograph of a cat", "a lithograph of a ship on stormy waves"),
            (2, 180, "an oil painting of a woman's face", "an oil painting of a vase of flowers"),
            (0, 180, "an oil painting of an old man with a beard", "an oil painting of a forest with a river")]


def anagram(i, seed, upright, rotated):
    name = f"anagram_{i}_seed{seed}"
    return grid(2, fig(jpg(A / f"{name}.png", f"a/{name}.jpg"), f"Upright: “{upright}”"),
                fig(jpg(A / f"{name}.png", f"a/{name}_rotated.jpg", rotate=True), f"Rotated 180°: “{rotated}”"))


def hybrid(i, seed, distant, close):
    name = f"hybrid_{i}_seed{seed}"
    src = jpg(A / f"{name}.png", f"a/{name}.jpg")
    return (f'<div class="hybrid">{fig(src, f"Close: “{close}”")}'
            f'<div class="far">{fig(src, f"Far: “{distant}”", "small")}{fig(src, "Farther", "tiny")}</div></div>')


HYBRIDS = [(3, 180, "a lithograph of a skull", "a lithograph of a city at night"),
           (0, 190, "a lithograph of an old man's face", "a lithograph of a rocky coastline"),
           (4, 180, "a lithograph of a woman's face", "a lithograph of a flower garden")]

A33 = f"""<section id="a33">
  <h2>Part A.3.3 · Visual anagrams</h2>
  <p>At every Euler step I compute two CFG velocities. One is for x<sub>t</sub> with prompt p<sub>1</sub>. The other is for x<sub>t</sub> rotated 180° with prompt p<sub>2</sub>, rotated back. The step uses their average, so the image has to be a good p<sub>1</sub> image upright and a good p<sub>2</sub> image upside down at the same time.</p>
  {code(NB_A, "visual_anagrams")}
  <p>I tried three prompt pairs with two seeds each and kept the better seed of each pair.</p>
  {"".join(anagram(*a) for a in ANAGRAMS)}
  <p>The cat and ship works best. Upright, a cat's face fills a framed print. Upside down, the cat's ears and face become the hull and rigging of a sailing ship, and the dark top of the print becomes the waves. In the second one the flowers double as the woman's cheeks and lips, and the vase becomes her dark hair. The vase reads more clearly than the face. In the third, the old man's beard is the forest's foliage and his bright forehead becomes the light on the river. Faces make good anagrams: a face needs only a few dark spots in the right places, and the loose painting and print styles hide the seams.</p>
</section>"""

A34 = f"""<section id="a34">
  <h2>Part A.3.4 · Hybrid images</h2>
  <p>This is Project 2's hybrid images, done during sampling. At each step I compute CFG velocities for two prompts, keep the low frequencies of the first and the high frequencies of the second, and add them. The low-pass filter is a Gaussian blur (kernel 265, σ = 16), and the high-pass part is a velocity minus its blur. Up close you see the high-frequency prompt. From far away, or at a small size, you see the low-frequency one.</p>
  {code(NB_A, "make_hybrids")}
  {"".join(hybrid(*h) for h in HYBRIDS)}
  <p>I tried five prompt pairs with two seeds each. The skull and city works best. Up close it is a skyline at night, but the dark clusters of buildings sit where a skull's eye sockets, nose and teeth go, so at 64 pixels it is clearly a skull. In the coastline, a rock arch and the cave around it make the brow, eye and nose of a face. The woman's face is carried mostly by a big black shape of hair, with the flowers as her features. Two pairs did not work. No dog appeared in the mountain village, and the owl in the fruit bowl only shows up, in one seed, as two dark fruits for eyes and leaves for ear tufts. A hybrid works when the far subject has a strong, simple silhouette and the near subject is made of many small shapes that can be arranged to fill it.</p>
</section>"""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cs180 project 3 · fun with flow models</title>
<style>{CSS}</style>
</head>
<body>

<header>
  <p class="back"><a href="../">← All projects</a></p>
  <p class="eyebrow">CS180 · Fall 2026 · Project 3</p>
  <h1>Fun with Flow Models</h1>
  <p>Vikram Bhamre · UC Berkeley</p>
  <nav>
    <a href="#a1">A.1 Training and sampling</a>
    <a href="#a2">A.2 CFG</a>
    <a href="#a31">A.3.1 Editing</a>
    <a href="#a32">A.3.2 Text editing</a>
    <a href="#a33">A.3.3 Anagrams</a>
    <a href="#a34">A.3.4 Hybrids</a>
    <a href="#b1">B.1 One-step denoiser</a>
    <a href="#b2">B.2 Flow matching</a>
  </nav>
</header>

<main>

{A1}

{A2}

{A31}

{A32}

{A33}

{A34}

{B1}

{B2}

</main>

<footer>
  CS180 Project 3, October 2026.
</footer>

</body>
</html>
"""

Path("index.html").write_text(PAGE)
print(len(PAGE))
