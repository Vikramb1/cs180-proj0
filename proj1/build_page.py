import json
from pathlib import Path

SINGLE = ["cathedral", "monastery", "tobolsk"]
MULTI = ["church", "emir", "harvesters", "icon", "ilemselga", "melons", "religous_painting", "self_portrait", "siren", "three_generations", "wharf"]
TITLES = {
    "cathedral": "Cathedral", "monastery": "Monastery", "tobolsk": "Tobolsk", "church": "Church",
    "emir": "Emir of Bukhara", "harvesters": "Harvesters", "icon": "Icon", "ilemselga": "Ilemselga",
    "melons": "Melons", "religous_painting": "Religious painting", "self_portrait": "Self portrait",
    "siren": "Siren", "three_generations": "Three generations", "wharf": "Wharf",
}


def title(name):
    return TITLES.get(name, name.replace("_", " ").capitalize())


def fmt(v):
    return f"({v[0]}, {v[1]})"


PRIMARY = "ncc_grad"


def figure(name, r, key=PRIMARY, suffix="", label=None):
    g, rr = r[key]["g"], r[key]["r"]
    return f'''    <figure>
      <img src="media/{name}{suffix}.jpg" alt="{label or title(name)}, colorized" loading="lazy">
      <figcaption><strong>{label or title(name)}</strong><span>G {fmt(g)} · R {fmt(rr)}</span></figcaption>
    </figure>'''


def row(name, r):
    h, w = r["ncc"]["shape"]
    cells = [title(name), f"{w}×{h}", str(r["ncc"]["levels"]), fmt(r["ncc"]["g"]), fmt(r["ncc"]["r"]), fmt(r["ssd"]["g"]), fmt(r["ssd"]["r"]), fmt(r[PRIMARY]["g"]), fmt(r[PRIMARY]["r"])]
    return "      <tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def fmt_loss(v):
    return "" if v is None else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")


def loss_row(name, r):
    out = []
    for key, label in (("ncc", "NCC"), ("ssd", "L2"), (PRIMARY, "Edge NCC")):
        m = r[key]
        cells = [title(name), label, fmt(m["g"]), fmt_loss(m.get("g_loss")), fmt(m["r"]), fmt_loss(m.get("r_loss"))]
        out.append("      <tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "\n".join(out)


def loss_table(names, results):
    rows = "\n".join(loss_row(n, results[n]) for n in names)
    return f'''  <div class="tablewrap">
    <table>
      <thead>
        <tr><th>Image</th><th>Metric</th><th>G offset</th><th>G loss</th><th>R offset</th><th>R loss</th></tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>'''


def table(names, results):
    rows = "\n".join(row(n, results[n]) for n in names)
    return f'''  <div class="tablewrap">
    <table>
      <thead>
        <tr><th>Image</th><th>Plate size</th><th>Levels</th><th>G (NCC)</th><th>R (NCC)</th><th>G (L2)</th><th>R (L2)</th><th>G (edge NCC)</th><th>R (edge NCC)</th></tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>'''


def grid(names, results):
    return '  <div class="grid">\n' + "\n".join(figure(n, results[n]) for n in names) + "\n  </div>"


def section(sid, heading, intro, names, results, tbl=table):
    if not names:
        return ""
    return f'''<section id="{sid}">
  <h2>{heading}</h2>
  <p>{intro}</p>
{grid(names, results)}
{tbl(names, results)}
</section>
'''


APPROACH = '''<section id="approach">
  <h2>Approach</h2>
  <p>The blue plate is fixed, and the green and red plates are aligned to it. Every image is first converted to floating point so that JPEGs and TIFFs are treated identically. An offset is applied with a circular shift, which wraps a few rows and columns around to the far side of the image. All scoring is done only on the interior of the plates: a margin of 10% of the height and width is trimmed from every side before the two plates are compared.</p>
  <p>Both the L2 similarity metric and the NCC similarity metric are implemented, computed over the interior, and turned into a loss. For single-scale alignment, every integer offset in a window of ±15 pixels in each direction is tried. The metric is evaluated for each of those, and the offset with the lowest loss gives the final image. This is good enough for the low-resolution JPEGs.</p>
  <p>For the full-resolution TIFFs, we use an image pyramid. The first thing we do is downsample by 2 until the shorter side falls under 400 pixels. We then run the exhaustive search, and the offset found there is doubled and used as the center of a small ±3 search at the next finer level. This repeats down to the original image resolution.</p>
</section>
'''


def main():
    results = json.loads(Path("media/results.json").read_text())
    extra = [n for n in sorted(results) if n not in SINGLE and n not in MULTI]
    single = [n for n in SINGLE if n in results]
    multi = [n for n in MULTI if n in results]
    nav = ['<a href="#approach">Approach</a>', '<a href="#single">Single-scale</a>', '<a href="#multi">Multi-scale</a>']
    if extra:
        nav.append('<a href="#extra">More plates</a>')
    nav.append('<a href="#failures">Failures</a>')
    nav.append('<a href="#bells">Bells and whistles</a>')
    nav_html = "\n    ".join(nav)
    body = APPROACH
    body += section("single", "Single-scale alignment", "The low-resolution JPEG plates are aligned with one exhaustive search over a ±15 pixel window. The offsets are displayed as well as the losses.", single, results, loss_table)
    body += section("multi", "Multi-scale alignment", "The full-resolution TIFF plates are aligned with an image pyramid. The table lists NCC and L2 on raw pixel values, and NCC on edge maps.", multi, results)
    body += section("extra", "Additional plates from the collection", "Plates chosen from the Library of Congress Prokudin-Gorskii collection and aligned with the same pyramid.", extra, results)
    emir = results.get("emir")
    failure = ""
    if emir:
        failure = f'''<section id="failures">
  <h2>Failures</h2>
  <p>Emir of Bukhara did not align on raw pixel values. The result is close, but it is still visibly off. Aligning on edge maps instead, described below, fixes the alignment.</p>
</section>
<section id="bells">
  <h2>Bells and whistles</h2>
  <p>Edge-based alignment. Instead of comparing raw brightness, each pyramid level is converted to a gradient magnitude map, computed from scratch with central differences in x and y, and NCC is run on that map. Edges are in the same place on each plate regardless of the brightness of a region, so this fixes the issue that broke the original alignment.</p>
  <div class="grid">
{figure("emir", emir, "ncc", "_raw", "Emir, raw NCC (before)")}
{figure("emir", emir, PRIMARY, "", "Emir, edge NCC (after)")}
  </div>
</section>
'''
    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cs180 project 1 · colorizing the prokudin-gorskii collection</title>
<style>
  :root {{
    --bg: #faf9f6;
    --ink: #1b1b1b;
    --muted: #1b1b1b;
    --rule: #e3e0d9;
    --accent: #1b1b1b;
    --max: 960px;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 17px/1.6 Georgia, "Iowan Old Style", "Times New Roman", serif;
  }}
  header, main, footer {{ max-width: var(--max); margin: 0 auto; padding: 0 24px; }}
  header {{ padding-top: 64px; padding-bottom: 40px; border-bottom: 1px solid var(--rule); }}
  .back {{ margin: 0 0 20px; font-size: 15px; }}
  .back a {{ color: var(--muted); text-decoration: none; }}
  .back a:hover {{ color: var(--accent); }}
  .eyebrow {{
    font-size: 15px; margin: 0 0 16px;
  }}
  h1 {{ font-size: 40px; line-height: 1.15; font-weight: 400; margin: 0 0 12px; }}
  header p {{ color: var(--muted); margin: 0; }}
  nav {{ margin-top: 24px; font-size: 15px; }}
  nav a {{ color: var(--ink); text-decoration: none; margin-right: 20px; border-bottom: 1px solid var(--rule); }}
  nav a:hover {{ border-color: var(--accent); }}
  section {{ padding: 56px 0; border-bottom: 1px solid var(--rule); }}
  section:last-of-type {{ border-bottom: 0; }}
  h2 {{ font-size: 28px; font-weight: 400; margin: 0 0 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px 20px; margin-bottom: 32px; }}
  figure {{ margin: 0; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; background: #ddd; }}
  figcaption {{ margin-top: 8px; font-size: 15px; display: flex; justify-content: space-between; gap: 12px; }}
  figcaption span {{ color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .tablewrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 15px; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
  th {{ font-weight: normal; }}
  p {{ margin: 0 0 16px; }}
  footer {{ padding: 40px 24px 64px; color: var(--muted); font-size: 15px; }}
  body {{ text-transform: lowercase; }}
  pre, code {{ text-transform: none; }}
  @media (max-width: 640px) {{
    h1 {{ font-size: 30px; }}
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <p class="back"><a href="../">← All projects</a></p>
  <p class="eyebrow">CS180 · Fall 2026 · Project 1</p>
  <h1>Colorizing the Prokudin-Gorskii Photo Collection</h1>
  <p>Vikram Bhamre · UC Berkeley</p>
  <nav>
    {nav_html}
  </nav>
</header>

<main>

{body}{failure}
</main>

<footer>
  CS180 Project 1, September 2026. Source plates from the Library of Congress Prokudin-Gorskii collection.
</footer>

</body>
</html>
'''
    Path("index.html").write_text(html)


if __name__ == "__main__":
    main()
