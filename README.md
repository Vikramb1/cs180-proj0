# CS180 project pages

Static site. `index.html` is the landing page with one button per project.

- `proj0/` — Project 0, Becoming Friends with Your Camera. `proj0/index.html` plus `proj0/media/`.
- `proj1/` — Project 1, Colorizing the Prokudin-Gorskii Collection. `proj1/index.html`, `proj1/media/` (web-sized outputs and `results.json`), and the alignment code (`colorize.py`, which only uses library calls for reading, resizing, and writing images).

## Project 1 pipeline

    python3 -m venv .venv
    .venv/bin/pip install numpy scikit-image imageio
    cd proj1
    ../.venv/bin/python colorize.py data/*.jpg data/*.tif --metric ncc
    ../.venv/bin/python colorize.py data/emir.tif --metric ncc --suffix _raw --tag ncc
    ../.venv/bin/python colorize.py data/*.jpg data/*.tif --metric ncc --feature grad
    ../.venv/bin/python colorize.py data/*.jpg data/*.tif --metric ssd --out /tmp/ssd --web /tmp/ssd_web
    ../.venv/bin/python build_page.py

Put input plates in `proj1/data/` (gitignored). `colorize.py` writes full-resolution results to `proj1/out/` (gitignored), web-sized JPEGs and offsets to `proj1/media/`, and `build_page.py` regenerates `proj1/index.html` from `proj1/media/results.json`. Any plate not in the course set is listed under "Additional plates".

## Project 2 pipeline

    .venv/bin/pip install scipy opencv-python-headless matplotlib
    cd proj2
    ../.venv/bin/python part1.py
    ../.venv/bin/python part2.py
    ../.venv/bin/python build_page.py

`data/` holds the course images (`cameraman.png`, `taj.jpg`, `spline/`, `hybrid_starter/`). The hybrids and blends also read photos from `../proj0/media` and the full-size Project 1 outputs in `../proj1/out`.

## Publish on GitHub Pages

Push `main` and set Settings → Pages → Source: "Deploy from a branch", Branch: `main`, folder `/ (root)`. The site appears at `https://<your-username>.github.io/cs180-proj0/`.
