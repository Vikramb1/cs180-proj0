from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).parent / "data"
SCALE = 0.25


def load(name, scale=SCALE):
    image = Image.open(DATA / f"{name}.jpg").convert("RGB")
    size = (round(image.width * scale), round(image.height * scale))
    return np.asarray(image.resize(size, Image.LANCZOS), float) / 255


def save(path, im):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8)).save(path, quality=92)
